"""
RCLane training loop (PyTorch).

Follows the paper's recipe: AdamW, lr 6e-4, poly LR schedule, sum of the 6 loss
terms. The dataset is selected with `--dataset`; each loader lives in its own
`dataset_<name>.py` and is imported lazily, so a branch that only ships one loader
still runs.

Supports resumable training (--resume), per-epoch + best/last checkpoints, optional
validation with CULane-style lane-IoU F1 (--eval-list --eval-f1), full step/epoch
logging, and best-effort upload of each epoch's checkpoint to an HF model repo
(--push-to), so long GPU jobs survive interruption and every epoch is recoverable.

Examples:
    # CARLA: train + validate on the val split with F1, push each epoch to the Hub
    python train.py --dataset carla --data-root ../RCLane/data/dataset \
        --label label_train.json --eval-list label_val.json --eval-f1 \
        --vision b0 --epochs 20 --batch 32 --amp --device cuda \
        --push-to BanVienCorp/LaneATT-Carla-checkpoints

    # resume a job that stopped
    python train.py --dataset carla --data-root ../RCLane/data/dataset \
        --resume checkpoints/last.pth --eval-list label_val.json --eval-f1 --device cuda

    python train.py --dataset culane --data-root ../CULane \
        --train-list list/train_gt.txt --vision b0 --epochs 20 --batch 32 --device cuda

    python train.py --dataset curvelanes --data-root ../CurveLanes \
        --train-list train/train.txt --vision b0 --epochs 20 --batch 32 --device cuda
"""

import os
import time
import argparse
import random
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from rclane import RCLane
from loss import RCLaneLoss
from dataset import collate, normalize_image

# per-dataset default list file (relative to --data-root)
_DEFAULT_LIST = {"culane": "list/train_gt.txt", "curvelanes": "train/train.txt"}
_LOSS_KEYS = ("loss", "seg_pos", "seg_neg", "up_arrow", "down_arrow",
              "up_bound", "down_bound")


def _dist_ready():
    return dist.is_available() and dist.is_initialized()


def _worker_init(_worker_id):
    """Keep every DataLoader process single-threaded.

    Parallelism comes from many loader processes. Letting OpenCV/Torch create a
    thread team inside each of them would multiply 42 workers into hundreds of
    runnable threads and slow a 46-vCPU host down through contention.
    """
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class DistributedEvalSampler(Sampler):
    """Shard evaluation without the duplicate padding of DistributedSampler."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def _reduce_sums(running, count, device):
    values = [running.get(k, 0.0) for k in _LOSS_KEYS] + [float(count)]
    if _dist_ready():
        tensor = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        values = tensor.cpu().tolist()
    return dict(zip(_LOSS_KEYS, values[:-1])), values[-1]


def _reduce_max(value, device):
    if not _dist_ready():
        return value
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _resolve_amp_dtype(name, device):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _sync_model_buffers(model):
    if _dist_ready():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


def poly_lr(optimizer, base_lr, step, total_steps, power=0.9):
    progress = min(step / max(1, total_steps), 1.0)
    lr = base_lr * (1 - progress) ** power
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def _rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _load_rng_state(state, device):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and "cuda" in state:
        cuda_state = state["cuda"]
        # Older single-GPU checkpoints stored a list from get_rng_state_all().
        if isinstance(cuda_state, (list, tuple)):
            index = device.index or 0
            cuda_state = cuda_state[min(index, len(cuda_state) - 1)]
        torch.cuda.set_rng_state(cuda_state.cpu(), device=device)


def save_checkpoint(path, model, optim, scaler, args, epoch, step,
                    best_score, metrics, device, total_steps, monitor_name,
                    monitor_mode):
    state = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "step": step,
        "total_steps": total_steps,
        "best_loss": best_score,
        "best_score": best_score,
        "monitor_name": monitor_name,
        "monitor_mode": monitor_mode,
        "metrics": metrics,
        "args": vars(args),
        "rng_state": _rng_state(device),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optim, scaler, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    _load_rng_state(ckpt.get("rng_state"), device)
    start_epoch = int(ckpt.get("next_epoch", ckpt.get("epoch", -1) + 1))
    step = int(ckpt.get("step", start_epoch))
    best_score = float(ckpt.get("best_score", ckpt.get("best_loss", float("inf"))))
    return start_epoch, step, best_score, ckpt


def push_checkpoints(repo, subdir, files):
    """Upload the given checkpoint files to an HF model repo (best-effort).

    Runs after every epoch so each epoch's weights land on the Hub while the job
    is still going -- if the job dies you keep every epoch you have already paid
    for. Auth uses the HF_TOKEN env var (forwarded via `hf jobs run --secrets
    HF_TOKEN`). Failures are logged and never interrupt training.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [push] huggingface_hub not installed; skipping upload")
        return
    api = HfApi()
    try:
        api.create_repo(repo, repo_type="model", exist_ok=True)
    except Exception as e:  # noqa: BLE001 -- best-effort, keep training alive
        print(f"  [push] create_repo warning: {e}")
    for f in files:
        if not f or not os.path.exists(f):
            continue
        try:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"{subdir}/{os.path.basename(f)}",
                repo_id=repo,
                repo_type="model",
            )
            print(f"  [push] uploaded {os.path.basename(f)} -> {repo}/{subdir}")
        except Exception as e:  # noqa: BLE001
            print(f"  [push] upload of {f} failed: {e}")


def monitor_spec(args, has_eval):
    if args.eval_f1:
        return "val_f1", "max"
    if has_eval:
        return "val_loss", "min"
    return "loss", "min"


def initial_best_score(mode):
    return -float("inf") if mode == "max" else float("inf")


def is_better(value, best, mode):
    return value > best if mode == "max" else value < best


def build_dataset(args):
    """Lazily import and build the selected dataset."""
    return build_dataset_split(
        dataset=args.dataset,
        data_root=args.data_root,
        label=args.label,
        list_file=args.train_list or _DEFAULT_LIST.get(args.dataset),
        cache_dir=args.cache_dir,
        max_samples=args.subset,
    )


def build_dataset_split(dataset, data_root, label=None, list_file=None,
                        cache_dir=None, max_samples=None):
    """Build a dataset split. `list_file` is relative to data_root.

    CARLA is annotated as one JSON-lines label file per split (label_train.json,
    label_val.json, label_test.json). The train split passes its file via `label`;
    the eval split passes its file via `list_file` (from --eval-list), so a
    validation split reads label_val.json instead of reusing the training label.
    """
    if dataset == "carla":
        from dataset_carla import CarlaLaneDataset
        carla_label = list_file or label
        if not carla_label:
            raise ValueError("carla requires a label file (--label / --eval-list)")
        return CarlaLaneDataset(
            label_json=os.path.join(data_root, carla_label),
            data_root=data_root,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
    if not list_file:
        raise ValueError(f"{dataset} requires a list file")
    split_file = os.path.join(data_root, list_file)
    if dataset == "culane":
        from dataset_culane import CULaneDataset
        cls = CULaneDataset
    else:  # curvelanes
        from dataset_curvelanes import CurveLanesDataset
        cls = CurveLanesDataset
    return cls(list_file=split_file, data_root=data_root,
               cache_dir=cache_dir, max_samples=max_samples)


def build_loader(ds, args, device, shuffle, drop_last, workers=None,
                 batch_size=None, sampler=None):
    workers = args.workers if workers is None else workers
    batch_size = args.batch if batch_size is None else batch_size
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate,
        drop_last=drop_last,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init,
    )
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
    return DataLoader(ds, **loader_kwargs)


def evaluate(model, crit, dl, device, use_amp, amp_dtype):
    model.eval()
    running = {}
    n_samples = 0
    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in dl:
            batch_size = imgs.shape[0]
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            amp_ctx = torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ) if use_amp else nullcontext()
            with amp_ctx:
                out = crit(model(imgs), targets)
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + v.item() * batch_size
            n_samples += batch_size
    running, n_samples = _reduce_sums(running, n_samples, device)
    model.train()
    n_samples = max(1.0, n_samples)
    metrics = {f"val_{k}": running[k] / n_samples for k in _LOSS_KEYS}
    metrics["val_time"] = _reduce_max(time.time() - t0, device)
    return metrics


def _target_from_gt(gt):
    return {
        "seg_map": torch.from_numpy(gt["seg_map"]).long(),
        "up_arrow": torch.from_numpy(gt["up_arrow"]).float(),
        "down_arrow": torch.from_numpy(gt["down_arrow"]).float(),
        "up_bound": torch.from_numpy(gt["up_bound"]).float(),
        "down_bound": torch.from_numpy(gt["down_bound"]).float(),
    }


def _rasterize_lanes(lanes, width, height, lane_width):
    """Rasterize each lane once into a boolean mask; return masks + their areas.

    Done once per lane (P + G rasterizations), not once per pred/gt pair, and on a
    downscaled canvas -- lane-IoU is a ratio, so scaling pred and gt (and the line
    width) by the same factor leaves it essentially unchanged while cutting the
    pixel count quadratically. This is the hot path of F1 eval; on an untrained
    model `decode` emits many spurious lanes, so P*G full-res mask ops dominate.
    """
    masks, areas = [], []
    for pts in lanes:
        mask = np.zeros((height, width), np.uint8)
        p = np.asarray(pts, dtype=np.float32)
        if p.ndim == 2 and len(p) >= 2:
            p[:, 0] = np.clip(p[:, 0], 0, width - 1)
            p[:, 1] = np.clip(p[:, 1], 0, height - 1)
            cv2.polylines(mask, [p.astype(np.int32)], False, 1, lane_width)
        m = mask.astype(bool)
        masks.append(m)
        areas.append(int(m.sum()))
    return masks, areas


def _match_count(pred_lanes, gt_lanes, width, height, iou_thr, lane_width,
                 scale=0.25):
    if not pred_lanes or not gt_lanes:
        return 0
    w = max(1, int(round(width * scale)))
    h = max(1, int(round(height * scale)))
    lw = max(1, int(round(lane_width * scale)))
    pred_s = [np.asarray(l, dtype=np.float32) * scale for l in pred_lanes]
    gt_s = [np.asarray(l, dtype=np.float32) * scale for l in gt_lanes]
    pred_masks, pred_area = _rasterize_lanes(pred_s, w, h, lw)
    gt_masks, gt_area = _rasterize_lanes(gt_s, w, h, lw)

    graph = [[] for _ in pred_lanes]
    for pi in range(len(pred_lanes)):
        if pred_area[pi] == 0:
            continue
        for gi in range(len(gt_lanes)):
            if gt_area[gi] == 0:
                continue
            inter = int(np.count_nonzero(pred_masks[pi] & gt_masks[gi]))
            if inter == 0:
                continue
            union = pred_area[pi] + gt_area[gi] - inter
            if union > 0 and inter / union >= iou_thr:
                graph[pi].append(gi)

    match_gt = [-1] * len(gt_lanes)

    def dfs(pi, seen):
        for gi in graph[pi]:
            if seen[gi]:
                continue
            seen[gi] = True
            if match_gt[gi] == -1 or dfs(match_gt[gi], seen):
                match_gt[gi] = pi
                return True
        return False

    matched = 0
    for pi in range(len(pred_lanes)):
        if dfs(pi, [False] * len(gt_lanes)):
            matched += 1
    return matched


def _scale_pred_lanes(decoded_lanes, ow, oh, model_w, model_h):
    sx, sy = ow / model_w, oh / model_h
    lanes = []
    for lane in decoded_lanes:
        xy = lane.xy()
        if len(xy) < 2:
            continue
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        lanes.append(xy)
    return lanes


class _F1EvalDataset(Dataset):
    """Wrap a LaneEncodeDataset so a DataLoader can parallelize image load + GT
    encode across workers, while still returning the original-space lanes that
    F1 matching needs (the base __getitem__ drops them)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_bgr, lanes_orig, ow, oh, key = self.ds._load(idx)
        gt = self.ds._get_gt(key, lanes_orig, ow, oh)
        x = normalize_image(img_bgr, self.ds.W, self.ds.H)
        lanes = [np.asarray(l, dtype=np.float32) for l in lanes_orig]
        return x, _target_from_gt(gt), (lanes, ow, oh)


class _CacheWarmDataset(Dataset):
    """Compute only GT cache entries, one sample per DataLoader task."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_bgr, lanes_orig, ow, oh, key = self.ds._load(idx)
        del img_bgr
        existed = os.path.exists(self.ds._cache_path(key))
        self.ds._get_gt(key, lanes_orig, ow, oh)
        return int(not existed)


def warm_cache(ds, args, device, rank, world_size, name):
    """Fan GT generation over all loader workers before large GPU batches.

    Normal auto-batching assigns a complete batch to one worker. On a cold
    cache that makes the first batch look hung while a single worker encodes
    dozens of samples. A batch-size-one warming pass exposes one cache item per
    task, keeping all 42 loader processes busy on the H200x2 host.
    """
    if not args.warm_cache or not ds.cache_dir:
        return
    warm_ds = _CacheWarmDataset(ds)
    sampler = DistributedEvalSampler(warm_ds, world_size, rank) \
        if world_size > 1 else None
    kwargs = dict(
        dataset=warm_ds,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=_worker_init,
    )
    if args.workers > 0:
        kwargs["prefetch_factor"] = max(2, args.prefetch)
    loader = DataLoader(**kwargs)
    t0 = time.time()
    created = seen = 0
    for batch in loader:
        created += int(batch.sum().item())
        seen += len(batch)
        if rank == 0 and seen % 2000 == 0:
            print(f"  warm {name}: {seen}/{len(loader)} local samples")
    totals = torch.tensor([created, seen], dtype=torch.float64, device=device)
    if _dist_ready():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    elapsed = _reduce_max(time.time() - t0, device)
    if rank == 0:
        print(f"warm {name} cache: {int(totals[1].item())} checked, "
              f"{int(totals[0].item())} created in {elapsed:.1f}s")


def _f1_collate(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    keys = batch[0][1].keys()
    targets = {k: torch.stack([b[1][k] for b in batch], 0) for k in keys}
    metas = [b[2] for b in batch]
    return imgs, targets, metas


def _f1_decode_match(payload):
    """Worker (separate process): decode one image's maps into lanes and match
    them against GT, returning (matches, n_pred, n_gt).

    This is the eval bottleneck -- relay-chain decode + IoU matching, pure
    numpy/Python and CPU-bound. Fanning it out across processes is what lets a
    46-vCPU box actually use its cores; no CUDA is touched here."""
    (seg, ua, da, ub, db, gt_lanes, ow, oh, model_w, model_h,
     decode_kwargs, iou_thr, lane_width, scale) = payload
    from decode import decode
    lanes = decode(seg, ua, da, ub, db, **decode_kwargs)
    pred_lanes = _scale_pred_lanes(lanes, ow, oh, model_w, model_h)
    gts = [g for g in gt_lanes if len(g) >= 2]
    matches = _match_count(pred_lanes, gts, ow, oh, iou_thr, lane_width, scale)
    return matches, len(pred_lanes), len(gts)


def evaluate_f1(model, crit, ds, args, device, use_amp, amp_dtype, rank,
                world_size):
    """Distributed GPU inference overlapped with CPU decode + IoU matching."""
    model.eval()
    running = {}
    n_samples = 0
    t0 = time.time()

    eval_ds = _F1EvalDataset(ds)
    load_workers = args.eval_workers if args.eval_workers is not None else args.workers
    sampler = DistributedEvalSampler(eval_ds, world_size, rank) \
        if world_size > 1 else None
    loader_kwargs = dict(
        batch_size=(args.eval_batch or args.batch),
        shuffle=False,
        sampler=sampler,
        num_workers=load_workers,
        collate_fn=_f1_collate,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init,
    )
    if load_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
    loader = DataLoader(eval_ds, **loader_kwargs)

    decode_kwargs = dict(
        seg_threshold=args.decode_seg_threshold,
        seed_threshold=args.decode_seed_threshold,
        seed_min_dist=args.decode_seed_min_dist,
        score_thresh=args.decode_score_thresh,
        iou_thresh=args.decode_nms_iou,
        max_seeds=args.decode_max_seeds,
        nms_max_lanes=args.decode_nms_max_lanes,
        nms_scale=args.decode_nms_scale,
    )

    tp = fp = fn = 0
    n_proc = args.eval_decode_workers
    if n_proc is None:
        n_proc = max(1, ((os.cpu_count() or 1) - world_size) // world_size)
    pending = deque()
    max_pending = max(1, n_proc * 2)

    def collect(result):
        nonlocal tp, fp, fn
        matches, n_pred, n_gt = result
        tp += matches
        fp += max(0, n_pred - matches)
        fn += max(0, n_gt - matches)

    # Spawn is intentional: forking after CUDA initialization can deadlock.
    pool_ctx = ProcessPoolExecutor(
        max_workers=n_proc,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(0,),
    ) if n_proc > 1 else nullcontext(None)
    with pool_ctx as executor, torch.no_grad():
        for imgs, targets, metas in loader:
            batch_size = imgs.shape[0]
            imgs = imgs.to(device, non_blocking=True)
            tgt = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            amp_ctx = torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ) if use_amp else nullcontext()
            with amp_ctx:
                preds = model(imgs)
                out = crit(preds, tgt)
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + v.item() * batch_size
            n_samples += batch_size

            seg = torch.softmax(preds["seg_map"], dim=1)[:, 1].float().cpu().numpy()
            ua = preds["up_arrow"].float().cpu().numpy()
            da = preds["down_arrow"].float().cpu().numpy()
            ub = preds["up_bound"].float().cpu().numpy()
            db = preds["down_bound"].float().cpu().numpy()
            for b, (gt_lanes, ow, oh) in enumerate(metas):
                payload = (seg[b], ua[b], da[b], ub[b], db[b], gt_lanes,
                           ow, oh, ds.W, ds.H, decode_kwargs,
                           args.f1_iou_thresh, args.f1_lane_width,
                           args.f1_eval_scale)
                if executor is None:
                    collect(_f1_decode_match(payload))
                else:
                    pending.append(executor.submit(_f1_decode_match, payload))
                    if len(pending) >= max_pending:
                        collect(pending.popleft().result())
        while pending:
            collect(pending.popleft().result())

    running, n_samples = _reduce_sums(running, n_samples, device)
    counts = torch.tensor([tp, fp, fn], dtype=torch.float64, device=device)
    if _dist_ready():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    tp, fp, fn = (int(v) for v in counts.cpu().tolist())

    denom_p = tp + fp
    denom_r = tp + fn
    precision = tp / denom_p if denom_p else 0.0
    recall = tp / denom_r if denom_r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_samples = max(1.0, n_samples)
    metrics = {f"val_{k}": running[k] / n_samples for k in _LOSS_KEYS}
    metrics.update({
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_tp": float(tp),
        "val_fp": float(fp),
        "val_fn": float(fn),
        "val_time": _reduce_max(time.time() - t0, device),
    })
    model.train()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="carla",
                    choices=["carla", "culane", "curvelanes"])
    ap.add_argument("--data-root", default="../RCLane/data/dataset",
                    help="CARLA dataset dir, or CULane/CurveLanes root")
    ap.add_argument("--label", default="label_train.json",
                    help="CARLA label file, relative to data-root")
    ap.add_argument("--train-list", default=None,
                    help="CULane/CurveLanes list file, relative to data-root")
    ap.add_argument("--vision", default="b0", choices=["b0", "b1", "b2"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--subset", type=int, default=None, help="cap #train samples")
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader workers per DDP process/GPU")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="batches prefetched per worker when workers > 0")
    ap.add_argument("--warm-cache", action="store_true",
                    help="parallelize cold GT cache generation one sample/task "
                         "before training")
    ap.add_argument("--amp", action="store_true",
                    help="use CUDA automatic mixed precision")
    ap.add_argument("--amp-dtype", default="auto",
                    choices=["auto", "float16", "bfloat16"],
                    help="AMP dtype; auto prefers bfloat16 on H100/H200")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir", default="./gt_cache_train")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path to resume from, e.g. checkpoints/last.pth")
    ap.add_argument("--push-to", default=None,
                    help="HF model repo (e.g. BanVienCorp/LaneATT-Carla-checkpoints) "
                         "to upload each epoch's checkpoint to; needs HF_TOKEN in env")
    ap.add_argument("--push-subdir", default=None,
                    help="folder inside --push-to repo; defaults to <dataset>-<vision>")
    ap.add_argument("--eval-list", default=None,
                    help="validation split file, relative to data-root "
                         "(CARLA: label_val.json; CULane/CurveLanes: a list file)")
    ap.add_argument("--eval-subset", type=int, default=None,
                    help="cap #validation samples")
    ap.add_argument("--eval-batch", type=int, default=None,
                    help="validation batch size; defaults to --batch")
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="validation dataloader workers (load + GT encode); "
                         "defaults to --workers")
    ap.add_argument("--eval-decode-workers", type=int, default=None,
                    help="processes for the parallel F1 decode + IoU matching; "
                         "count is per DDP rank/GPU")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run validation every N epochs when --eval-list is set")
    ap.add_argument("--eval-f1", action="store_true",
                    help="compute CULane-style lane IoU F1 on the eval split")
    ap.add_argument("--f1-iou-thresh", type=float, default=0.5)
    ap.add_argument("--f1-lane-width", type=int, default=30)
    ap.add_argument("--f1-eval-scale", type=float, default=0.25,
                    help="downscale factor for the F1 IoU raster canvas "
                         "(0.25 = 16x fewer pixels, ~14x faster; 1.0 = full res)")
    ap.add_argument("--decode-seg-threshold", type=float, default=0.5)
    ap.add_argument("--decode-seed-threshold", type=float, default=None)
    ap.add_argument("--decode-seed-min-dist", type=int, default=2)
    ap.add_argument("--decode-score-thresh", type=float, default=0.10)
    ap.add_argument("--decode-nms-iou", type=float, default=0.5)
    ap.add_argument("--decode-max-seeds", type=int, default=1024,
                    help="cap relay-chain seeds per image before decoding")
    ap.add_argument("--decode-nms-max-lanes", type=int, default=128,
                    help="keep top-scoring candidates before lane IoU NMS")
    ap.add_argument("--decode-nms-scale", type=float, default=0.25,
                    help="downscale factor for cached lane-NMS masks")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    requested_device = torch.device(args.device)
    if distributed:
        backend = "nccl" if requested_device.type == "cuda" else "gloo"
        if requested_device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = requested_device
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = requested_device
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
            torch.cuda.set_device(device)
    is_main = rank == 0

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if is_main:
        os.makedirs(args.out, exist_ok=True)
    if distributed:
        dist.barrier()
    push_subdir = args.push_subdir or f"{args.dataset}-{args.vision}"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(args.amp_dtype, device)
    if is_main:
        dtype_name = str(amp_dtype).removeprefix("torch.") if use_amp else "float32"
        print(f"dataset={args.dataset} | device={device.type} x{world_size} "
              f"| vision={args.vision} | subset={args.subset} "
              f"| batch/gpu={args.batch} | global_batch={args.batch * world_size} "
              f"| epochs={args.epochs} | workers/gpu={args.workers} "
              f"| amp={dtype_name}")

    ds = build_dataset(args)
    train_sampler = DistributedSampler(
        ds, num_replicas=world_size, rank=rank, shuffle=True,
        seed=args.seed, drop_last=True,
    ) if distributed else None
    dl = build_loader(ds, args, device, shuffle=True, drop_last=True,
                      sampler=train_sampler)
    if is_main:
        print(f"train samples: {len(ds)} | optimizer steps/epoch: {len(dl)}")
    if len(dl) == 0:
        raise ValueError("No training batches. Reduce --batch or increase --subset/dataset size.")
    eval_dl = None
    eval_ds = None
    if args.eval_list:
        eval_ds = build_dataset_split(
            dataset=args.dataset,
            data_root=args.data_root,
            label=args.label,
            list_file=args.eval_list,
            cache_dir=args.cache_dir,
            max_samples=args.eval_subset,
        )
        if args.eval_f1:
            eval_batches = int(np.ceil(
                len(eval_ds) / float((args.eval_batch or args.batch) * world_size)
            ))
            if is_main:
                print(f"eval samples: {len(eval_ds)} | distributed batches/eval: "
                      f"{eval_batches} | f1=True")
        else:
            eval_sampler = DistributedEvalSampler(eval_ds, world_size, rank) \
                if distributed else None
            eval_dl = build_loader(
                eval_ds,
                args,
                device,
                shuffle=False,
                drop_last=False,
                workers=args.eval_workers if args.eval_workers is not None else args.workers,
                batch_size=args.eval_batch or args.batch,
                sampler=eval_sampler,
            )
            if is_main:
                print(f"eval samples: {len(eval_ds)} | batches/rank: {len(eval_dl)}")
            if len(eval_dl) == 0:
                raise ValueError("No eval batches. Increase --eval-subset or reduce --eval-batch.")

    warm_cache(ds, args, device, rank, world_size, "train")
    if eval_ds is not None:
        warm_cache(eval_ds, args, device, rank, world_size, "eval")

    base_model = RCLane(vision=args.vision, img_size=(320, 800)).to(device)
    crit = RCLaneLoss()
    optim = torch.optim.AdamW(base_model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    total_steps = args.epochs * len(dl)
    step = 0
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, step, best_loss, ckpt = load_checkpoint(
            args.resume, base_model, optim, scaler, device
        )
        if is_main:
            print(f"resumed {args.resume} | start_epoch={start_epoch} "
                  f"| step={step} | best_loss={best_loss:.3f}")
        if is_main and ckpt.get("total_steps") != total_steps:
            print(f"warning: checkpoint total_steps={ckpt.get('total_steps')} "
                  f"but current total_steps={total_steps}")
        current_monitor, current_monitor_mode = monitor_spec(
            args, eval_dl is not None or eval_ds is not None
        )
        resume_monitor = ckpt.get(
            "monitor_name",
            "val_loss" if "val_loss" in ckpt.get("metrics", {}) else "loss",
        )
        resume_monitor_mode = ckpt.get("monitor_mode", "min")
        if resume_monitor != current_monitor or resume_monitor_mode != current_monitor_mode:
            if is_main:
                print(f"warning: checkpoint monitor={resume_monitor}/{resume_monitor_mode} "
                      f"but current monitor={current_monitor}/{current_monitor_mode}; "
                      "resetting best score")
            best_loss = initial_best_score(current_monitor_mode)

    model = DDP(
        base_model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        static_graph=True,
    ) if distributed else base_model
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = {}
        t0 = time.time()
        end = t0
        for it, (imgs, targets) in enumerate(dl):
            data_time = time.time() - end
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            amp_ctx = torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ) if use_amp else nullcontext()
            with amp_ctx:
                preds = model(imgs)
                out = crit(preds, targets)
            loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            lr = poly_lr(optim, args.lr, step, total_steps)
            step += 1
            batch_time = time.time() - end
            end = time.time()

            for k, v in out.items():
                running[k] = running.get(k, 0.0) + v.item()
            running["data_time"] = running.get("data_time", 0.0) + data_time
            running["batch_time"] = running.get("batch_time", 0.0) + batch_time
            if is_main and (it + 1) % args.log_every == 0:
                avg = running["loss"] / (it + 1)
                avg_data = running["data_time"] / (it + 1)
                avg_batch = running["batch_time"] / (it + 1)
                ips = args.batch * world_size / max(avg_batch, 1e-9)
                print(f"  e{epoch} [{it+1}/{len(dl)}] loss={loss.item():.3f} "
                      f"(avg {avg:.3f}) lr={lr:.2e} data={avg_data:.2f}s "
                      f"step={avg_batch:.2f}s img/s={ips:.2f}")

        reduced, n = _reduce_sums(running, len(dl), device)
        metrics = {k: reduced[k] / max(1.0, n) for k in _LOSS_KEYS}
        epoch_time = _reduce_max(time.time() - t0, device)
        metrics["epoch_time"] = epoch_time
        msg = " | ".join(f"{k}={metrics[k]:.3f}" for k in
                         _LOSS_KEYS)
        if is_main:
            print(f"epoch {epoch} done in {epoch_time:.1f}s :: {msg}")
        if (eval_dl is not None or eval_ds is not None) and (epoch + 1) % args.eval_every == 0:
            _sync_model_buffers(base_model)
            if args.eval_f1:
                eval_metrics = evaluate_f1(
                    base_model, crit, eval_ds, args, device, use_amp,
                    amp_dtype, rank, world_size,
                )
            else:
                eval_metrics = evaluate(
                    base_model, crit, eval_dl, device, use_amp, amp_dtype
                )
            metrics.update(eval_metrics)
            eval_keys = ["val_loss", "val_seg_pos", "val_seg_neg",
                         "val_up_arrow", "val_down_arrow",
                         "val_up_bound", "val_down_bound"]
            if args.eval_f1:
                eval_keys += ["val_precision", "val_recall", "val_f1"]
            eval_msg = " | ".join(f"{k}={eval_metrics[k]:.3f}" for k in eval_keys)
            if is_main:
                print(f"eval epoch {epoch} in {eval_metrics['val_time']:.1f}s :: "
                      f"{eval_msg}")

        monitor_name, monitor_mode = monitor_spec(
            args, "val_loss" in metrics or "val_f1" in metrics
        )
        monitor = metrics.get(monitor_name, metrics["loss"])
        is_best = is_better(monitor, best_loss, monitor_mode)
        if is_best:
            best_loss = monitor

        if is_main:
            ckpt = os.path.join(args.out, f"rclane_{args.vision}_e{epoch}.pth")
            save_checkpoint(ckpt, base_model, optim, scaler, args, epoch, step,
                            best_loss, metrics, device, total_steps, monitor_name,
                            monitor_mode)
            print(f"saved {ckpt}")
            last_ckpt = os.path.join(args.out, "last.pth")
            save_checkpoint(last_ckpt, base_model, optim, scaler, args, epoch, step,
                            best_loss, metrics, device, total_steps, monitor_name,
                            monitor_mode)
            print(f"saved {last_ckpt}")
            pushed = [ckpt, last_ckpt]
            if is_best:
                best_ckpt = os.path.join(args.out, "best.pth")
                save_checkpoint(best_ckpt, base_model, optim, scaler, args, epoch, step,
                                best_loss, metrics, device, total_steps, monitor_name,
                                monitor_mode)
                print(f"saved {best_ckpt} (best {monitor_name} {best_loss:.3f})")
                pushed.append(best_ckpt)
            if args.push_to:
                push_checkpoints(args.push_to, push_subdir, pushed)
        if distributed:
            dist.barrier()

    if is_main:
        print("training done.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
