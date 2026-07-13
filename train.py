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
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from rclane import RCLane
from loss import RCLaneLoss
from dataset import collate, normalize_image
from decode import decode_predictions

# per-dataset default list file (relative to --data-root)
_DEFAULT_LIST = {"culane": "list/train_gt.txt", "curvelanes": "train/train.txt"}


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
        state["cuda"] = torch.cuda.get_rng_state_all()
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
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


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


def build_loader(ds, args, device, shuffle, drop_last, workers=None, batch_size=None):
    workers = args.workers if workers is None else workers
    batch_size = args.batch if batch_size is None else batch_size
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        drop_last=drop_last,
        pin_memory=(device.type == "cuda"),
    )
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
    return DataLoader(ds, **loader_kwargs)


def evaluate(model, crit, dl, device, use_amp):
    model.eval()
    running = {}
    t0 = time.time()
    with torch.no_grad():
        for imgs, targets in dl:
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
            with amp_ctx:
                out = crit(model(imgs), targets)
            for k, v in out.items():
                running[k] = running.get(k, 0.0) + v.item()
    model.train()
    n = len(dl)
    metrics = {f"val_{k}": running[k] / n for k in
               ["loss", "seg_pos", "seg_neg", "up_arrow", "down_arrow",
                "up_bound", "down_bound"]}
    metrics["val_time"] = time.time() - t0
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


def evaluate_f1(model, crit, ds, args, device, use_amp):
    model.eval()
    running = {}
    tp = fp = fn = 0
    batch = []
    metas = []
    t0 = time.time()

    def flush():
        nonlocal tp, fp, fn, batch, metas
        if not batch:
            return
        imgs, targets = collate(batch)
        imgs = imgs.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            preds = model(imgs)
            out = crit(preds, targets)
        for k, v in out.items():
            running[k] = running.get(k, 0.0) + v.item()

        decoded = decode_predictions(
            preds,
            seg_threshold=args.decode_seg_threshold,
            seed_threshold=args.decode_seed_threshold,
            seed_min_dist=args.decode_seed_min_dist,
            score_thresh=args.decode_score_thresh,
            iou_thresh=args.decode_nms_iou,
        )
        for lanes_pred, meta in zip(decoded, metas):
            gt_lanes, ow, oh = meta
            pred_lanes = _scale_pred_lanes(lanes_pred, ow, oh, ds.W, ds.H)
            gt_lanes = [np.asarray(lane, dtype=np.float32) for lane in gt_lanes
                        if len(lane) >= 2]
            matches = _match_count(
                pred_lanes, gt_lanes, ow, oh, args.f1_iou_thresh,
                args.f1_lane_width, scale=args.f1_eval_scale
            )
            tp += matches
            fp += max(0, len(pred_lanes) - matches)
            fn += max(0, len(gt_lanes) - matches)
        batch = []
        metas = []

    with torch.no_grad():
        for idx in range(len(ds)):
            img_bgr, lanes_orig, ow, oh, key = ds._load(idx)
            gt = ds._get_gt(key, lanes_orig, ow, oh)
            x = normalize_image(img_bgr, ds.W, ds.H)
            batch.append((x, _target_from_gt(gt)))
            metas.append((lanes_orig, ow, oh))
            if len(batch) >= (args.eval_batch or args.batch):
                flush()
        flush()

    denom_p = tp + fp
    denom_r = tp + fn
    precision = tp / denom_p if denom_p else 0.0
    recall = tp / denom_r if denom_r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = max(1, int(np.ceil(len(ds) / float(args.eval_batch or args.batch))))
    metrics = {f"val_{k}": running[k] / n for k in
               ["loss", "seg_pos", "seg_neg", "up_arrow", "down_arrow",
                "up_bound", "down_bound"]}
    metrics.update({
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_tp": float(tp),
        "val_fp": float(fp),
        "val_fn": float(fn),
        "val_time": time.time() - t0,
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
    ap.add_argument("--subset", type=int, default=None, help="cap #train samples")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=4,
                    help="batches prefetched per worker when workers > 0")
    ap.add_argument("--amp", action="store_true",
                    help="use CUDA automatic mixed precision")
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
                    help="validation dataloader workers; defaults to --workers")
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
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    push_subdir = args.push_subdir or f"{args.dataset}-{args.vision}"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"dataset={args.dataset} | device={device} | vision={args.vision} "
          f"| subset={args.subset} | batch={args.batch} | epochs={args.epochs} "
          f"| workers={args.workers} | amp={args.amp and device.type == 'cuda'}")

    ds = build_dataset(args)
    loader_kwargs = dict(
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
    dl = DataLoader(ds, **loader_kwargs)
    print(f"train samples: {len(ds)} | batches/epoch: {len(dl)}")
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
            eval_batches = int(np.ceil(len(eval_ds) / float(args.eval_batch or args.batch)))
            print(f"eval samples: {len(eval_ds)} | batches/eval: {eval_batches} | f1=True")
        else:
            eval_dl = build_loader(
                eval_ds,
                args,
                device,
                shuffle=False,
                drop_last=False,
                workers=args.eval_workers if args.eval_workers is not None else args.workers,
                batch_size=args.eval_batch or args.batch,
            )
            print(f"eval samples: {len(eval_ds)} | batches/eval: {len(eval_dl)}")
            if len(eval_dl) == 0:
                raise ValueError("No eval batches. Increase --eval-subset or reduce --eval-batch.")

    model = RCLane(vision=args.vision, img_size=(320, 800)).to(device)
    crit = RCLaneLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = args.epochs * len(dl)
    step = 0
    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        start_epoch, step, best_loss, ckpt = load_checkpoint(
            args.resume, model, optim, scaler, device
        )
        print(f"resumed {args.resume} | start_epoch={start_epoch} "
              f"| step={step} | best_loss={best_loss:.3f}")
        if ckpt.get("total_steps") != total_steps:
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
            print(f"warning: checkpoint monitor={resume_monitor}/{resume_monitor_mode} "
                  f"but current monitor={current_monitor}/{current_monitor_mode}; "
                  "resetting best score")
            best_loss = initial_best_score(current_monitor_mode)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        running = {}
        t0 = time.time()
        end = t0
        for it, (imgs, targets) in enumerate(dl):
            data_time = time.time() - end
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
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
            if (it + 1) % args.log_every == 0:
                avg = running["loss"] / (it + 1)
                avg_data = running["data_time"] / (it + 1)
                avg_batch = running["batch_time"] / (it + 1)
                ips = args.batch / max(avg_batch, 1e-9)
                print(f"  e{epoch} [{it+1}/{len(dl)}] loss={loss.item():.3f} "
                      f"(avg {avg:.3f}) lr={lr:.2e} data={avg_data:.2f}s "
                      f"step={avg_batch:.2f}s img/s={ips:.2f}")

        n = len(dl)
        metrics = {k: running[k] / n for k in
                   ["loss", "seg_pos", "seg_neg", "up_arrow", "down_arrow",
                    "up_bound", "down_bound"]}
        epoch_time = time.time() - t0
        metrics["epoch_time"] = epoch_time
        msg = " | ".join(f"{k}={metrics[k]:.3f}" for k in
                         ["loss", "seg_pos", "seg_neg", "up_arrow", "down_arrow",
                          "up_bound", "down_bound"])
        print(f"epoch {epoch} done in {epoch_time:.1f}s :: {msg}")
        if (eval_dl is not None or eval_ds is not None) and (epoch + 1) % args.eval_every == 0:
            if args.eval_f1:
                eval_metrics = evaluate_f1(model, crit, eval_ds, args, device, use_amp)
            else:
                eval_metrics = evaluate(model, crit, eval_dl, device, use_amp)
            metrics.update(eval_metrics)
            eval_keys = ["val_loss", "val_seg_pos", "val_seg_neg",
                         "val_up_arrow", "val_down_arrow",
                         "val_up_bound", "val_down_bound"]
            if args.eval_f1:
                eval_keys += ["val_precision", "val_recall", "val_f1"]
            eval_msg = " | ".join(f"{k}={eval_metrics[k]:.3f}" for k in eval_keys)
            print(f"eval epoch {epoch} in {eval_metrics['val_time']:.1f}s :: {eval_msg}")

        monitor_name, monitor_mode = monitor_spec(
            args, "val_loss" in metrics or "val_f1" in metrics
        )
        monitor = metrics.get(monitor_name, metrics["loss"])
        is_best = is_better(monitor, best_loss, monitor_mode)
        if is_best:
            best_loss = monitor

        ckpt = os.path.join(args.out, f"rclane_{args.vision}_e{epoch}.pth")
        save_checkpoint(ckpt, model, optim, scaler, args, epoch, step,
                        best_loss, metrics, device, total_steps, monitor_name,
                        monitor_mode)
        print(f"saved {ckpt}")
        last_ckpt = os.path.join(args.out, "last.pth")
        save_checkpoint(last_ckpt, model, optim, scaler, args, epoch, step,
                        best_loss, metrics, device, total_steps, monitor_name,
                        monitor_mode)
        print(f"saved {last_ckpt}")
        pushed = [ckpt, last_ckpt]
        if is_best:
            best_ckpt = os.path.join(args.out, "best.pth")
            save_checkpoint(best_ckpt, model, optim, scaler, args, epoch, step,
                            best_loss, metrics, device, total_steps, monitor_name,
                            monitor_mode)
            print(f"saved {best_ckpt} (best {monitor_name} {best_loss:.3f})")
            pushed.append(best_ckpt)
        if args.push_to:
            push_checkpoints(args.push_to, push_subdir, pushed)

    print("training done.")


if __name__ == "__main__":
    main()
