"""
RCLane training loop (PyTorch).

Follows the paper's recipe: AdamW, lr 6e-4, poly LR schedule, sum of the 6 loss
terms. The dataset is selected with `--dataset`; each loader lives in its own
`dataset_<name>.py` and is imported lazily, so a branch that only ships one loader
still runs.

Examples:
    python train.py --dataset carla --data-root ../RCLane/data/dataset \
        --vision b0 --epochs 20 --batch 32 --device cuda

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

import numpy as np
import torch
from torch.utils.data import DataLoader

from rclane import RCLane
from loss import RCLaneLoss
from dataset import collate

# per-dataset default list file (relative to --data-root)
_DEFAULT_LIST = {"culane": "list/train_gt.txt", "curvelanes": "train/train.txt"}


def poly_lr(optimizer, base_lr, step, total_steps, power=0.9):
    lr = base_lr * (1 - step / max(1, total_steps)) ** power
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
                    best_loss, metrics, device, total_steps):
    state = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "step": step,
        "total_steps": total_steps,
        "best_loss": best_loss,
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
    best_loss = float(ckpt.get("best_loss", float("inf")))
    return start_epoch, step, best_loss, ckpt


def build_dataset(args):
    """Lazily import and build the selected dataset."""
    if args.dataset == "carla":
        from dataset_carla import CarlaLaneDataset
        return CarlaLaneDataset(
            label_json=os.path.join(args.data_root, args.label),
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            max_samples=args.subset,
        )
    list_file = os.path.join(args.data_root, args.train_list or _DEFAULT_LIST[args.dataset])
    if args.dataset == "culane":
        from dataset_culane import CULaneDataset
        cls = CULaneDataset
    else:  # curvelanes
        from dataset_curvelanes import CurveLanesDataset
        cls = CurveLanesDataset
    return cls(list_file=list_file, data_root=args.data_root,
               cache_dir=args.cache_dir, max_samples=args.subset)


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
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
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

        is_best = metrics["loss"] < best_loss
        if is_best:
            best_loss = metrics["loss"]

        ckpt = os.path.join(args.out, f"rclane_{args.vision}_e{epoch}.pth")
        save_checkpoint(ckpt, model, optim, scaler, args, epoch, step,
                        best_loss, metrics, device, total_steps)
        print(f"saved {ckpt}")
        last_ckpt = os.path.join(args.out, "last.pth")
        save_checkpoint(last_ckpt, model, optim, scaler, args, epoch, step,
                        best_loss, metrics, device, total_steps)
        print(f"saved {last_ckpt}")
        if is_best:
            best_ckpt = os.path.join(args.out, "best.pth")
            save_checkpoint(best_ckpt, model, optim, scaler, args, epoch, step,
                            best_loss, metrics, device, total_steps)
            print(f"saved {best_ckpt} (best train loss {best_loss:.3f})")

    print("training done.")


if __name__ == "__main__":
    main()
