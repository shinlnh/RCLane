"""
RCLane training loop (PyTorch).

Follows the paper's recipe: AdamW, lr 6e-4, poly LR schedule, sum of the 6 loss
terms. Kept intentionally small and readable -- this is the driver that ties the
network, encode-based dataset, and loss together.

Example:
    python train.py --data-root ../RCLane/data/dataset \
        --vision b0 --subset 64 --epochs 3 --batch 2 --device cpu
"""

import os
import time
import argparse

import torch
from torch.utils.data import DataLoader

from rclane import RCLane
from loss import RCLaneLoss
from dataset import CarlaLaneDataset, collate
from dataset_culane import CULaneDataset


def poly_lr(optimizer, base_lr, step, total_steps, power=0.9):
    lr = base_lr * (1 - step / max(1, total_steps)) ** power
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="culane", choices=["culane", "carla"])
    ap.add_argument("--data-root", default="../RCLane/data/CULane",
                    help="CULane root, or the CARLA 'dataset' dir")
    ap.add_argument("--train-list", default="list/train_gt.txt",
                    help="CULane list file (relative to data-root), unused for carla")
    ap.add_argument("--vision", default="b0", choices=["b0", "b1", "b2"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--subset", type=int, default=None, help="cap #train samples")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir", default="./gt_cache_train")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    print(f"dataset={args.dataset} | device={device} | vision={args.vision} "
          f"| subset={args.subset} | batch={args.batch} | epochs={args.epochs}")

    if args.dataset == "culane":
        ds = CULaneDataset(
            list_file=os.path.join(args.data_root, args.train_list),
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            max_samples=args.subset,
        )
    else:
        ds = CarlaLaneDataset(
            label_json=os.path.join(args.data_root, "label_train.json"),
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            max_samples=args.subset,
        )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, pin_memory=(device.type == "cuda"))
    print(f"train samples: {len(ds)} | batches/epoch: {len(dl)}")

    model = RCLane(vision=args.vision, img_size=(320, 800)).to(device)
    crit = RCLaneLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * len(dl)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        running = {}
        t0 = time.time()
        for it, (imgs, targets) in enumerate(dl):
            imgs = imgs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            preds = model(imgs)
            out = crit(preds, targets)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            optim.step()
            lr = poly_lr(optim, args.lr, step, total_steps)
            step += 1

            for k, v in out.items():
                running[k] = running.get(k, 0.0) + v.item()
            if (it + 1) % args.log_every == 0:
                avg = running["loss"] / (it + 1)
                print(f"  e{epoch} [{it+1}/{len(dl)}] loss={loss.item():.3f} "
                      f"(avg {avg:.3f}) lr={lr:.2e}")

        n = len(dl)
        msg = " | ".join(f"{k}={running[k]/n:.3f}" for k in
                         ["loss", "seg_pos", "seg_neg", "up_arrow", "down_arrow",
                          "up_bound", "down_bound"])
        print(f"epoch {epoch} done in {time.time()-t0:.1f}s :: {msg}")

        ckpt = os.path.join(args.out, f"rclane_{args.vision}_e{epoch}.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt)
        print(f"saved {ckpt}")

    print("training done.")


if __name__ == "__main__":
    main()
