"""Evaluate RCLane checkpoints on one CARLA split.

F1-only evaluation reads sparse lane annotations directly. With ``--with-loss``,
the dense GT cache is warmed once and shared by every checkpoint. Results are
written atomically after each model so a long sweep remains inspectable if it is
interrupted.
"""

import argparse
import gc
import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
import torch

from loss import RCLaneLoss
from rclane import RCLane
from train import build_dataset_split, evaluate_f1, warm_cache


def checkpoint_order(path):
    match = re.search(r"_e(\d+)\.pth$", os.path.basename(path))
    return (int(match.group(1)) if match else 10**9, os.path.basename(path))


def save_results(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def state_dict_digest(state_dict):
    """Content hash used to avoid re-evaluating byte-identical model weights."""
    digest = hashlib.sha256()
    for name, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        # View raw storage as bytes so uncommon dtypes such as bfloat16 remain
        # hashable even when NumPy cannot represent them directly.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument(
        "--eval-list",
        required=True,
        help="CARLA JSONL label file, relative to --data-root",
    )
    parser.add_argument(
        "--checkpoints",
        required=True,
        help="checkpoint path or glob pattern (quote globs in the shell)",
    )
    parser.add_argument("--output", default="eval_results/checkpoint_eval.json")
    parser.add_argument("--cache-dir", default="gt_cache_eval")
    parser.add_argument("--vision", default="b0", choices=["b0", "b1", "b2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=11,
                        help="workers used once to warm the GT cache")
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--eval-decode-workers", type=int, default=7)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--eval-log-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--with-loss", action="store_true",
        help="also encode dense GT and calculate loss; slower and unnecessary for F1 ranking",
    )
    parser.add_argument("--f1-iou-thresh", type=float, default=0.5)
    parser.add_argument("--f1-lane-width", type=int, default=30)
    parser.add_argument("--f1-eval-scale", type=float, default=0.25)
    parser.add_argument("--decode-seg-threshold", type=float, default=0.5)
    parser.add_argument("--decode-seed-threshold", type=float, default=None)
    parser.add_argument("--decode-seed-min-dist", type=int, default=2)
    parser.add_argument("--decode-score-thresh", type=float, default=0.10)
    parser.add_argument("--decode-nms-iou", type=float, default=0.5)
    parser.add_argument("--decode-max-seeds", type=int, default=1024)
    parser.add_argument("--decode-nms-max-lanes", type=int, default=128)
    parser.add_argument("--decode-nms-scale", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    args.warm_cache = args.with_loss
    args.eval_f1 = True
    args.eval_skip_loss = not args.with_loss
    args.eval_subset = args.max_samples

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        device = torch.device("cuda", device.index or 0)
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    checkpoints = sorted(glob.glob(args.checkpoints), key=checkpoint_order)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints match {args.checkpoints!r}")

    dataset = build_dataset_split(
        dataset="carla",
        data_root=args.data_root,
        list_file=args.eval_list,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
    )
    print(f"dataset={args.eval_list} | samples={len(dataset)}")
    print(f"checkpoints={len(checkpoints)} | device={device} | batch={args.eval_batch}")
    warm_cache(dataset, args, device, rank=0, world_size=1, name="eval")

    model = RCLane(vision=args.vision, img_size=(320, 800)).to(device)
    criterion = RCLaneLoss()
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = torch.float16
    results = []
    seen_models = {}

    for index, checkpoint_path in enumerate(checkpoints, 1):
        started = time.time()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        epoch = int(checkpoint.get("epoch", -1))
        step = int(checkpoint.get("step", -1))
        model_digest = state_dict_digest(checkpoint["model"])
        name = os.path.basename(checkpoint_path)
        if model_digest in seen_models:
            source = seen_models[model_digest]
            record = {
                **source,
                "checkpoint": name,
                "path": os.path.abspath(checkpoint_path),
                "epoch": epoch,
                "step": step,
                "model_sha256": model_digest,
                "reused_from": source["checkpoint"],
                "wall_time": time.time() - started,
            }
            results.append(record)
            save_results(args.output, results)
            print(
                f"[{index}/{len(checkpoints)}] {name}: identical weights to "
                f"{source['checkpoint']}; reused F1={record['val_f1']:.6f}"
            )
            del checkpoint
            gc.collect()
            continue

        model.load_state_dict(checkpoint["model"])
        del checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        print(f"[{index}/{len(checkpoints)}] evaluating {name} (epoch={epoch})")
        metrics = evaluate_f1(
            model, criterion, dataset, args, device, use_amp, amp_dtype,
            rank=0, world_size=1,
        )
        record = {
            "checkpoint": name,
            "path": os.path.abspath(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "model_sha256": model_digest,
            **{key: float(value) for key, value in metrics.items()},
            "wall_time": time.time() - started,
        }
        if device.type == "cuda":
            record["peak_cuda_memory_gib"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )
        results.append(record)
        seen_models[model_digest] = record
        save_results(args.output, results)
        loss_text = (
            f"{record['val_loss']:.6f}" if "val_loss" in record else "skipped"
        )
        print(
            f"  {name}: F1={record['val_f1']:.6f} "
            f"P={record['val_precision']:.6f} R={record['val_recall']:.6f} "
            f"loss={loss_text} time={record['wall_time']:.1f}s "
            f"VRAM={record.get('peak_cuda_memory_gib', 0.0):.2f}GiB"
        )

    ranked = sorted(
        results,
        key=lambda item: (
            item["val_f1"], item["val_precision"], item["val_recall"],
            -item.get("val_loss", float("inf")),
        ),
        reverse=True,
    )
    print("ranking:")
    for rank, item in enumerate(ranked, 1):
        print(
            f"  {rank:2d}. {item['checkpoint']}: "
            f"F1={item['val_f1']:.6f}"
        )
    print(f"best={ranked[0]['checkpoint']} | output={args.output}")


if __name__ == "__main__":
    main()
