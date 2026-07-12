"""
Run single-image RCLane inference from a saved checkpoint.

Default behavior uses the best CULane F1 checkpoint from checkpoints/f1_resume
and the first image in data/CULane/list/val_gt.txt, then writes an overlay image.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import normalize_image
from decode import decode_predictions
from rclane import RCLane


DEFAULT_IMG_SIZE = (320, 800)  # H, W used by train.py
COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 120, 0),
    (255, 0, 180),
    (80, 120, 255),
    (180, 255, 0),
]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_image_path(image, data_root, list_file):
    if image:
        p = Path(image)
        if p.exists():
            return p
        p = Path(data_root) / image.lstrip("/\\")
        if p.exists():
            return p
        raise FileNotFoundError(f"image not found: {image}")

    list_path = Path(data_root) / list_file
    if not list_path.exists():
        raise FileNotFoundError(
            f"no --image provided and default list is missing: {list_path}"
        )
    with list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0].lstrip("/\\")
            p = Path(data_root) / rel
            if p.exists():
                return p
    raise FileNotFoundError(f"no readable image found in {list_path}")


def scale_lanes_to_original(lanes, original_w, original_h, model_w, model_h):
    sx = original_w / float(model_w)
    sy = original_h / float(model_h)
    out = []
    for lane in lanes:
        xy = lane.xy()
        if len(xy) < 2:
            continue
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        out.append((xy, lane.score))
    return out


def draw_lanes(img_bgr, lanes):
    out = img_bgr.copy()
    for idx, (xy, score) in enumerate(lanes):
        pts = np.round(xy).astype(np.int32)
        color = COLORS[idx % len(COLORS)]
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=8)
        for x, y in pts[:: max(1, len(pts) // 20)]:
            cv2.circle(out, (int(x), int(y)), 3, color, -1)
        x0, y0 = pts[min(len(pts) - 1, len(pts) // 2)]
        cv2.putText(
            out,
            f"{idx}:{score:.2f}",
            (int(x0), int(y0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def save_seg_heatmap(seg_prob, original_w, original_h, out_path):
    heat = cv2.resize(seg_prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    heat = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), heat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="./checkpoints/f1_resume/best.pth")
    ap.add_argument("--image", default=None,
                    help="image path. If omitted, the first image in --list is used")
    ap.add_argument("--data-root", default="./data/CULane")
    ap.add_argument("--list", default="list/val_gt.txt",
                    help="CULane list used only when --image is omitted")
    ap.add_argument("--out", default="./runs/rclane_test.jpg")
    ap.add_argument("--seg-out", default="./runs/rclane_test_seg.jpg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vision", default=None,
                    help="model variant; defaults to checkpoint args, then b0")
    ap.add_argument("--decode-seg-threshold", type=float, default=None)
    ap.add_argument("--decode-seed-threshold", type=float, default=None)
    ap.add_argument("--decode-seed-min-dist", type=int, default=None)
    ap.add_argument("--decode-score-thresh", type=float, default=None)
    ap.add_argument("--decode-nms-iou", type=float, default=None)
    args = ap.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    ckpt_args = ckpt.get("args", {})
    vision = args.vision or ckpt_args.get("vision", "b0")
    device = torch.device(args.device)

    model = RCLane(vision=vision, img_size=DEFAULT_IMG_SIZE).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    img_path = resolve_image_path(args.image, args.data_root, args.list)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(str(img_path))
    original_h, original_w = img_bgr.shape[:2]
    model_h, model_w = DEFAULT_IMG_SIZE

    x = normalize_image(img_bgr, model_w, model_h).unsqueeze(0).to(device)
    use_amp = bool(ckpt_args.get("amp", False)) and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else torch.no_grad()
    with torch.no_grad(), amp_ctx:
        preds = model(x)

    decode_kwargs = {
        "seg_threshold": args.decode_seg_threshold
        if args.decode_seg_threshold is not None
        else ckpt_args.get("decode_seg_threshold", 0.5),
        "seed_threshold": args.decode_seed_threshold
        if args.decode_seed_threshold is not None
        else ckpt_args.get("decode_seed_threshold", None),
        "seed_min_dist": args.decode_seed_min_dist
        if args.decode_seed_min_dist is not None
        else ckpt_args.get("decode_seed_min_dist", 2),
        "score_thresh": args.decode_score_thresh
        if args.decode_score_thresh is not None
        else ckpt_args.get("decode_score_thresh", 0.10),
        "iou_thresh": args.decode_nms_iou
        if args.decode_nms_iou is not None
        else ckpt_args.get("decode_nms_iou", 0.5),
    }
    decoded = decode_predictions(preds, **decode_kwargs)[0]
    lanes = scale_lanes_to_original(decoded, original_w, original_h, model_w, model_h)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_lanes(img_bgr, lanes)
    cv2.imwrite(str(out_path), overlay)

    if args.seg_out:
        seg_prob = torch.softmax(preds["seg_map"], dim=1)[0, 1].detach().cpu().numpy()
        seg_path = Path(args.seg_out)
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        save_seg_heatmap(seg_prob, original_w, original_h, seg_path)
    else:
        seg_path = None

    metrics = ckpt.get("metrics", {})
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint epoch: {ckpt.get('epoch')} | val_f1: {metrics.get('val_f1')}")
    print(f"image: {img_path}")
    print(f"decoded lanes: {len(lanes)}")
    print(f"overlay: {out_path}")
    if seg_path:
        print(f"seg heatmap: {seg_path}")


if __name__ == "__main__":
    main()
