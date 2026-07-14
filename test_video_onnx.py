"""Run an RCLane ONNX model on a video and render predicted lanes + timing.

The input video may already contain ground-truth annotations. Predictions are
drawn as solid cyan polylines with a black outline so they remain distinguishable
from colored GT points.
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dataset import normalize_image
from decode import decode


MODEL_HEIGHT = 320
MODEL_WIDTH = 800
OUTPUT_NAMES = (
    "seg_map",
    "up_arrow",
    "down_arrow",
    "up_bound",
    "down_bound",
)


def softmax_foreground(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits[:, 1] / np.sum(exp_logits, axis=1)


def create_session(model_path, provider, allow_tf32):
    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable; install the CUDA 13 "
                "onnxruntime-gpu build from requirements.txt"
            )
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": "0",
                    "use_tf32": "1" if allow_tf32 else "0",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "1",
                    "do_copy_in_default_stream": "1",
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    if provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            f"CUDA provider was requested but session uses {session.get_providers()}"
        )
    return session


def draw_predictions(frame, lanes):
    height, width = frame.shape[:2]
    sx = width / MODEL_WIDTH
    sy = height / MODEL_HEIGHT
    for index, lane in enumerate(lanes):
        points = lane.xy().copy()
        if len(points) < 2:
            continue
        points[:, 0] *= sx
        points[:, 1] *= sy
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        points = np.round(points).astype(np.int32)
        cv2.polylines(frame, [points], False, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.polylines(frame, [points], False, (255, 255, 0), 5, cv2.LINE_AA)

        near = points[int(np.argmax(points[:, 1]))]
        lane_id = lane.lane_id if lane.lane_id is not None else index
        label = f"P{lane_id} {lane.score:.2f}"
        position = (int(near[0]) + 8, max(32, int(near[1]) - 8))
        cv2.putText(
            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 5, cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 0), 2, cv2.LINE_AA,
        )


def draw_runtime(frame, provider, lane_count, infer_ms, decode_ms, pipeline_ms,
                 rolling_pipeline_ms, source_fps, input_has_gt):
    height, width = frame.shape[:2]
    box_width, box_height = min(690, width - 20), 160
    left, top = width - box_width - 10, 10
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (left, top), (width - 10, top + box_height), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    model_fps = 1000.0 / max(infer_ms, 1e-9)
    pipeline_fps = 1000.0 / max(rolling_pipeline_ms, 1e-9)
    realtime = "YES" if pipeline_fps >= source_fps else "NO"
    lines = (
        f"RCLane e19 ONNX {provider.upper()} | lanes={lane_count}",
        f"infer {infer_ms:6.1f} ms ({model_fps:5.1f} FPS)",
        f"decode {decode_ms:6.1f} ms | pipeline {pipeline_ms:6.1f} ms",
        f"rolling pipeline {pipeline_fps:5.1f} FPS | realtime@{source_fps:g}: {realtime}",
    )
    for row, text in enumerate(lines):
        cv2.putText(
            frame, text, (left + 14, top + 32 + row * 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
            2, cv2.LINE_AA,
        )

    legend = (
        "GT: colored dots | Prediction: cyan solid lines"
        if input_has_gt else "Prediction: cyan solid lines"
    )
    cv2.putText(
        frame, legend, (16, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        frame, legend, (16, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def timing_summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="RCLane ONNX model")
    parser.add_argument("--video", required=True, help="input video")
    parser.add_argument("--output", default="runs/video_test.mp4")
    parser.add_argument("--summary", default=None,
                        help="timing JSON; defaults next to output video")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--allow-tf32", action="store_true",
                        help="faster CUDA math with slightly larger numerical drift")
    parser.add_argument("--input-has-gt", action="store_true",
                        help="show a GT-vs-prediction legend for pre-labeled video")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="process only the first N frames")
    parser.add_argument("--decode-seg-threshold", type=float, default=0.5)
    parser.add_argument("--decode-seed-threshold", type=float, default=None)
    parser.add_argument("--decode-seed-min-dist", type=int, default=2)
    parser.add_argument("--decode-score-thresh", type=float, default=0.10)
    parser.add_argument("--decode-nms-iou", type=float, default=0.5)
    parser.add_argument("--decode-max-seeds", type=int, default=1024)
    parser.add_argument("--decode-nms-max-lanes", type=int, default=128)
    parser.add_argument("--decode-nms-scale", type=float, default=0.25)
    parser.add_argument(
        "--max-ego-lanes", type=int, default=4,
        help="post-process to at most N reliable lanes nearest the ego vehicle",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary else output_path.with_suffix(".json")
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.max_ego_lanes is not None and args.max_ego_lanes <= 0:
        raise ValueError("--max-ego-lanes must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )

    session = create_session(model_path, args.provider, args.allow_tf32)
    warmup = np.zeros((1, 3, MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)
    for _ in range(5):
        session.run(list(OUTPUT_NAMES), {"images": warmup})

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = min(source_frames, args.max_frames or source_frames)

    writer = cv2.VideoWriter(
        str(temporary_output), cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps, (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open video writer: {temporary_output}")

    timings = {"preprocess": [], "inference": [], "decode": [], "pipeline": []}
    lane_counts = []
    rolling = deque(maxlen=30)
    started = time.perf_counter()
    frame_index = 0
    try:
        while frame_index < target_frames:
            ok, frame = capture.read()
            if not ok:
                break
            pipeline_start = time.perf_counter()

            stage = time.perf_counter()
            images = normalize_image(frame, MODEL_WIDTH, MODEL_HEIGHT)
            images = images.unsqueeze(0).numpy()
            preprocess_ms = (time.perf_counter() - stage) * 1000

            stage = time.perf_counter()
            seg_logits, up_arrow, down_arrow, up_bound, down_bound = session.run(
                list(OUTPUT_NAMES), {"images": images}
            )
            inference_ms = (time.perf_counter() - stage) * 1000

            stage = time.perf_counter()
            seg_prob = softmax_foreground(seg_logits)[0]
            lanes = decode(
                seg_prob,
                up_arrow[0],
                down_arrow[0],
                up_bound[0],
                down_bound[0],
                seg_threshold=args.decode_seg_threshold,
                seed_threshold=args.decode_seed_threshold,
                seed_min_dist=args.decode_seed_min_dist,
                score_thresh=args.decode_score_thresh,
                iou_thresh=args.decode_nms_iou,
                max_seeds=args.decode_max_seeds,
                nms_max_lanes=args.decode_nms_max_lanes,
                nms_scale=args.decode_nms_scale,
                max_output_lanes=args.max_ego_lanes,
            )
            decode_ms = (time.perf_counter() - stage) * 1000
            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

            timings["preprocess"].append(preprocess_ms)
            timings["inference"].append(inference_ms)
            timings["decode"].append(decode_ms)
            timings["pipeline"].append(pipeline_ms)
            lane_counts.append(len(lanes))
            rolling.append(pipeline_ms)

            draw_predictions(frame, lanes)
            draw_runtime(
                frame, args.provider, len(lanes), inference_ms, decode_ms,
                pipeline_ms, float(np.median(rolling)), source_fps,
                args.input_has_gt,
            )
            writer.write(frame)
            frame_index += 1
            if frame_index % 25 == 0 or frame_index == target_frames:
                elapsed = time.perf_counter() - started
                print(
                    f"frame {frame_index}/{target_frames} | "
                    f"pipeline={pipeline_ms:.1f}ms lanes={len(lanes)} "
                    f"elapsed={elapsed:.1f}s"
                )
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        if temporary_output.exists():
            temporary_output.unlink()
        raise RuntimeError("input video produced no frames")
    os.replace(temporary_output, output_path)

    summary = {
        "model": str(model_path),
        "video": str(video_path),
        "output": str(output_path),
        "provider": args.provider,
        "session_providers": session.get_providers(),
        "allow_tf32": args.allow_tf32,
        "postprocessing": {
            "max_ego_lanes": args.max_ego_lanes,
            "min_score_ratio": 0.5,
            "balance_sides": True,
        },
        "resolution": [width, height],
        "source_fps": source_fps,
        "source_frames": source_frames,
        "processed_frames": frame_index,
        "wall_time_seconds": time.perf_counter() - started,
        "timings": {name: timing_summary(values) for name, values in timings.items()},
        "pipeline_fps_from_median": 1000.0 / np.median(timings["pipeline"]),
        "lane_count": {
            "mean": float(np.mean(lane_counts)),
            "min": int(min(lane_counts)),
            "max": int(max(lane_counts)),
        },
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_summary, summary_path)

    print(f"video OK: {output_path}")
    print(f"summary: {summary_path}")
    print(
        f"median pipeline={summary['timings']['pipeline']['median_ms']:.1f}ms "
        f"({summary['pipeline_fps_from_median']:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
