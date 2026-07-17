"""Run an RCLane ONNX model on a video and render predicted lanes + timing.

The input video may already contain ground-truth annotations. Ego-lane boundaries
are highlighted in green/orange, other predictions in cyan, all with a black
outline so they remain distinguishable from colored GT points.
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

from dataset import normalize_image_numpy
from decode import (
    configure_decode_threads,
    decode,
    ego_lane_boundaries,
    warmup_decode_backend,
)


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
    # Binary softmax: exp(l1)/(exp(l0)+exp(l1)) = sigmoid(l1-l0).
    # This halves exponent work and avoids allocating a two-channel temporary.
    difference = logits[:, 0] - logits[:, 1]
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(difference))


def create_session(model_path, provider, allow_tf32, trt_cache_dir=None):
    available = ort.get_available_providers()
    cuda_options = {
        "device_id": "0",
        "use_tf32": "1" if allow_tf32 else "0",
        "cudnn_conv_algo_search": "HEURISTIC",
        "cudnn_conv_use_max_workspace": "1",
        "do_copy_in_default_stream": "1",
    }
    if provider == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "TensorrtExecutionProvider is unavailable in ONNX Runtime"
            )
        try:
            import tensorrt  # noqa: F401 - preloads libnvinfer for ORT
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT provider requested but tensorrt-cu13 is not installed"
            ) from exc
        cache_dir = Path(
            trt_cache_dir or Path(model_path).resolve().parent / "trt_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        shape = f"images:1x3x{MODEL_HEIGHT}x{MODEL_WIDTH}"
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": "0",
                    "trt_fp16_enable": "True",
                    "trt_engine_cache_enable": "True",
                    "trt_engine_cache_path": str(cache_dir),
                    "trt_timing_cache_enable": "True",
                    "trt_timing_cache_path": str(cache_dir),
                    "trt_force_timing_cache": "True",
                    "trt_builder_optimization_level": "3",
                    "trt_max_workspace_size": str(2 * 1024 ** 3),
                    "trt_min_subgraph_size": "1",
                    "trt_profile_min_shapes": shape,
                    "trt_profile_opt_shapes": shape,
                    "trt_profile_max_shapes": shape,
                },
            ),
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]
    elif provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable; install the CUDA 13 "
                "onnxruntime-gpu build from requirements.txt"
            )
        providers = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    expected_provider = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }.get(provider)
    if expected_provider and session.get_providers()[0] != expected_provider:
        raise RuntimeError(
            f"{expected_provider} was requested but session uses "
            f"{session.get_providers()}"
        )
    return session


def _lane_points_in_frame(lane, width, height):
    points = lane.xy().copy()
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.float32)
    points[:, 0] *= width / MODEL_WIDTH
    points[:, 1] *= height / MODEL_HEIGHT
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def _x_at_rows(points, rows):
    order = np.argsort(points[:, 1])
    ys = points[order, 1]
    xs = points[order, 0]
    unique_y, inverse = np.unique(ys, return_inverse=True)
    x_sum = np.zeros(len(unique_y), dtype=np.float64)
    count = np.zeros(len(unique_y), dtype=np.float64)
    np.add.at(x_sum, inverse, xs)
    np.add.at(count, inverse, 1)
    return np.interp(rows, unique_y, x_sum / count)


def draw_ego_corridor(frame, ego_left, ego_right):
    """Shade the visible region bounded by the two current-lane boundaries."""
    if ego_left is None or ego_right is None:
        return
    height, width = frame.shape[:2]
    left = _lane_points_in_frame(ego_left, width, height)
    right = _lane_points_in_frame(ego_right, width, height)
    if len(left) < 2 or len(right) < 2:
        return
    y_start = max(float(left[:, 1].min()), float(right[:, 1].min()))
    y_stop = min(float(left[:, 1].max()), float(right[:, 1].max()))
    if y_stop - y_start < 8:
        return
    rows = np.linspace(y_start, y_stop, 64)
    left_x = _x_at_rows(left, rows)
    right_x = _x_at_rows(right, rows)
    valid = right_x > left_x
    if np.count_nonzero(valid) < 2:
        return
    rows = rows[valid]
    left_edge = np.column_stack((left_x[valid], rows))
    right_edge = np.column_stack((right_x[valid], rows))[::-1]
    polygon = np.round(np.vstack((left_edge, right_edge))).astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (40, 140, 40), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def draw_predictions(frame, lanes):
    height, width = frame.shape[:2]
    ego_left, ego_right = ego_lane_boundaries(lanes)
    draw_ego_corridor(frame, ego_left, ego_right)

    for index, lane in enumerate(lanes):
        points = _lane_points_in_frame(lane, width, height)
        if len(points) < 2:
            continue
        points = np.round(points).astype(np.int32)
        role = getattr(lane, "lane_role", None)
        if role == "ego_left":
            color, role_text, thickness = (0, 255, 0), "EGO-L", 7
        elif role == "ego_right":
            color, role_text, thickness = (0, 165, 255), "EGO-R", 7
        else:
            color, role_text, thickness = (255, 255, 0), "", 5
        cv2.polylines(
            frame, [points], False, (0, 0, 0), thickness + 5, cv2.LINE_AA
        )
        cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)

        near = points[int(np.argmax(points[:, 1]))]
        lane_id = lane.lane_id if lane.lane_id is not None else index
        role_label = f" {role_text}" if role_text else ""
        label = f"P{lane_id}{role_label} {lane.score:.2f}"
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )[0]
        label_x = min(max(8, int(near[0]) + 8), width - text_size[0] - 8)
        position = (label_x, max(32, int(near[1]) - 8))
        cv2.putText(
            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 5, cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, 2, cv2.LINE_AA,
        )


def draw_runtime(frame, provider, lane_count, infer_ms, decode_ms, pipeline_ms,
                 rolling_pipeline_ms, source_fps, input_has_gt, ego_left,
                 ego_right):
    height, width = frame.shape[:2]
    box_width, box_height = min(690, width - 20), 196
    left, top = width - box_width - 10, 10
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (left, top), (width - 10, top + box_height), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    model_fps = 1000.0 / max(infer_ms, 1e-9)
    pipeline_fps = 1000.0 / max(rolling_pipeline_ms, 1e-9)
    realtime = "YES" if pipeline_fps >= source_fps else "NO"
    left_id = f"P{ego_left.lane_id}" if ego_left is not None else "missing"
    right_id = f"P{ego_right.lane_id}" if ego_right is not None else "missing"
    lines = (
        f"RCLane e19 ONNX {provider.upper()} | lanes={lane_count}",
        f"ego lane boundaries: {left_id} (L) | {right_id} (R)",
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

    prediction_legend = "EGO-L: green | EGO-R: orange | other: cyan"
    legend = (
        f"GT: colored dots | {prediction_legend}"
        if input_has_gt else prediction_legend
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
    parser.add_argument(
        "--provider", choices=["tensorrt", "cuda", "cpu"],
        default="tensorrt",
    )
    parser.add_argument("--trt-cache-dir", default="exports/trt_cache")
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
    parser.add_argument(
        "--decode-crawl-backend", choices=("auto", "numba", "numpy"),
        default="auto",
    )
    parser.add_argument("--decode-cpu-threads", type=int, default=8)
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

    cv2.setNumThreads(1)
    configure_decode_threads(args.decode_cpu_threads)
    session = create_session(
        model_path, args.provider, args.allow_tf32, args.trt_cache_dir
    )
    warmup_decode_backend(args.decode_crawl_backend)
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
    ego_left_frames = 0
    ego_right_frames = 0
    ego_pair_frames = 0
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
            images = normalize_image_numpy(frame, MODEL_WIDTH, MODEL_HEIGHT)
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
                crawl_backend=args.decode_crawl_backend,
            )
            decode_ms = (time.perf_counter() - stage) * 1000
            pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

            timings["preprocess"].append(preprocess_ms)
            timings["inference"].append(inference_ms)
            timings["decode"].append(decode_ms)
            timings["pipeline"].append(pipeline_ms)
            lane_counts.append(len(lanes))
            rolling.append(pipeline_ms)
            ego_left, ego_right = ego_lane_boundaries(lanes)
            ego_left_frames += int(ego_left is not None)
            ego_right_frames += int(ego_right is not None)
            ego_pair_frames += int(ego_left is not None and ego_right is not None)

            draw_predictions(frame, lanes)
            draw_runtime(
                frame, args.provider, len(lanes), inference_ms, decode_ms,
                pipeline_ms, float(np.median(rolling)), source_fps,
                args.input_has_gt, ego_left, ego_right,
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
            "lane_id_semantics": {
                "P0": "next boundary left of ego lane",
                "P1": "ego lane left boundary",
                "P2": "ego lane right boundary",
                "P3": "next boundary right of ego lane",
            },
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
        "ego_lane_boundary_coverage": {
            "left_frames": ego_left_frames,
            "right_frames": ego_right_frames,
            "pair_frames": ego_pair_frames,
            "pair_rate": ego_pair_frames / frame_index,
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
