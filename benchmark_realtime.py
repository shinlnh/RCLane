"""Benchmark the optimized RCLane -> decode -> BEV core pipeline.

Rendering, video writing and source-frame acquisition are reported separately
from the sequential per-frame ADAS latency. Each frame completes inference,
decode and raw-model BEV projection before the next frame is processed.
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numba
import numpy as np

from bev import BevRange, CameraCalibration
from dataset import normalize_image_numpy
from decode import decode, warmup_decode_backend
from test_video_bev_onnx import clip_lane_results_to_funnel, lane_to_record
from test_video_onnx import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    OUTPUT_NAMES,
    create_session,
    softmax_foreground,
    timing_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--provider", choices=("tensorrt", "cuda"),
                        default="tensorrt")
    parser.add_argument("--trt-cache-dir", default="exports/trt_cache")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-seeds", type=int, default=1024)
    parser.add_argument("--report", default="runs/realtime_benchmark.json")
    return parser.parse_args()


def projection_args():
    return SimpleNamespace(funnel_margin=0.10)


def postprocess(outputs, calibration, bev_range, config, max_seeds):
    started = time.perf_counter()
    stage = time.perf_counter()
    seg_prob = softmax_foreground(outputs[0])[0]
    softmax_ms = (time.perf_counter() - stage) * 1000.0
    stage = time.perf_counter()
    lanes = decode(
        seg_prob, outputs[1][0], outputs[2][0], outputs[3][0], outputs[4][0],
        max_seeds=max_seeds,
        nms_max_lanes=128,
        max_output_lanes=4,
        crawl_backend="numba",
        point_nms_backend="numba",
    )
    decode_ms = (time.perf_counter() - stage) * 1000.0
    stage = time.perf_counter()
    lane_results = [
        lane_to_record(lane, calibration, bev_range, 0.5) for lane in lanes
    ]
    funnel = clip_lane_results_to_funnel(lane_results, config)
    bev_ms = (time.perf_counter() - stage) * 1000.0
    return {
        "softmax_ms": softmax_ms,
        "decode_ms": decode_ms,
        "bev_ms": bev_ms,
        "postprocess_ms": (time.perf_counter() - started) * 1000.0,
        "lane_count": len(lanes),
        "funnel_clipped_lanes": len(funnel["clipped_lanes"]),
        "funnel_rejected_lanes": len(funnel["rejected_lanes"]),
    }


def load_frames(path, start_frame, max_frames):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    read_timings = []
    for _ in range(max_frames):
        started = time.perf_counter()
        ok, frame = capture.read()
        read_timings.append((time.perf_counter() - started) * 1000.0)
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("video produced no benchmark frames")
    return frames, read_timings[:len(frames)]


def summarize_results(results):
    return {
        name: timing_summary([result[name] for result in results])
        for name in ("softmax_ms", "decode_ms", "bev_ms", "postprocess_ms")
    }


def main():
    args = parse_args()
    if args.start_frame < 0 or args.max_frames <= 0:
        raise ValueError("start-frame must be >=0 and max-frames positive")
    if args.cpu_threads <= 0 or args.cpu_threads > numba.config.NUMBA_NUM_THREADS:
        raise ValueError(
            f"cpu-threads must be in [1, {numba.config.NUMBA_NUM_THREADS}]"
        )
    if args.max_seeds <= 0:
        raise ValueError("max-seeds must be positive")
    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)
    numba.set_num_threads(args.cpu_threads)
    warmup_decode_backend("numba")
    session = create_session(
        model_path, args.provider, False, args.trt_cache_dir
    )
    warmup = np.zeros((1, 3, MODEL_HEIGHT, MODEL_WIDTH), np.float32)
    for _ in range(10):
        session.run(list(OUTPUT_NAMES), {"images": warmup})
    frames, read_timings = load_frames(
        video_path, args.start_frame, args.max_frames
    )
    calibration = CameraCalibration()
    bev_range = BevRange()
    config = projection_args()

    sequential_results = []
    preprocess_timings = []
    inference_timings = []
    total_timings = []
    for frame in frames:
        total_started = time.perf_counter()
        stage = time.perf_counter()
        images = normalize_image_numpy(frame, MODEL_WIDTH, MODEL_HEIGHT)
        preprocess_timings.append((time.perf_counter() - stage) * 1000.0)
        stage = time.perf_counter()
        outputs = session.run(list(OUTPUT_NAMES), {"images": images})
        inference_timings.append((time.perf_counter() - stage) * 1000.0)
        sequential_results.append(postprocess(
            outputs, calibration, bev_range, config, args.max_seeds
        ))
        total_timings.append((time.perf_counter() - total_started) * 1000.0)

    report = {
        "model": str(model_path),
        "video": str(video_path),
        "provider": session.get_providers()[0],
        "frames": len(frames),
        "start_frame": args.start_frame,
        "cpu_threads": args.cpu_threads,
        "max_seeds": args.max_seeds,
        "rendering_included": False,
        "source_read_ms": timing_summary(read_timings),
        "sequential": {
            "preprocess_ms": timing_summary(preprocess_timings),
            "inference_ms": timing_summary(inference_timings),
            **summarize_results(sequential_results),
            "core_pipeline_ms": timing_summary(total_timings),
            "fps_from_median_latency": 1000.0 / np.median(total_timings),
        },
        "lane_count": {
            "mean": float(np.mean([
                result["lane_count"] for result in sequential_results
            ])),
            "min": int(min(
                result["lane_count"] for result in sequential_results
            )),
            "max": int(max(
                result["lane_count"] for result in sequential_results
            )),
        },
        "bev_projection": {
            "mode": "raw_model_projection",
            "parallel_assumption": False,
            "synthetic_lanes": False,
            "funnel_clipped_lane_fits": int(sum(
                result["funnel_clipped_lanes"]
                for result in sequential_results
            )),
            "funnel_rejected_lane_fits": int(sum(
                result["funnel_rejected_lanes"]
                for result in sequential_results
            )),
        },
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, report_path)
    print(f"report: {report_path}")
    print(
        "sequential median={:.3f}ms ({:.2f} FPS)".format(
            report["sequential"]["core_pipeline_ms"]["median_ms"],
            report["sequential"]["fps_from_median_latency"],
        )
    )


if __name__ == "__main__":
    main()
