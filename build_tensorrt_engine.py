"""Build/cache the RCLane TensorRT FP16 engine and verify its outputs.

TensorRT is used through ONNX Runtime's TensorRT Execution Provider so the
cached ``.engine`` remains consumable by the same runtime pipeline. The script
refuses silent provider fallback and records numerical/performance comparisons
against CUDA FP32.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from test_video_onnx import OUTPUT_NAMES, create_session, timing_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", default="exports/trt_cache")
    parser.add_argument("--report", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def benchmark(session, images, warmup, iterations):
    for _ in range(warmup):
        session.run(list(OUTPUT_NAMES), {"images": images})
    timings = []
    outputs = None
    for _ in range(iterations):
        started = time.perf_counter()
        outputs = session.run(list(OUTPUT_NAMES), {"images": images})
        timings.append((time.perf_counter() - started) * 1000.0)
    return outputs, timing_summary(timings)


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    model_path = Path(args.model).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report else cache_dir / "build_report.json"
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    images = rng.normal(size=(1, 3, 320, 800)).astype(np.float32)

    build_started = time.perf_counter()
    trt_session = create_session(
        model_path, "tensorrt", allow_tf32=False, trt_cache_dir=cache_dir
    )
    build_seconds = time.perf_counter() - build_started
    if trt_session.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError("TensorRT provider silently fell back")
    cuda_session = create_session(model_path, "cuda", allow_tf32=False)

    trt_outputs, trt_timing = benchmark(
        trt_session, images, args.warmup, args.iterations
    )
    cuda_outputs, cuda_timing = benchmark(
        cuda_session, images, args.warmup, args.iterations
    )
    comparisons = {}
    for name, trt_output, cuda_output in zip(
        OUTPUT_NAMES, trt_outputs, cuda_outputs
    ):
        difference = trt_output.astype(np.float64) - cuda_output.astype(
            np.float64
        )
        comparisons[name] = {
            "shape": list(trt_output.shape),
            "max_abs": float(np.max(np.abs(difference))),
            "mean_abs": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
        }

    cache_files = []
    for path in sorted(cache_dir.iterdir()):
        if path.is_file():
            cache_files.append({
                "path": str(path),
                "size_bytes": path.stat().st_size,
            })
    engines = [
        item for item in cache_files if item["path"].endswith(".engine")
    ]
    if not engines:
        raise RuntimeError(f"TensorRT did not create an engine in {cache_dir}")

    report = {
        "model": str(model_path),
        "provider": trt_session.get_providers()[0],
        "precision": "fp16",
        "input_shape": list(images.shape),
        "build_or_cache_load_seconds": build_seconds,
        "tensorrt_timing": trt_timing,
        "cuda_fp32_timing": cuda_timing,
        "speedup_from_median": (
            cuda_timing["median_ms"] / trt_timing["median_ms"]
        ),
        "output_comparison_to_cuda_fp32": comparisons,
        "cache_files": cache_files,
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, report_path)
    print(f"TensorRT engine OK: {engines[0]['path']}")
    print(f"report: {report_path}")
    print(
        "median inference: TensorRT={:.3f}ms CUDA={:.3f}ms speedup={:.2f}x".format(
            trt_timing["median_ms"], cuda_timing["median_ms"],
            report["speedup_from_median"],
        )
    )


if __name__ == "__main__":
    main()
