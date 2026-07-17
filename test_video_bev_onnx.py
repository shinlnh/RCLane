"""Run RCLane ONNX and export ego-centric cubic lane models in BEV.

Each JSONL frame stores one metric polynomial per visible marking:

    Y(X) = c0 + c1*X + c2*X^2 + c3*X^3

X points forward from ego and Y points left. Coefficients are valid only inside
the exported ``x_domain_m``; the implementation never extrapolates a detected
lane to the full 300 m visualization range.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from bev import (
    BevRange,
    CameraCalibration,
    clip_cubic_fit_to_funnel,
    evaluate_cubic,
    fit_cubic_lane,
    model_lane_to_ground,
)
from dataset import normalize_image_numpy
from decode import configure_decode_threads, decode, warmup_decode_backend
from test_video_onnx import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    OUTPUT_NAMES,
    create_session,
    draw_predictions,
    softmax_foreground,
    timing_summary,
)


LANE_COLORS = {
    0: (255, 255, 0),
    1: (0, 255, 0),
    2: (0, 165, 255),
    3: (255, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="runs/video_bev_e19.mp4")
    parser.add_argument("--polynomials", default=None,
                        help="per-frame JSONL; defaults next to output")
    parser.add_argument("--summary", default=None)
    parser.add_argument(
        "--provider", choices=("tensorrt", "cuda", "cpu"),
        default="tensorrt",
    )
    parser.add_argument("--trt-cache-dir", default="exports/trt_cache")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="zero-based source frame at which processing starts",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=300.0)
    parser.add_argument("--y-min", type=float, default=-85.0)
    parser.add_argument("--y-max", type=float, default=85.0)
    parser.add_argument(
        "--max-cubic-rmse", type=float, default=0.5,
        help="reject a cubic visualization above this metric RMSE",
    )
    parser.add_argument("--decode-seg-threshold", type=float, default=0.5)
    parser.add_argument("--decode-seed-threshold", type=float, default=None)
    parser.add_argument("--decode-seed-min-dist", type=int, default=2)
    parser.add_argument("--decode-score-thresh", type=float, default=0.10)
    parser.add_argument("--decode-nms-iou", type=float, default=0.5)
    parser.add_argument("--decode-max-seeds", type=int, default=1024)
    parser.add_argument(
        "--decode-crawl-backend", choices=("auto", "numba", "numpy"),
        default="auto",
    )
    parser.add_argument("--decode-cpu-threads", type=int, default=8)
    parser.add_argument("--decode-nms-max-lanes", type=int, default=128)
    parser.add_argument("--decode-nms-scale", type=float, default=0.25)
    parser.add_argument("--max-ego-lanes", type=int, default=4)
    parser.add_argument(
        "--funnel-margin", type=float, default=0.10,
        help="metric tolerance around the calibrated camera funnel",
    )
    return parser.parse_args()


def lane_to_record(lane, calibration, bev_range, max_cubic_rmse=0.5):
    points, scores = model_lane_to_ground(
        lane, MODEL_WIDTH, MODEL_HEIGHT, calibration, bev_range
    )
    fit = fit_cubic_lane(points, scores, iterations=3)
    lane_id = int(lane.lane_id) if lane.lane_id is not None else None
    valid_fit = fit is not None and fit["rmse"] <= max_cubic_rmse
    record = {
        "lane_id": f"P{lane_id}" if lane_id is not None else None,
        "lane_index": lane_id,
        "role": lane.lane_role,
        "score": float(lane.score),
        "projected_point_count": int(len(points)),
        "valid_fit": valid_fit,
        "fit_status": (
            "ok" if valid_fit else
            "rmse_above_limit" if fit is not None else
            "insufficient_geometry"
        ),
    }
    if fit is not None:
        record.update({
            "polynomial": "Y(X)=c0+c1*X+c2*X^2+c3*X^3",
            "coefficients_c0_to_c3": [
                float(value) for value in fit["coefficients"]
            ],
            "x_domain_m": [fit["x_min"], fit["x_max"]],
            "rmse_m": fit["rmse"],
            "fit_point_count": fit["point_count"],
            "inlier_count": fit["inlier_count"],
            "inlier_ratio": fit["inlier_ratio"],
        })
    return {
        "record": record,
        "points": points,
        "fit": fit if valid_fit else None,
    }


def clip_lane_results_to_funnel(lane_results, args):
    """Clip raw model-derived cubics without changing their geometry.

    Every output lane corresponds one-to-one with a decoded image-space lane.
    No lane is synthesized, shifted, reordered, or forced to be parallel. The
    cubic coefficients remain unchanged; only the declared X domain may shrink
    when the fitted curve leaves the calibrated camera footprint.
    """
    calibration = CameraCalibration()
    report = {
        "mode": "raw_model_projection",
        "parallel_assumption": False,
        "synthetic_lanes": False,
        "camera_x_m": float(calibration.camera_to_vehicle_matrix[0, 3]),
        "horizontal_fov_deg": float(calibration.horizontal_fov_deg),
        "margin_m": float(args.funnel_margin),
        "clipped_lanes": [],
        "rejected_lanes": [],
    }
    for result in lane_results:
        fit = result["fit"]
        record = result["record"]
        if fit is None:
            continue
        original_coefficients = np.asarray(
            fit["coefficients"], dtype=np.float64
        ).copy()
        clipped_fit, clip_report = clip_cubic_fit_to_funnel(
            fit,
            camera_x_m=report["camera_x_m"],
            horizontal_fov_deg=report["horizontal_fov_deg"],
            margin_m=args.funnel_margin,
        )
        lane_label = record["lane_id"]
        if clipped_fit is None:
            result["fit"] = None
            record.update({
                "valid_fit": False,
                "fit_status": "outside_camera_funnel",
                "funnel_guard": clip_report,
            })
            report["rejected_lanes"].append(lane_label)
            continue
        if not np.array_equal(
            original_coefficients, clipped_fit["coefficients"]
        ):
            raise AssertionError("funnel clipping changed cubic coefficients")
        result["fit"] = clipped_fit
        record["x_domain_m"] = [
            float(clipped_fit["x_min"]), float(clipped_fit["x_max"])
        ]
        record["funnel_guard"] = clip_report
        if clip_report["clipped"]:
            report["clipped_lanes"].append(lane_label)
    return report


def _bev_pixel(x_forward, y_left, bev_range, bounds):
    left, top, right, bottom = bounds
    px = left + (bev_range.y_max - y_left) / (
        bev_range.y_max - bev_range.y_min
    ) * (right - left)
    py = top + (bev_range.x_max - x_forward) / (
        bev_range.x_max - bev_range.x_min
    ) * (bottom - top)
    return np.column_stack((px, py))


def _draw_dashed_curve(canvas, points, color, thickness=4,
                       dash_points=7, gap_points=4):
    stride = dash_points + gap_points
    for start in range(0, len(points) - 1, stride):
        segment = points[start:min(start + dash_points + 1, len(points))]
        if len(segment) >= 2:
            cv2.polylines(
                canvas, [segment], False, (0, 0, 0), thickness + 4,
                cv2.LINE_AA,
            )
            cv2.polylines(
                canvas, [segment], False, color, thickness, cv2.LINE_AA,
            )


def draw_bev(lane_results, bev_range, calibration, funnel_report=None,
             width=640, height=1080):
    canvas = np.full((height, width, 3), (35, 19, 10), dtype=np.uint8)
    bounds = (72, 62, width - 24, height - 70)
    left, top, right, bottom = bounds

    # Requested sensor-style ROI funnel: ego at the apex and the configured
    # metric ROI at its far edge. It is a visualization of the output ROI, not
    # a replacement for the calibrated camera ray/ground intersection.
    camera_x = float(calibration.camera_to_vehicle_matrix[0, 3])
    half_fov_radians = np.deg2rad(
        calibration.horizontal_fov_deg * 0.5
    )
    fov_half_width = min(
        bev_range.y_max,
        max(0.0, bev_range.x_max - camera_x) * np.tan(half_fov_radians),
    )
    roi_metric = np.array(
        ((camera_x, 0.0),
         (bev_range.x_max, fov_half_width),
         (bev_range.x_max, -fov_half_width)),
        dtype=np.float64,
    )
    roi_pixels = np.round(_bev_pixel(
        roi_metric[:, 0], roi_metric[:, 1], bev_range, bounds
    )).astype(np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [roi_pixels], (105, 74, 105), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.polylines(canvas, [roi_pixels], True, (150, 120, 165), 2, cv2.LINE_AA)

    for x in np.arange(
        np.ceil(bev_range.x_min / 50.0) * 50.0,
        bev_range.x_max + 0.1,
        50.0,
    ):
        half_width = min(
            fov_half_width,
            max(0.0, x - camera_x) * np.tan(half_fov_radians),
        )
        y_at_left_edge = half_width
        y_at_right_edge = -half_width
        range_line = _bev_pixel(
            np.array([x, x]),
            np.array([y_at_left_edge, y_at_right_edge]),
            bev_range,
            bounds,
        )
        range_line = np.round(range_line).astype(np.int32)
        row = int(round(range_line[0, 1]))
        cv2.line(
            canvas, tuple(range_line[0]), tuple(range_line[1]),
            (118, 91, 120), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"{x:.0f}m", (8, min(bottom, max(top + 12, row + 5))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 190, 210), 1,
            cv2.LINE_AA,
        )

    zero = _bev_pixel(
        np.array([bev_range.x_min]), np.array([0.0]), bev_range, bounds
    )[0].astype(int)
    far_center = _bev_pixel(
        np.array([bev_range.x_max]), np.array([0.0]), bev_range, bounds
    )[0].astype(int)
    for y0 in range(zero[1] - 12, far_center[1], -28):
        y1 = max(far_center[1], y0 - 15)
        cv2.line(canvas, (zero[0], y0), (far_center[0], y1),
                 (225, 215, 230), 2, cv2.LINE_AA)
    ego = np.array(
        ((zero[0] - 10, bottom - 22), (zero[0] + 10, bottom - 22),
         (zero[0] + 14, bottom), (zero[0] - 14, bottom)),
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [ego], (238, 238, 238), cv2.LINE_AA)
    cv2.polylines(canvas, [ego], True, (20, 20, 20), 2, cv2.LINE_AA)

    formula_rows = []
    for result in lane_results:
        record = result["record"]
        lane_id = record["lane_index"]
        color = LANE_COLORS.get(lane_id, (220, 220, 220))
        points = result["points"]
        if len(points):
            pixels = _bev_pixel(
                points[:, 0], points[:, 1], bev_range, bounds
            )
            pixels = np.round(pixels).astype(np.int32)
            for pixel in pixels[::max(1, len(pixels) // 40)]:
                cv2.circle(canvas, tuple(pixel), 2, color, -1, cv2.LINE_AA)

        fit = result["fit"]
        if fit is None:
            rmse = record.get("rmse_m")
            reason = (
                f"rejected rmse={rmse:.2f}m" if rmse is not None
                else "insufficient geometry"
            )
            formula_rows.append((color, f"{record['lane_id']}: {reason}"))
            continue
        x = np.linspace(fit["x_min"], fit["x_max"], 160)
        y = evaluate_cubic(fit["coefficients"], x)
        valid = (
            np.isfinite(y)
            & (y >= bev_range.y_min)
            & (y <= bev_range.y_max)
        )
        curve = _bev_pixel(x[valid], y[valid], bev_range, bounds)
        if len(curve) >= 2:
            curve = np.round(curve).astype(np.int32)
            _draw_dashed_curve(canvas, curve, color)
            cv2.putText(
                canvas, record["lane_id"], tuple(curve[-1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA,
            )
        formula_rows.append((
            color,
            f"{record['lane_id']}: X={fit['x_min']:.1f}..{fit['x_max']:.1f}m  "
            f"rmse={fit['rmse']:.2f}m",
        ))

    cv2.putText(
        canvas,
        f"CAMERA FOV {calibration.horizontal_fov_deg:.0f}deg: "
        "X forward / Y left",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "RAW MODEL LANES | no synthetic/parallel prior",
        (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (80, 220, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Y left (+)                          Y right (-)",
        (left, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (210, 210, 210), 1, cv2.LINE_AA,
    )
    for row, (color, text) in enumerate(formula_rows[:4]):
        cv2.putText(
            canvas, text, (82, 82 + row * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA,
        )
    return canvas


def _preprocess_and_infer(session, frame):
    """Run the producer stage; safe to execute in a dedicated GPU thread."""
    stage = time.perf_counter()
    images = normalize_image_numpy(frame, MODEL_WIDTH, MODEL_HEIGHT)
    preprocess_ms = (time.perf_counter() - stage) * 1000.0

    stage = time.perf_counter()
    outputs = session.run(list(OUTPUT_NAMES), {"images": images})
    inference_ms = (time.perf_counter() - stage) * 1000.0
    return outputs, preprocess_ms, inference_ms


def _decode_and_generate_bev_result(outputs, args, calibration, bev_range):
    """Generate metric BEV data without drawing, compositing, or file I/O."""
    stage = time.perf_counter()
    lanes = decode(
        softmax_foreground(outputs[0])[0], outputs[1][0], outputs[2][0],
        outputs[3][0], outputs[4][0],
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
    decode_ms = (time.perf_counter() - stage) * 1000.0

    stage = time.perf_counter()
    lane_results = [
        lane_to_record(lane, calibration, bev_range, args.max_cubic_rmse)
        for lane in lanes
    ]
    funnel_report = clip_lane_results_to_funnel(lane_results, args)
    valid_records = [
        result["record"] for result in lane_results
        if result["record"]["valid_fit"]
    ]
    bev_result_ms = (time.perf_counter() - stage) * 1000.0
    return {
        "lanes": lanes,
        "lane_results": lane_results,
        "funnel_report": funnel_report,
        "valid_records": valid_records,
        "decode_ms": decode_ms,
        "bev_result_ms": bev_result_ms,
    }


def generate_bev_results(capture, target_frames, start_frame, session, args,
                         calibration, bev_range):
    """Generate ordered BEV results sequentially, one complete frame at a time.

    No BEV/camera drawing, compositing, video writing, or JSON serialization is
    performed inside this timed region.
    """
    timings = {
        "preprocess": [],
        "inference": [],
        "decode": [],
        "bev_result": [],
        "pipeline": [],
    }
    results = []
    generation_started = time.perf_counter()

    def finish(frame_index, inference_payload):
        outputs, preprocess_ms, inference_ms = inference_payload
        bundle = _decode_and_generate_bev_result(
            outputs, args, calibration, bev_range
        )
        compute_ms = (
            preprocess_ms + inference_ms + bundle["decode_ms"]
            + bundle["bev_result_ms"]
        )
        bundle.update({
            "frame_index": frame_index,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "compute_ms": compute_ms,
        })
        results.append(bundle)
        timings["preprocess"].append(preprocess_ms)
        timings["inference"].append(inference_ms)
        timings["decode"].append(bundle["decode_ms"])
        timings["bev_result"].append(bundle["bev_result_ms"])
        timings["pipeline"].append(compute_ms)
        processed = len(results)
        if processed % 25 == 0 or processed == target_frames:
            elapsed = time.perf_counter() - generation_started
            print(
                f"result {processed}/{target_frames} "
                f"(source={frame_index}) | "
                f"lanes={len(bundle['lanes'])} "
                f"cubic={len(bundle['valid_records'])} "
                f"decode={bundle['decode_ms']:.1f}ms "
                f"bev-result={bundle['bev_result_ms']:.1f}ms "
                f"throughput={processed / max(elapsed, 1e-9):.1f} FPS",
                flush=True,
            )

    for offset in range(target_frames):
        ok, frame = capture.read()
        if not ok:
            break
        finish(
            start_frame + offset,
            _preprocess_and_infer(session, frame),
        )

    generation_seconds = time.perf_counter() - generation_started
    if not results:
        raise RuntimeError("input video produced no BEV results")
    return results, timings, generation_seconds


def make_composite(frame, bev_canvas, frame_index, fit_count, pipeline_ms,
                   funnel_report=None, result_generation_fps=None):
    output = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bev_width = 640
    output[:, :bev_width] = cv2.resize(
        bev_canvas, (bev_width, 1080), interpolation=cv2.INTER_AREA
    )
    camera_width = 1920 - bev_width
    scaled_height = int(round(frame.shape[0] * camera_width / frame.shape[1]))
    camera_view = cv2.resize(frame, (camera_width, scaled_height),
                             interpolation=cv2.INTER_AREA)
    top = (1080 - scaled_height) // 2
    output[top:top + scaled_height, bev_width:] = camera_view
    cv2.line(output, (bev_width, 0), (bev_width, 1079), (255, 255, 255), 2)
    if result_generation_fps is None:
        result_generation_fps = 1000.0 / max(float(pipeline_ms), 1e-9)
    runtime_label = (
        f"frame={frame_index} | cubic lanes={fit_count} | "
        f"BEV-result={result_generation_fps:.1f} FPS "
        "(sequential, no render)"
    )
    cv2.putText(
        output, runtime_label,
        (bev_width + 24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
        (0, 0, 0), 5, cv2.LINE_AA,
    )
    cv2.putText(
        output, runtime_label,
        (bev_width + 24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        output, "CAMERA: raw decode (not back-projected)",
        (bev_width + 24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        output, "CAMERA: raw decode (not back-projected)",
        (bev_width + 24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return output


def main():
    args = parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")
    if args.max_cubic_rmse <= 0:
        raise ValueError("--max-cubic-rmse must be positive")
    if not args.x_min < args.x_max or not args.y_min < args.y_max:
        raise ValueError("invalid BEV range")
    if args.funnel_margin < 0:
        raise ValueError("--funnel-margin must be non-negative")

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    polynomial_path = (
        Path(args.polynomials).expanduser().resolve()
        if args.polynomials else output_path.with_suffix(".lanes.jsonl")
    )
    summary_path = (
        Path(args.summary).expanduser().resolve()
        if args.summary else output_path.with_suffix(".json")
    )
    for path in (model_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output_path, polynomial_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    calibration = CameraCalibration()
    bev_range = BevRange(args.x_min, args.x_max, args.y_min, args.y_max)
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
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (width, height) != (calibration.width, calibration.height):
        capture.release()
        raise ValueError(
            f"calibration is {calibration.width}x{calibration.height}, "
            f"video is {width}x{height}; crop/resize must be calibrated"
        )
    if args.start_frame >= source_frames:
        capture.release()
        raise ValueError(
            f"--start-frame {args.start_frame} is outside {source_frames} frames"
        )
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    available_frames = source_frames - args.start_frame
    target_frames = min(available_frames, args.max_frames or available_frames)

    overall_started = time.perf_counter()
    try:
        results, timings, generation_seconds = generate_bev_results(
            capture, target_frames, args.start_frame, session, args,
            calibration, bev_range,
        )
    finally:
        capture.release()
    processed_frames = len(results)
    result_generation_fps = processed_frames / generation_seconds

    fit_counts = Counter()
    funnel_clipped_lanes = Counter()
    funnel_rejected_lanes = Counter()
    lane_counts = []
    for bundle in results:
        valid_records = bundle["valid_records"]
        funnel_report = bundle["funnel_report"]
        for record in valid_records:
            fit_counts[record["lane_id"]] += 1
        funnel_clipped_lanes.update(
            funnel_report["clipped_lanes"]
        )
        funnel_rejected_lanes.update(
            funnel_report["rejected_lanes"]
        )
        lane_counts.append(len(bundle["lanes"]))

    temporary_polynomials = polynomial_path.with_suffix(
        polynomial_path.suffix + ".tmp"
    )
    with temporary_polynomials.open("w") as polynomial_file:
        for bundle in results:
            polynomial_file.write(json.dumps({
                "frame_index": bundle["frame_index"],
                "timestamp_seconds": bundle["frame_index"] / fps,
                "coordinate_system": {"X": "forward_m", "Y": "left_m"},
                "funnel_guard": bundle["funnel_report"],
                "lanes": [
                    result["record"] for result in bundle["lane_results"]
                ],
            }) + "\n")
    os.replace(temporary_polynomials, polynomial_path)

    # Rendering is a separate, untimed pass. Reopen the source video so none
    # of these operations can affect the reported BEV-result throughput.
    temporary_output = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    render_capture = cv2.VideoCapture(str(video_path))
    if not render_capture.isOpened():
        raise RuntimeError(f"cannot reopen video for rendering: {video_path}")
    if args.start_frame:
        render_capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    writer = cv2.VideoWriter(
        str(temporary_output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (1920, 1080),
    )
    if not writer.isOpened():
        render_capture.release()
        raise RuntimeError(f"cannot open writer: {temporary_output}")
    render_started = time.perf_counter()
    try:
        for position, bundle in enumerate(results, 1):
            ok, frame = render_capture.read()
            if not ok:
                raise RuntimeError(
                    f"render pass stopped before frame {bundle['frame_index']}"
                )
            bev_canvas = draw_bev(
                bundle["lane_results"], bev_range, calibration,
                bundle["funnel_report"],
            )
            draw_predictions(frame, bundle["lanes"])
            composite = make_composite(
                frame, bev_canvas, bundle["frame_index"],
                len(bundle["valid_records"]), bundle["compute_ms"],
                bundle["funnel_report"],
                result_generation_fps=result_generation_fps,
            )
            writer.write(composite)
            if position % 100 == 0 or position == processed_frames:
                print(
                    f"render {position}/{processed_frames} "
                    "(excluded from BEV-result FPS)",
                    flush=True,
                )
    finally:
        render_capture.release()
        writer.release()
    render_seconds = time.perf_counter() - render_started
    os.replace(temporary_output, output_path)

    summary = {
        "model": str(model_path),
        "video": str(video_path),
        "output_video": str(output_path),
        "output_polynomials": str(polynomial_path),
        "processed_frames": processed_frames,
        "start_frame": args.start_frame,
        "source_frames": source_frames,
        "fps": fps,
        "provider": args.provider,
        "execution": {
            "mode": "sequential_per_frame",
            "result_generation_scope": (
                "video read + normalize + inference + decode + metric "
                "projection + cubic fit + camera-funnel clipping"
            ),
            "excluded_from_result_fps": [
                "draw_bev",
                "draw_predictions",
                "make_composite",
                "video_encode_write",
                "json_serialize_write",
            ],
            "result_generation_wall_seconds": generation_seconds,
            "result_generation_fps": result_generation_fps,
            "render_wall_seconds": render_seconds,
        },
        "coordinate_system": {
            "X": "forward from ego, metres",
            "Y": "left of ego, metres",
            "Z": "not exported; local road plane is Z=0",
        },
        "camera": {
            "resolution": [calibration.width, calibration.height],
            "horizontal_fov_deg": calibration.horizontal_fov_deg,
            "intrinsic": calibration.intrinsic.tolist(),
            "camera_to_vehicle": calibration.camera_to_vehicle_matrix.tolist(),
            "distortion": None,
        },
        "bev_range_m": {
            "X": [bev_range.x_min, bev_range.x_max],
            "Y": [bev_range.y_min, bev_range.y_max],
        },
        "polynomial": {
            "formula": "Y(X)=c0+c1*X+c2*X^2+c3*X^3",
            "coefficient_order": ["c0", "c1", "c2", "c3"],
            "max_accepted_rmse_m": args.max_cubic_rmse,
            "fit_counts_by_lane": dict(sorted(fit_counts.items())),
        },
        "bev_projection": {
            "mode": "raw_model_projection",
            "parallel_assumption": False,
            "synthetic_lanes": False,
            "funnel_margin_m": args.funnel_margin,
            "funnel_clipped_lane_fits": int(
                sum(funnel_clipped_lanes.values())
            ),
            "funnel_clipped_by_lane": dict(
                sorted(funnel_clipped_lanes.items())
            ),
            "funnel_rejected_lane_fits": int(
                sum(funnel_rejected_lanes.values())
            ),
            "funnel_rejected_by_lane": dict(
                sorted(funnel_rejected_lanes.items())
            ),
        },
        "lane_count": {
            "mean": float(np.mean(lane_counts)),
            "min": int(min(lane_counts)),
            "max": int(max(lane_counts)),
        },
        "timings": {
            name: timing_summary(values) for name, values in timings.items()
        },
        "wall_time_seconds": time.perf_counter() - overall_started,
    }
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_summary, summary_path)
    print(f"video OK: {output_path}")
    print(f"polynomials OK: {polynomial_path}")
    print(f"summary: {summary_path}")
    print(
        f"BEV-result throughput: {result_generation_fps:.2f} FPS "
        f"({generation_seconds:.3f}s, render excluded)"
    )


if __name__ == "__main__":
    main()
