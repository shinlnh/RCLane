"""Render native C++ decoded lanes in a separate, untimed video pass."""

import argparse
from collections import deque
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decode import Lane
from bev import BevRange, CameraCalibration
from test_video_bev_onnx import draw_bev, make_composite
from test_video_onnx import draw_predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark-report", default=None)
    return parser.parse_args()


def load_lane(payload):
    lane = Lane(800, 320)
    lane.points = np.asarray(payload["points"], dtype=np.float32)
    lane._score_sum = float(payload["score"]) * len(lane.points)
    lane.lane_id = int(payload["lane_id"])
    lane.lane_role = payload["role"]
    lane.is_ego_boundary = lane.lane_role in ("ego_left", "ego_right")
    lane.lateral_rank = None
    return lane


def load_bev_result(payload):
    fit_payload = payload["fit"]
    fit = None
    if payload["fit_accepted"] and fit_payload is not None:
        fit = {
            "coefficients": np.asarray(
                fit_payload["coefficients"], dtype=np.float64
            ),
            "x_min": float(fit_payload["x_min"]),
            "x_max": float(fit_payload["x_max"]),
            "rmse": float(fit_payload["rmse"]),
            "point_count": int(fit_payload["point_count"]),
            "inlier_count": int(fit_payload["inlier_count"]),
        }
    lane_id = int(payload["lane_id"])
    record = {
        "lane_id": f"P{lane_id}",
        "lane_index": lane_id,
        "role": payload["role"],
        "score": float(payload["score"]),
        "valid_fit": fit is not None,
    }
    if fit_payload is not None:
        record["rmse_m"] = float(fit_payload["rmse"])
    points = np.asarray(payload["points"], dtype=np.float64)
    return {
        "record": record,
        "points": points[:, :2] if points.size else np.empty((0, 2)),
        "fit": fit,
    }


def funnel_report(bev_payloads):
    return {
        "mode": "raw_model_projection",
        "parallel_assumption": False,
        "synthetic_lanes": False,
        "clipped_lanes": [
            f"P{item['lane_id']}" for item in bev_payloads
            if item["funnel_clipped"]
        ],
        "rejected_lanes": [
            f"P{item['lane_id']}" for item in bev_payloads
            if item["fit"] is not None and not item["fit_accepted"]
        ],
    }


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration = CameraCalibration()
    bev_range = BevRange()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temporary = output_path.with_name(output_path.stem + ".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create temporary video: {temporary}")

    rendered = 0
    rolling_core_ms = deque(maxlen=20)
    try:
        with results_path.open() as results:
            for line in results:
                payload = json.loads(line)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"video ended before result frame {payload['frame_index']}"
                    )
                lanes = [load_lane(item) for item in payload["lanes"]]
                bev_payloads = payload["bev_lanes"]
                lane_results = [
                    load_bev_result(item) for item in bev_payloads
                ]
                guard = funnel_report(bev_payloads)
                timing = payload["timing"]
                rolling_core_ms.append(float(timing["core_ms"]))
                rolling_fps = 1000.0 / max(
                    float(np.mean(rolling_core_ms)), 1e-9
                )
                draw_predictions(frame, lanes)
                bev_canvas = draw_bev(
                    lane_results, bev_range, calibration, guard
                )
                composite = make_composite(
                    frame, bev_canvas, int(payload["frame_index"]),
                    sum(item["fit"] is not None for item in lane_results),
                    float(timing["core_ms"]), guard,
                    result_generation_fps=rolling_fps,
                )
                detail = (
                    f"C++ current={timing['core_ms']:.1f}ms | "
                    f"infer={timing['inference_ms']:.1f} "
                    f"decode={timing['decode_ms']:.1f} "
                    f"BEV-result={timing['bev_result_ms']:.2f}ms | "
                    "draw/encode excluded"
                )
                cv2.putText(
                    composite, detail, (664, 101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    composite, detail, (664, 101),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
                writer.write(composite)
                rendered += 1
                if rendered % 200 == 0:
                    print(f"rendered {rendered} frames", flush=True)
    finally:
        capture.release()
        writer.release()

    encoded = output_path.with_name(output_path.stem + ".h264.tmp.mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-i", str(temporary), "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(encoded),
        ],
        check=True,
    )
    os.replace(encoded, output_path)
    temporary.unlink(missing_ok=True)
    print(f"rendered video: {output_path} ({rendered} frames)")


if __name__ == "__main__":
    main()
