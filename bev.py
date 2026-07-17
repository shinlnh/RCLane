"""Ego-centric inverse-perspective mapping and cubic lane fitting.

The public coordinate convention is ADAS-style:

    X: forward from the ego origin (metres)
    Y: left of ego (metres)

Z is used only internally to intersect an image ray with the local road plane
``Z = 0``. The returned BEV lane points and cubic models are strictly 2-D.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    width: int = 1920
    height: int = 1080
    horizontal_fov_deg: float = 30.0
    fx: float = 3582.768775
    fy: float = 3582.768775
    cx: float = 960.0
    cy: float = 540.0
    camera_to_vehicle: tuple = (
        (0.997564, 0.0, 0.069756, 1.0),
        (0.0, 1.0, 0.0, 0.0),
        (-0.069756, 0.0, 0.997564, 1.8),
        (0.0, 0.0, 0.0, 1.0),
    )

    @property
    def intrinsic(self):
        return np.array(
            ((self.fx, 0.0, self.cx),
             (0.0, self.fy, self.cy),
             (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    @property
    def camera_to_vehicle_matrix(self):
        return np.asarray(self.camera_to_vehicle, dtype=np.float64)


@dataclass(frozen=True)
class BevRange:
    x_min: float = 0.0
    x_max: float = 300.0
    y_min: float = -85.0
    y_max: float = 85.0


@lru_cache(maxsize=8)
def _projection_constants(calibration):
    optical_to_ue = np.array(
        ((0.0, 0.0, 1.0),
         (1.0, 0.0, 0.0),
         (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    mount = calibration.camera_to_vehicle_matrix
    return (
        np.linalg.inv(calibration.intrinsic),
        optical_to_ue,
        mount[:3, 3].copy(),
        mount[:3, :3].copy(),
    )


def pixels_to_ground(pixels, calibration=CameraCalibration(),
                     bev_range=BevRange()):
    """Project raw-image pixels onto the local ground plane.

    Args:
        pixels: ``(N, 2)`` raw-image coordinates ``(u, v)``.
        calibration: pinhole intrinsics and fixed camera-to-vehicle mount.
        bev_range: accepted metric ROI.

    Returns:
        ``(points_xy, valid_mask)``. ``points_xy`` contains only valid points in
        ADAS coordinates ``(X forward, Y left)``. ``valid_mask`` indexes the
        input array and is useful for carrying per-point scores through IPM.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")

    finite = np.isfinite(pixels).all(axis=1)
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    inverse_intrinsic, optical_to_ue, origin_vehicle, rotation = (
        _projection_constants(calibration)
    )
    rays_optical = (inverse_intrinsic @ homogeneous.T).T

    # CARLA/UE camera axes are X forward, Y right, Z up. OpenCV optical axes
    # are x right, y down, z forward: optical -> UE = (z, x, -y).
    rays_camera_ue = (optical_to_ue @ rays_optical.T).T
    rays_vehicle = (rotation @ rays_camera_ue.T).T
    dz = rays_vehicle[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = -origin_vehicle[2] / dz
        ground_vehicle = origin_vehicle + scale[:, None] * rays_vehicle

    # CARLA Y points right; public BEV Y points left.
    x_forward = ground_vehicle[:, 0]
    y_left = -ground_vehicle[:, 1]
    valid = (
        finite
        & np.isfinite(ground_vehicle).all(axis=1)
        & (dz < -1e-8)
        & (scale > 0.0)
        & (x_forward >= bev_range.x_min)
        & (x_forward <= bev_range.x_max)
        & (y_left >= bev_range.y_min)
        & (y_left <= bev_range.y_max)
    )
    return np.column_stack((x_forward[valid], y_left[valid])), valid


def ground_to_pixels(points_xy, calibration=CameraCalibration()):
    """Project ADAS ground points to raw pixels (used for calibration tests)."""
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    # ADAS Y-left -> CARLA Y-right.
    points_vehicle = np.column_stack(
        (points_xy[:, 0], -points_xy[:, 1], np.zeros(len(points_xy)))
    )
    mount = calibration.camera_to_vehicle_matrix
    rotation_vehicle_to_camera = mount[:3, :3].T
    points_camera_ue = (
        rotation_vehicle_to_camera
        @ (points_vehicle - mount[:3, 3]).T
    ).T
    points_optical = np.column_stack(
        (points_camera_ue[:, 1], -points_camera_ue[:, 2], points_camera_ue[:, 0])
    )
    projected = (calibration.intrinsic @ points_optical.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / projected[:, 2:3]


def model_lane_to_ground(lane, model_width=800, model_height=320,
                         calibration=CameraCalibration(),
                         bev_range=BevRange()):
    """Convert one decoded RCLane polyline into metric BEV points + scores."""
    if len(lane.points) == 0:
        return np.empty((0, 2)), np.empty(0)
    lane_points = np.asarray(lane.points, dtype=np.float64)
    raw_pixels = lane_points[:, :2].copy()
    raw_pixels[:, 0] *= calibration.width / float(model_width)
    raw_pixels[:, 1] *= calibration.height / float(model_height)
    ground, valid = pixels_to_ground(raw_pixels, calibration, bev_range)
    return ground, lane_points[valid, 2]


def _weighted_lstsq(design, targets, weights):
    root_weight = np.sqrt(np.maximum(weights, 1e-8))
    return np.linalg.lstsq(
        design * root_weight[:, None], targets * root_weight, rcond=None
    )[0]


def fit_cubic_lane(points_xy, point_scores=None, min_points=6,
                   huber_delta_m=0.30, iterations=5):
    """Robustly fit ``Y(X) = c0 + c1 X + c2 X^2 + c3 X^3``.

    X is normalized internally for numerical stability out to 300 metres. The
    returned coefficients are converted back to metric X and ordered
    ``[c0, c1, c2, c3]``.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < min_points:
        return None
    order = np.argsort(points[:, 0])
    x = points[order, 0]
    y = points[order, 1]
    if np.ptp(x) < 1.0:
        return None

    if point_scores is None:
        base_weights = np.ones(len(x), dtype=np.float64)
    else:
        scores = np.asarray(point_scores, dtype=np.float64)[finite][order]
        base_weights = np.clip(scores, 0.05, 1.0)

    center = float((x.min() + x.max()) * 0.5)
    scale = float(max((x.max() - x.min()) * 0.5, 1.0))
    z = (x - center) / scale
    design = np.column_stack((np.ones(len(z)), z, z ** 2, z ** 3))
    weights = base_weights.copy()
    coefficients_normalized = _weighted_lstsq(design, y, weights)
    for _ in range(iterations):
        residual = y - design @ coefficients_normalized
        robust_weight = np.minimum(
            1.0, huber_delta_m / np.maximum(np.abs(residual), 1e-8)
        )
        weights = base_weights * robust_weight
        coefficients_normalized = _weighted_lstsq(design, y, weights)

    residual = y - design @ coefficients_normalized
    inliers = np.abs(residual) <= max(huber_delta_m, 2.5 * np.median(np.abs(residual)))
    if np.count_nonzero(inliers) >= min_points:
        coefficients_normalized = _weighted_lstsq(
            design[inliers], y[inliers], base_weights[inliers]
        )
    else:
        inliers = np.ones(len(x), dtype=bool)

    # Compose p((X - center) / scale) and return ascending metric coefficients.
    normalized_polynomial = np.polynomial.Polynomial(coefficients_normalized)
    metric_argument = np.polynomial.Polynomial((-center / scale, 1.0 / scale))
    metric_polynomial = normalized_polynomial(metric_argument)
    coefficients = np.zeros(4, dtype=np.float64)
    coefficients[:len(metric_polynomial.coef)] = metric_polynomial.coef

    fitted = np.polynomial.polynomial.polyval(x[inliers], coefficients)
    rmse = float(np.sqrt(np.mean((y[inliers] - fitted) ** 2)))
    return {
        "coefficients": coefficients,
        "x_min": float(x[inliers].min()),
        "x_max": float(x[inliers].max()),
        "rmse": rmse,
        "point_count": int(len(x)),
        "inlier_count": int(np.count_nonzero(inliers)),
        "inlier_ratio": float(np.mean(inliers)),
    }


def evaluate_cubic(coefficients, x):
    return np.polynomial.polynomial.polyval(
        np.asarray(x, dtype=np.float64), np.asarray(coefficients, dtype=np.float64)
    )


def _copy_fit(fit):
    copied = dict(fit)
    copied["coefficients"] = np.asarray(
        fit["coefficients"], dtype=np.float64
    ).copy()
    return copied


def clip_cubic_fit_to_funnel(fit, camera_x_m=1.0,
                             horizontal_fov_deg=30.0, margin_m=0.10,
                             sample_step_m=0.5, minimum_span_m=1.0):
    """Restrict a cubic's declared domain to the visible camera funnel.

    The polynomial coefficients are left unchanged. Only the longest
    contiguous, physically visible X interval is exported, which makes the
    ``x_domain_m`` contract explicit and prevents a renderer from drawing an
    otherwise valid cubic after it leaves the camera footprint.
    """
    copied = _copy_fit(fit)
    x_min = float(fit["x_min"])
    x_max = float(fit["x_max"])
    report = {
        "original_x_domain_m": [x_min, x_max],
        "x_domain_m": None,
        "clipped": False,
        "valid": False,
    }
    if x_max - x_min < minimum_span_m:
        return None, report
    count = max(3, int(np.ceil((x_max - x_min) / sample_step_m)) + 1)
    x = np.linspace(x_min, x_max, count)
    y = evaluate_cubic(fit["coefficients"], x)
    half_width = np.maximum(0.0, x - camera_x_m) * np.tan(
        np.deg2rad(horizontal_fov_deg * 0.5)
    )
    valid = (
        np.isfinite(y)
        & (x >= camera_x_m)
        & (np.abs(y) <= half_width + margin_m)
    )

    runs = []
    start = None
    for index, value in enumerate(valid):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(valid) - 1):
            end = index if value and index == len(valid) - 1 else index - 1
            if x[end] - x[start] >= minimum_span_m:
                runs.append((start, end))
            start = None
    if not runs:
        return None, report
    start, end = max(runs, key=lambda run: x[run[1]] - x[run[0]])
    copied["x_min"] = float(x[start])
    copied["x_max"] = float(x[end])
    report.update({
        "x_domain_m": [copied["x_min"], copied["x_max"]],
        "clipped": bool(
            copied["x_min"] > x_min + 1e-6
            or copied["x_max"] < x_max - 1e-6
        ),
        "valid": True,
    })
    return copied, report


def _self_test():
    calibration = CameraCalibration()
    bev_range = BevRange()
    ground = np.array(
        ((5.0, 0.0), (20.0, 3.5), (50.0, -3.5), (150.0, 2.0),
         (299.0, 0.0)),
        dtype=np.float64,
    )
    pixels = ground_to_pixels(ground, calibration)
    reconstructed, valid = pixels_to_ground(pixels, calibration, bev_range)
    assert valid.all()
    assert np.allclose(reconstructed, ground, atol=1e-5), (
        reconstructed, ground
    )

    x = np.linspace(5.0, 180.0, 80)
    truth = np.array((1.8, 1.5e-2, -1.2e-4, 3.0e-7))
    y = evaluate_cubic(truth, x)
    y[20] += 4.0
    fitted = fit_cubic_lane(np.column_stack((x, y)))
    assert fitted is not None
    prediction = evaluate_cubic(fitted["coefficients"], x)
    clean = np.ones(len(x), dtype=bool)
    clean[20] = False
    clean_rmse = np.sqrt(np.mean(
        (prediction[clean] - evaluate_cubic(truth, x[clean])) ** 2
    ))
    assert clean_rmse < 0.02

    runaway_fit = {
        "coefficients": np.array((-5.0, 0.4, -0.013, 1.15e-4)),
        "x_min": 10.0, "x_max": 130.0, "rmse": 0.1,
        "point_count": 50, "inlier_count": 50, "inlier_ratio": 1.0,
    }
    source_coefficients = runaway_fit["coefficients"].copy()
    clipped, funnel_report = clip_cubic_fit_to_funnel(runaway_fit)
    assert clipped is not None and funnel_report["clipped"]
    assert clipped["x_max"] < runaway_fit["x_max"]
    assert np.array_equal(clipped["coefficients"], source_coefficients)
    check_x = np.linspace(clipped["x_min"], clipped["x_max"], 300)
    check_y = evaluate_cubic(clipped["coefficients"], check_x)
    funnel_half_width = (check_x - 1.0) * np.tan(np.deg2rad(15.0))
    assert np.all(np.abs(check_y) <= funnel_half_width + 0.11)
    print("OK -- pixel/ground projection round trip")
    print("OK -- robust metric cubic lane fit")
    print("OK -- raw cubic funnel clipping preserves coefficients")


if __name__ == "__main__":
    _self_test()
