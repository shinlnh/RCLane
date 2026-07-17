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


def evaluate_cubic_derivative(coefficients, x):
    """Evaluate dY/dX for an ascending-order cubic coefficient vector."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    derivative = np.array(
        (coefficients[1], 2.0 * coefficients[2], 3.0 * coefficients[3]),
        dtype=np.float64,
    )
    return np.polynomial.polynomial.polyval(
        np.asarray(x, dtype=np.float64), derivative
    )


def _copy_fit(fit):
    copied = dict(fit)
    copied["coefficients"] = np.asarray(
        fit["coefficients"], dtype=np.float64
    ).copy()
    return copied


def _shared_samples(left_fit, right_fit, sample_step_m):
    x_min = max(float(left_fit["x_min"]), float(right_fit["x_min"]))
    x_max = min(float(left_fit["x_max"]), float(right_fit["x_max"]))
    if x_max - x_min < sample_step_m:
        return np.empty(0, dtype=np.float64)
    count = max(3, int(np.ceil((x_max - x_min) / sample_step_m)) + 1)
    return np.linspace(x_min, x_max, count, dtype=np.float64)


def _normal_gap(left_fit, right_fit, x):
    """Approximate signed left-to-right distance normal to the mean curve."""
    left_y = evaluate_cubic(left_fit["coefficients"], x)
    right_y = evaluate_cubic(right_fit["coefficients"], x)
    left_slope = evaluate_cubic_derivative(left_fit["coefficients"], x)
    right_slope = evaluate_cubic_derivative(right_fit["coefficients"], x)
    mean_slope = 0.5 * (left_slope + right_slope)
    return (left_y - right_y) / np.sqrt(1.0 + mean_slope ** 2)


def _longest_true_run_m(mask, x):
    longest = 0.0
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            if end > start:
                longest = max(longest, float(x[end] - x[start]))
            start = None
    return longest


def _estimate_lane_width(models, nominal_lane_width_m, sample_step_m):
    """Estimate one marking-to-marking width from healthy near-range gaps."""
    candidates = []
    for left, right in zip(models, models[1:]):
        lane_steps = right["lane_index"] - left["lane_index"]
        if lane_steps <= 0:
            continue
        x = _shared_samples(left["fit"], right["fit"], sample_step_m)
        if len(x) < 3:
            continue
        # Prefer the closest 40%: this is where IPM is best conditioned and
        # where decoded markings are least likely to have merged at a horizon.
        near_count = max(3, int(np.ceil(len(x) * 0.40)))
        per_lane_gap = _normal_gap(
            left["fit"], right["fit"], x[:near_count]
        ) / lane_steps
        plausible = per_lane_gap[
            np.isfinite(per_lane_gap)
            & (per_lane_gap >= 2.4)
            & (per_lane_gap <= 5.0)
        ]
        if len(plausible):
            candidates.append(float(np.median(plausible)))
    if not candidates:
        return float(nominal_lane_width_m), "nominal"
    estimated = float(np.median(candidates))
    return float(np.clip(estimated, 2.6, 4.5)), "near_range_median"


def analyze_lane_topology(models, lane_width_m, trigger_gap_ratio=0.55,
                          minimum_gap_m=1.0, minimum_bad_run_m=4.0,
                          sample_step_m=0.5):
    """Inspect ordered BEV cubics for collapsed gaps or lane crossings."""
    pair_reports = []
    trigger_pairs = []
    for left, right in zip(models, models[1:]):
        lane_steps = right["lane_index"] - left["lane_index"]
        if lane_steps <= 0:
            continue
        x = _shared_samples(left["fit"], right["fit"], sample_step_m)
        if len(x) < 3:
            continue
        gap = _normal_gap(left["fit"], right["fit"], x)
        expected_gap = lane_width_m * lane_steps
        trigger_gap = max(minimum_gap_m, trigger_gap_ratio * expected_gap)
        finite = np.isfinite(gap)
        bad = finite & (gap < trigger_gap)
        crossing = bool(np.any(finite & (gap <= 0.0)))
        longest_bad_run = _longest_true_run_m(bad, x)
        triggered = crossing or longest_bad_run >= minimum_bad_run_m
        report = {
            "left_lane": f"P{left['lane_index']}",
            "right_lane": f"P{right['lane_index']}",
            "lane_steps": int(lane_steps),
            "shared_x_domain_m": [float(x[0]), float(x[-1])],
            "expected_gap_m": float(expected_gap),
            "trigger_gap_m": float(trigger_gap),
            "minimum_gap_m": float(np.min(gap[finite])) if finite.any() else None,
            "longest_bad_run_m": float(longest_bad_run),
            "crossing": crossing,
            "triggered": bool(triggered),
        }
        pair_reports.append(report)
        if triggered:
            trigger_pairs.append(
                f"P{left['lane_index']}-P{right['lane_index']}"
            )
    return pair_reports, trigger_pairs


def _parallel_offset_fit(reference_fit, target_fit, offset_m,
                         sample_step_m=0.5):
    """Build a metric normal-offset curve and refit it as a cubic Y(X)."""
    # Never extrapolate a reference polynomial beyond the metric interval in
    # which it was fitted. High-order extrapolation was observed to turn one
    # short reference lane into a physically impossible far-range curve.
    x_min = max(float(reference_fit["x_min"]), float(target_fit["x_min"]))
    x_max = min(float(reference_fit["x_max"]), float(target_fit["x_max"]))
    span = x_max - x_min
    if span < 1.0:
        return None
    count = max(16, int(np.ceil(span / sample_step_m)) + 1)
    x = np.linspace(x_min, x_max, count)
    y = evaluate_cubic(reference_fit["coefficients"], x)
    slope = evaluate_cubic_derivative(reference_fit["coefficients"], x)
    norm = np.sqrt(1.0 + slope ** 2)
    offset_points = np.column_stack((
        x - offset_m * slope / norm,
        y + offset_m / norm,
    ))
    return fit_cubic_lane(
        offset_points, np.ones(len(offset_points)), min_points=6,
        huber_delta_m=0.05, iterations=3,
    )


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


def repair_parallel_lane_fits(models, nominal_lane_width_m=3.5,
                              trigger_gap_ratio=0.55,
                              minimum_gap_m=1.0,
                              minimum_bad_run_m=4.0,
                              sample_step_m=0.5,
                              maximum_reference_extrapolation_m=2.0,
                              force_repair=False):
    """Repair overlapping BEV lanes using one confidence-selected reference.

    ``models`` is a sequence of dictionaries with ``lane_index``, ``score``
    and a valid cubic ``fit``. If any adjacent pair collapses for a sustained
    distance (or crosses at all), the highest-confidence cubic is retained and
    all other detected markings are rebuilt as metric normal offsets. Setting
    ``force_repair`` applies that same constraint even when topology analysis
    finds no collapse. The function returns replacement fits plus a
    JSON-serializable audit report.

    This is deliberately a BEV-only topology prior. It does not alter decoded
    image-space polylines and does not back-project synthetic curves.
    """
    ordered = sorted(
        (model for model in models if model.get("fit") is not None),
        key=lambda model: model["lane_index"],
    )
    output_fits = {
        model["lane_index"]: _copy_fit(model["fit"]) for model in ordered
    }
    report = {
        "applied": False,
        "forced": bool(force_repair),
        "activation": "always_parallel" if force_repair else "triggered",
        "scope": "bev_only",
        "reference_lane": None,
        "reference_score": None,
        "reference_selection": None,
        "reference_domain_m": None,
        "required_domain_m": None,
        "lane_width_m": float(nominal_lane_width_m),
        "lane_width_source": "nominal",
        "trigger_pairs": [],
        "pairs_before": [],
        "pairs_after": [],
        "method": None,
        "offsets_m": {},
        "validation_passed": True,
    }
    if len(ordered) < 2:
        return output_fits, report

    lane_width_m, width_source = _estimate_lane_width(
        ordered, nominal_lane_width_m, sample_step_m
    )
    pairs_before, trigger_pairs = analyze_lane_topology(
        ordered, lane_width_m, trigger_gap_ratio, minimum_gap_m,
        minimum_bad_run_m, sample_step_m,
    )
    report.update({
        "lane_width_m": float(lane_width_m),
        "lane_width_source": width_source,
        "trigger_pairs": trigger_pairs,
        "pairs_before": pairs_before,
    })
    if not trigger_pairs and not force_repair:
        report["pairs_after"] = pairs_before
        return output_fits, report

    required_x_min = min(float(model["fit"]["x_min"]) for model in ordered)
    required_x_max = max(float(model["fit"]["x_max"]) for model in ordered)

    def uncovered_distance(model):
        fit = model["fit"]
        return max(
            max(0.0, float(fit["x_min"]) - required_x_min),
            max(0.0, required_x_max - float(fit["x_max"])),
        )

    coverage_candidates = [
        model for model in ordered
        if uncovered_distance(model) <= maximum_reference_extrapolation_m
    ]

    def shared_domain_support(model):
        fit = model["fit"]
        return sum(
            max(
                0.0,
                min(float(fit["x_max"]), float(other["fit"]["x_max"]))
                - max(float(fit["x_min"]), float(other["fit"]["x_min"])),
            )
            for other in ordered
            if other is not model
        )

    if force_repair:
        # Always-parallel mode must be able to derive every target from the
        # reference on a real shared X interval. Union coverage alone can pick
        # a long outer lane whose domain does not overlap a short inner lane.
        reference = max(
            ordered,
            key=lambda model: (
                shared_domain_support(model),
                float(model["score"]),
                float(model["fit"]["x_max"])
                - float(model["fit"]["x_min"]),
                -model["lane_index"],
            ),
        )
        reference_selection = "maximum_shared_domain_then_confidence"
    elif coverage_candidates:
        # Confidence remains the exact primary criterion, but only among lanes
        # whose fitted domain can safely support the requested repair.
        reference = max(
            coverage_candidates,
            key=lambda model: (float(model["score"]), -model["lane_index"]),
        )
        reference_selection = "highest_confidence_with_domain_coverage"
    else:
        # No lane spans the union. Prefer the least uncovered candidate and
        # truncate every generated target to the actual shared domain below.
        reference = min(
            ordered,
            key=lambda model: (
                uncovered_distance(model),
                -float(model["score"]),
                model["lane_index"],
            ),
        )
        reference_selection = "best_domain_coverage_then_confidence"
    reference_id = reference["lane_index"]
    offsets = {
        model["lane_index"]: float(
            (reference_id - model["lane_index"]) * lane_width_m
        )
        for model in ordered
    }
    repaired = {}
    normal_fit_ok = True
    for model in ordered:
        lane_id = model["lane_index"]
        if lane_id == reference_id:
            repaired[lane_id] = _copy_fit(reference["fit"])
            continue
        fit = _parallel_offset_fit(
            reference["fit"], model["fit"], offsets[lane_id], sample_step_m
        )
        if fit is None:
            normal_fit_ok = False
            break
        repaired[lane_id] = fit

    method = "normal_offset_cubic"
    passthrough_lane_ids = set()
    if normal_fit_ok:
        repaired_models = [
            {**model, "fit": repaired[model["lane_index"]]}
            for model in ordered
        ]
        pairs_after, remaining_triggers = analyze_lane_topology(
            repaired_models, lane_width_m, trigger_gap_ratio,
            minimum_gap_m, minimum_bad_run_m, sample_step_m,
        )
    else:
        remaining_triggers = ["normal_offset_fit_failed"]
        pairs_after = []

    # Independent cubic approximation of exact normal offsets can very rarely
    # reintroduce a far-range crossing. A shared-shape vertical translation is
    # the deterministic safety net: identical c1..c3 means crossings are
    # mathematically impossible over every shared X domain.
    if remaining_triggers:
        method = "shared_shape_vertical_offset_fallback"
        repaired = {}
        for model in ordered:
            lane_id = model["lane_index"]
            fit = _copy_fit(model["fit"])
            fit["x_min"] = max(
                float(reference["fit"]["x_min"]), float(fit["x_min"])
            )
            fit["x_max"] = min(
                float(reference["fit"]["x_max"]), float(fit["x_max"])
            )
            if fit["x_max"] - fit["x_min"] < 1.0:
                # Never publish an inverted synthetic domain. With no shared
                # support, retain the measured target fit; it cannot overlap
                # the reference inside BEV because their X domains are disjoint.
                repaired[lane_id] = _copy_fit(model["fit"])
                passthrough_lane_ids.add(lane_id)
                continue
            fit["coefficients"] = np.asarray(
                reference["fit"]["coefficients"], dtype=np.float64
            ).copy()
            fit["coefficients"][0] += offsets[lane_id]
            fit["rmse"] = 0.0
            fit["repair_fit_rmse"] = 0.0
            repaired[lane_id] = fit
        repaired_models = [
            {**model, "fit": repaired[model["lane_index"]]}
            for model in ordered
        ]
        pairs_after, remaining_triggers = analyze_lane_topology(
            repaired_models, lane_width_m, trigger_gap_ratio,
            minimum_gap_m, minimum_bad_run_m, sample_step_m,
        )

    validation_passed = not remaining_triggers and all(
        pair["minimum_gap_m"] is None or pair["minimum_gap_m"] > 0.0
        for pair in pairs_after
    )
    if not validation_passed:
        # Never publish an unvalidated synthetic topology.
        return output_fits, {
            **report,
            "reference_lane": f"P{reference_id}",
            "reference_score": float(reference["score"]),
            "reference_selection": reference_selection,
            "reference_domain_m": [
                float(reference["fit"]["x_min"]),
                float(reference["fit"]["x_max"]),
            ],
            "required_domain_m": [required_x_min, required_x_max],
            "method": method,
            "offsets_m": {f"P{k}": v for k, v in offsets.items()},
            "pairs_after": pairs_after,
            "validation_passed": False,
            "failure": "repaired topology did not pass ordering validation",
        }

    for lane_id, fit in repaired.items():
        if lane_id in passthrough_lane_ids:
            continue
        fit["parallel_repair"] = True
        fit["parallel_reference_lane"] = reference_id
        fit["parallel_offset_m"] = offsets[lane_id]
        fit["parallel_repair_method"] = method
    report.update({
        "applied": True,
        "reference_lane": f"P{reference_id}",
        "reference_score": float(reference["score"]),
        "reference_selection": reference_selection,
        "reference_domain_m": [
            float(reference["fit"]["x_min"]),
            float(reference["fit"]["x_max"]),
        ],
        "required_domain_m": [required_x_min, required_x_max],
        "method": method,
        "offsets_m": {f"P{k}": v for k, v in offsets.items()},
        "passthrough_lanes": [
            f"P{lane_id}" for lane_id in sorted(passthrough_lane_ids)
        ],
        "pairs_after": pairs_after,
        "validation_passed": True,
    })
    return repaired, report


def complete_four_parallel_lane_fits(models, nominal_lane_width_m=3.5,
                                     sample_step_m=0.5):
    """Return a complete parallel P0/P1/P2/P3 BEV lane set.

    Unlike :func:`repair_parallel_lane_fits`, this display/output mode does not
    require all four markings to have been detected. It chooses the measured
    fit with the greatest shared longitudinal support, retains it as the
    reference, and constructs every lane index in ``[0, 3]`` as a metric normal
    offset. Missing indices are explicitly reported as synthetic.

    Camera-funnel clipping is intentionally not performed here. The caller may
    display these completed geometric priors outside the current camera field
    of view while keeping image-space detections untouched.
    """
    ordered = sorted(
        (model for model in models if model.get("fit") is not None),
        key=lambda model: model["lane_index"],
    )
    source_ids = {int(model["lane_index"]) for model in ordered}
    target_ids = tuple(range(4))
    synthetic_ids = [lane_id for lane_id in target_ids if lane_id not in source_ids]
    report = {
        "applied": False,
        "forced": True,
        "activation": "complete_four_parallel",
        "scope": "bev_only",
        "reference_lane": None,
        "reference_score": None,
        "reference_selection": "maximum_shared_domain_then_confidence",
        "reference_domain_m": None,
        "required_domain_m": None,
        "lane_width_m": float(nominal_lane_width_m),
        "lane_width_source": "nominal",
        "source_lane_ids": [f"P{i}" for i in sorted(source_ids)],
        "completed_lane_ids": [f"P{i}" for i in target_ids],
        "synthetic_lane_ids": [f"P{i}" for i in synthetic_ids],
        "trigger_pairs": [],
        "pairs_before": [],
        "pairs_after": [],
        "method": None,
        "offsets_m": {},
        "passthrough_lanes": [],
        "validation_passed": False,
    }
    if not ordered:
        report["failure"] = "no valid measured BEV fit is available"
        return {}, report

    if len(ordered) >= 2:
        lane_width_m, width_source = _estimate_lane_width(
            ordered, nominal_lane_width_m, sample_step_m
        )
        pairs_before, trigger_pairs = analyze_lane_topology(
            ordered, lane_width_m, sample_step_m=sample_step_m
        )
    else:
        lane_width_m = float(nominal_lane_width_m)
        width_source = "nominal_single_reference"
        pairs_before, trigger_pairs = [], []

    def shared_domain_support(model):
        fit = model["fit"]
        return sum(
            max(
                0.0,
                min(float(fit["x_max"]), float(other["fit"]["x_max"]))
                - max(float(fit["x_min"]), float(other["fit"]["x_min"])),
            )
            for other in ordered
            if other is not model
        )

    reference = max(
        ordered,
        key=lambda model: (
            shared_domain_support(model),
            float(model["score"]),
            float(model["fit"]["x_max"])
            - float(model["fit"]["x_min"]),
            -int(model["lane_index"]),
        ),
    )
    reference_id = int(reference["lane_index"])
    reference_fit = reference["fit"]
    offsets = {
        lane_id: float((reference_id - lane_id) * lane_width_m)
        for lane_id in target_ids
    }

    completed = {}
    normal_fit_ok = True
    for lane_id in target_ids:
        if lane_id == reference_id:
            completed[lane_id] = _copy_fit(reference_fit)
            continue
        fit = _parallel_offset_fit(
            reference_fit, reference_fit, offsets[lane_id], sample_step_m
        )
        if fit is None:
            normal_fit_ok = False
            break
        completed[lane_id] = fit

    method = "normal_offset_cubic"
    if normal_fit_ok:
        completed_models = [
            {"lane_index": lane_id, "score": 1.0, "fit": completed[lane_id]}
            for lane_id in target_ids
        ]
        pairs_after, remaining_triggers = analyze_lane_topology(
            completed_models, lane_width_m, sample_step_m=sample_step_m
        )
    else:
        pairs_after = []
        remaining_triggers = ["normal_offset_fit_failed"]

    if remaining_triggers:
        # Identical c1..c3 with constant c0 spacing guarantees four ordered,
        # non-crossing lanes over the complete reference domain.
        method = "shared_shape_vertical_offset_fallback"
        completed = {}
        for lane_id in target_ids:
            fit = _copy_fit(reference_fit)
            fit["coefficients"][0] += offsets[lane_id]
            fit["rmse"] = 0.0
            fit["repair_fit_rmse"] = 0.0
            completed[lane_id] = fit
        completed_models = [
            {"lane_index": lane_id, "score": 1.0, "fit": completed[lane_id]}
            for lane_id in target_ids
        ]
        pairs_after, remaining_triggers = analyze_lane_topology(
            completed_models, lane_width_m, sample_step_m=sample_step_m
        )

    validation_passed = not remaining_triggers and all(
        pair["minimum_gap_m"] is None or pair["minimum_gap_m"] > 0.0
        for pair in pairs_after
    )
    required_x_min = min(float(model["fit"]["x_min"]) for model in ordered)
    required_x_max = max(float(model["fit"]["x_max"]) for model in ordered)
    report.update({
        "applied": bool(validation_passed),
        "reference_lane": f"P{reference_id}",
        "reference_score": float(reference["score"]),
        "reference_domain_m": [
            float(reference_fit["x_min"]), float(reference_fit["x_max"])
        ],
        "required_domain_m": [required_x_min, required_x_max],
        "lane_width_m": float(lane_width_m),
        "lane_width_source": width_source,
        "trigger_pairs": trigger_pairs,
        "pairs_before": pairs_before,
        "pairs_after": pairs_after,
        "method": method,
        "offsets_m": {f"P{k}": v for k, v in offsets.items()},
        "validation_passed": bool(validation_passed),
    })
    if not validation_passed:
        report["failure"] = "completed topology did not pass ordering validation"
        return {}, report

    for lane_id, fit in completed.items():
        fit["parallel_repair"] = True
        fit["parallel_reference_lane"] = reference_id
        fit["parallel_offset_m"] = offsets[lane_id]
        fit["parallel_repair_method"] = method
        fit["synthetic_lane"] = lane_id in synthetic_ids
    return completed, report


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
    y[20] += 4.0  # one large synthetic outlier
    fitted = fit_cubic_lane(np.column_stack((x, y)))
    assert fitted is not None
    prediction = evaluate_cubic(fitted["coefficients"], x)
    clean = np.ones(len(x), dtype=bool)
    clean[20] = False
    assert np.sqrt(np.mean((prediction[clean] - evaluate_cubic(truth, x[clean])) ** 2)) < 0.02

    # Four truly parallel curves must remain untouched.
    base = np.array((1.0, 0.03, 1.0e-4, -1.0e-7))
    models = []
    for lane_id in range(4):
        coefficients = base.copy()
        coefficients[0] += (1 - lane_id) * 3.5
        models.append({
            "lane_index": lane_id,
            "score": 0.9 - lane_id * 0.01,
            "fit": {
                "coefficients": coefficients,
                "x_min": 5.0, "x_max": 150.0, "rmse": 0.02,
                "point_count": 80, "inlier_count": 80,
                "inlier_ratio": 1.0,
            },
        })
    unchanged, clean_report = repair_parallel_lane_fits(models)
    assert not clean_report["applied"]
    assert all(
        np.allclose(unchanged[i]["coefficients"], models[i]["fit"]["coefficients"])
        for i in range(4)
    )

    # Always-parallel mode must rebuild even a healthy frame while preserving
    # ordered, non-crossing lane geometry.
    forced, forced_report = repair_parallel_lane_fits(
        models, force_repair=True
    )
    assert forced_report["applied"] and forced_report["forced"]
    forced_models = [
        {**model, "fit": forced[model["lane_index"]]} for model in models
    ]
    forced_pairs, forced_triggers = analyze_lane_topology(
        forced_models, forced_report["lane_width_m"]
    )
    assert not forced_triggers
    assert all(pair["minimum_gap_m"] > 0.0 for pair in forced_pairs)

    # Completing from only the two ego boundaries must synthesize both outer
    # markings and produce a valid four-lane topology.
    completed, complete_report = complete_four_parallel_lane_fits(models[1:3])
    assert complete_report["applied"]
    assert set(completed) == {0, 1, 2, 3}
    assert complete_report["synthetic_lane_ids"] == ["P0", "P3"]
    completed_models = [
        {"lane_index": lane_id, "fit": completed[lane_id]}
        for lane_id in range(4)
    ]
    completed_pairs, completed_triggers = analyze_lane_topology(
        completed_models, complete_report["lane_width_m"]
    )
    assert not completed_triggers
    assert all(pair["minimum_gap_m"] > 0.0 for pair in completed_pairs)

    # Corrupt P3 so it merges into and crosses P2 in the far range.
    corrupted = [{**model, "fit": _copy_fit(model["fit"])} for model in models]
    corrupted[2]["score"] = 0.98  # P2 must become the exact reference.
    corrupted[3]["fit"]["coefficients"] = np.array(
        (-3.2, 0.03, 3.0e-4, 1.5e-6)
    )
    repaired, repair_report = repair_parallel_lane_fits(corrupted)
    assert repair_report["applied"]
    assert repair_report["reference_lane"] == "P2"
    assert repair_report["validation_passed"]
    repaired_models = [
        {**model, "fit": repaired[model["lane_index"]]}
        for model in corrupted
    ]
    repaired_pairs, repaired_triggers = analyze_lane_topology(
        repaired_models, repair_report["lane_width_m"]
    )
    assert not repaired_triggers
    assert all(pair["minimum_gap_m"] > 0.0 for pair in repaired_pairs)

    # A short, high-confidence lane must not beat a slightly lower-confidence
    # reference that actually covers the required far-range repair domain.
    coverage_case = [
        {**model, "fit": _copy_fit(model["fit"])} for model in corrupted
    ]
    coverage_case[0]["score"] = 0.999
    coverage_case[0]["fit"]["x_max"] = 50.0
    coverage_case[1]["fit"]["x_max"] = 100.0
    coverage_case[2]["score"] = 0.80
    coverage_case[3]["fit"]["x_max"] = 100.0
    _, coverage_report = repair_parallel_lane_fits(coverage_case)
    assert coverage_report["applied"]
    assert coverage_report["reference_lane"] == "P2"
    assert coverage_report["reference_selection"] == (
        "highest_confidence_with_domain_coverage"
    )

    runaway_fit = {
        "coefficients": np.array((-5.0, 0.4, -0.013, 1.15e-4)),
        "x_min": 10.0, "x_max": 130.0, "rmse": 0.1,
        "point_count": 50, "inlier_count": 50, "inlier_ratio": 1.0,
    }
    clipped, funnel_report = clip_cubic_fit_to_funnel(runaway_fit)
    assert clipped is not None and funnel_report["clipped"]
    assert clipped["x_max"] < runaway_fit["x_max"]
    check_x = np.linspace(clipped["x_min"], clipped["x_max"], 300)
    check_y = evaluate_cubic(clipped["coefficients"], check_x)
    funnel_half_width = (check_x - 1.0) * np.tan(np.deg2rad(15.0))
    assert np.all(np.abs(check_y) <= funnel_half_width + 0.11)
    print("OK -- pixel/ground projection round trip")
    print("OK -- robust metric cubic lane fit")
    print("OK -- BEV parallel-lane topology repair")
    print("OK -- reference coverage and camera-funnel guard")


if __name__ == "__main__":
    _self_test()
