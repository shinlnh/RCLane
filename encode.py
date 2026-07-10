"""
RCLane label encoder -- ported to PyTorch pipeline from src/lane_codec.py::encode
(MindSpore repo). Pure NumPy + shapely, framework-agnostic.

Converts lane annotations (a list of polylines) into the 5 dense ground-truth maps
that the network is trained to predict:
    seg_map, up_arrow, down_arrow, up_bound, down_bound

This is a training-time preprocessing step (run per image, ideally cached), NOT part
of the network. It is the exact counterpart of the relay-chain decode.

Algorithm (per foreground pixel of a thick lane raster):
  1. Draw a circle of radius `step_length` around the pixel.
  2. Intersect the circle boundary with the nearest lane -> if it yields 2 points
     (MultiPoint), the pixel is a "fine" foreground point.
  3. The two intersection vectors become up_arrow / down_arrow (by smaller / larger y).
  4. Vectors to the lane endpoints give the up/down directions; a circle spanning
     pixel->endpoint intersected with the lane gives the arc length to the endpoint,
     stored (scaled) as up_bound / down_bound.

Output layout (matches rclane.py / loss.py, NCHW-friendly):
    seg_map:                 (H, W)      float32 in {0, 1}
    up/down_arrow/bound:     (2, H, W)   float32
"""

import numpy as np
import cv2
from shapely.geometry import Point, LineString, MultiPoint  # noqa: F401


# Defaults from the original default_config.yaml
IMG_SIZE = (320, 800)   # (H, W)
STEP_LENGTH = 10
LINE_WIDTH = 5
SEG_THRESHOLD = 0.5
BOUND_SCALE = 100.0     # bound value = arc_length / BOUND_SCALE + 1  (original uses 100)
BUFFER_QUAD_SEGS = 64   # circle smoothness (original used resolution=100)


def _to_linestrings(lanes_points):
    """lanes_points: list of lanes, each a list/array of (x, y). -> list of LineString."""
    out = []
    for pts in lanes_points:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and len(pts) >= 2:
            out.append(LineString(pts))
    return out


def _rasterize(linestrings, H, W, line_width):
    """Thick lane raster (coarse foreground mask), like cv2.polylines in the original."""
    mask = np.zeros((H, W), np.uint8)
    for ls in linestrings:
        xs, ys = ls.xy
        pts = np.stack([np.asarray(xs), np.asarray(ys)], axis=1).astype(np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=line_width)
    return (mask > 10).astype(np.float32)


def _points_of(geom):
    """Flatten a shapely geometry into a list of (x, y) points."""
    if geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Point":
        return [(geom.x, geom.y)]
    if gt == "MultiPoint":
        return [(p.x, p.y) for p in geom.geoms]
    if gt == "GeometryCollection":
        pts = []
        for g in geom.geoms:
            if g.geom_type == "Point":
                pts.append((g.x, g.y))
        return pts
    return []


def _up_down(points, center):
    """Split vectors (point - center) into the one with smallest y (up) and largest y (down)."""
    c = np.asarray(center, dtype=float)
    vecs = [np.asarray(p, dtype=float) - c for p in points]
    up = min(vecs, key=lambda v: v[1])
    down = max(vecs, key=lambda v: v[1])
    return up, down


def _arc_length(center_xy, delta, nearest):
    """
    Length of the lane arc between the pixel and an endpoint, approximated by the
    lane portion inside a circle whose diameter is that pixel->endpoint segment.
    Falls back to the straight-line half distance on failure (as in the original).
    """
    half = np.asarray(delta, dtype=float) / 2.0
    r = float(np.hypot(half[0], half[1]))
    if r <= 0:
        return 0.0
    ctr = np.asarray(center_xy, dtype=float) + half
    try:
        circle = Point(ctr[0], ctr[1]).buffer(r, quad_segs=BUFFER_QUAD_SEGS)
        length = circle.intersection(nearest).length
        if length == 0:
            length = r
    except Exception:
        length = r
    return length


def encode(lanes_points, img_size=IMG_SIZE, step_length=STEP_LENGTH,
           line_width=LINE_WIDTH, seg_threshold=SEG_THRESHOLD):
    """
    Args:
        lanes_points: list of lanes, each a list/array of (x, y) in the target
            image space (default 800 wide x 320 high).
    Returns:
        dict with 'seg_map' (H,W) and 'up_arrow'/'down_arrow'/'up_bound'/'down_bound'
        each (2, H, W), all float32.
    """
    H, W = img_size
    seg = np.zeros((H, W), np.float32)
    up_arrow = np.zeros((H, W, 2), np.float32)
    down_arrow = np.zeros((H, W, 2), np.float32)
    up_bound = np.zeros((H, W, 2), np.float32)
    down_bound = np.zeros((H, W, 2), np.float32)

    lanes = _to_linestrings(lanes_points)
    if len(lanes) > 0:
        coarse = _rasterize(lanes, H, W, line_width)
        ys, xs = np.where(coarse > seg_threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cp = Point(float(x), float(y))
            ring = cp.buffer(step_length, quad_segs=BUFFER_QUAD_SEGS).exterior
            nearest = min(lanes, key=lambda l: l.distance(cp))
            inter = ring.intersection(nearest)
            if inter.geom_type != "MultiPoint":
                continue  # near an endpoint the circle hits the lane once -> skip
            pts = _points_of(inter)
            if len(pts) < 2:
                continue

            seg[y, x] = 1.0
            u, d = _up_down(pts, (x, y))
            up_arrow[y, x] = u          # (dx, dy) toward the upper intersection
            down_arrow[y, x] = d        # (dx, dy) toward the lower intersection

            end_pts = _points_of(nearest.boundary)  # the 2 lane endpoints
            if len(end_pts) >= 2:
                u_end, d_end = _up_down(end_pts, (x, y))
            else:
                u_end, d_end = u, d
            up_bound[y, x] = _arc_length((x, y), u_end, nearest) / BOUND_SCALE + 1
            down_bound[y, x] = _arc_length((x, y), d_end, nearest) / BOUND_SCALE + 1

    chw = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1))  # (H,W,2) -> (2,H,W)
    return {
        "seg_map": seg,
        "up_arrow": chw(up_arrow),
        "down_arrow": chw(down_arrow),
        "up_bound": chw(up_bound),
        "down_bound": chw(down_bound),
    }


# --------------------------------------------------------------------------- #
#  smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    H, W = IMG_SIZE

    # a single curved lane crossing the image (points in 800x320 space)
    ys = np.linspace(20, 300, 40)
    xs = 400 + 180 * np.sin(ys / 300 * np.pi)   # S-curve
    lane = list(zip(xs.tolist(), ys.tolist()))

    gt = encode([lane])

    print("shapes:")
    print(f"  seg_map    {gt['seg_map'].shape}")
    for k in ("up_arrow", "down_arrow", "up_bound", "down_bound"):
        print(f"  {k:10s} {gt[k].shape}")

    n_fg = int(gt["seg_map"].sum())
    print(f"foreground pixels: {n_fg}")
    assert n_fg > 0, "no foreground produced!"

    ys_fg, xs_fg = np.where(gt["seg_map"] > 0.5)
    up_dy = gt["up_arrow"][1][ys_fg, xs_fg]     # y-component of up arrow
    down_dy = gt["down_arrow"][1][ys_fg, xs_fg]  # y-component of down arrow
    # up arrow should point up (dy < 0), down arrow down (dy > 0), on average
    print(f"mean up_arrow.dy   = {up_dy.mean():+.2f}  (expect < 0)")
    print(f"mean down_arrow.dy = {down_dy.mean():+.2f}  (expect > 0)")
    assert up_dy.mean() < 0 < down_dy.mean(), "arrow directions look wrong!"

    ub = gt["up_bound"][0][ys_fg, xs_fg]
    print(f"up_bound range     = [{ub.min():.2f}, {ub.max():.2f}]  (>= 1)")
    assert ub.min() >= 1.0, "bound values must be >= 1 (scaled + 1 offset)!"

    # arrow step length should be ~ step_length
    step = np.hypot(gt["up_arrow"][0][ys_fg, xs_fg], up_dy)
    print(f"mean up_arrow length = {step.mean():.2f}  (expect ~= {STEP_LENGTH})")

    print("OK -- encode produces sane GT maps.")
