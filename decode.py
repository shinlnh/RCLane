"""
RCLane relay-chain decoder -- ported from src/lane_codec.py (decode / decode_branch /
thresh_line / iou_nms) and src/lane_geometry.py (PointSelf / FloatLengthLine._iou).

Turns the 5 predicted maps back into lane polylines. This is the inference-time
counterpart of `encode`. Pure NumPy + OpenCV, framework-agnostic.

Pipeline:
  1. seeds = point-NMS on the seg foreground map (keep peaks >= min_dist apart).
  2. for each seed: crawl a forward (down_arrow) and backward (up_arrow) relay chain,
     each step = normalized transfer vector * step_length; stop via the distance
     (bound) heuristic once the walk leaves the foreground.
  3. merge reversed(up) + down into one lane.
  4. drop low-score lanes (thresh_line), then IoU-NMS to remove duplicates.

Note: the stopping rule inside `decode_branch` (RMS of remaining-step estimates,
the 0.75 factor, "keep going while on foreground") is NOT described in the paper --
it is ported verbatim from the MindSpore code. A `norm == 0` guard is added so a
zero transfer vector breaks the walk instead of producing NaNs.

Map layout (matches encode.py / rclane.py, channel-first):
    seg_prob:              (H, W)      foreground probability in [0, 1]
    up/down_arrow/bound:   (2, H, W)   float32
"""

import numpy as np
import cv2

try:
    from numba import njit, prange
except ImportError:  # portable fallback for environments without the JIT
    njit = None
    prange = range


if njit is not None:
    @njit(cache=True)
    def _greedy_seed_select_numba(sorted_x, sorted_y, height, width,
                                  radius, max_seeds):
        taken = np.zeros((height, width), dtype=np.uint8)
        limit = len(sorted_x) if max_seeds < 0 else min(
            len(sorted_x), max_seeds
        )
        seeds = np.empty((limit, 2), dtype=np.int32)
        seed_count = 0
        for candidate in range(len(sorted_x)):
            x = int(sorted_x[candidate])
            y = int(sorted_y[candidate])
            if taken[y, x] != 0:
                continue
            seeds[seed_count, 0] = x
            seeds[seed_count, 1] = y
            seed_count += 1
            y_start = max(0, y - radius)
            y_stop = min(height, y + radius + 1)
            x_start = max(0, x - radius)
            x_stop = min(width, x + radius + 1)
            for yy in range(y_start, y_stop):
                for xx in range(x_start, x_stop):
                    taken[yy, xx] = 1
            if max_seeds >= 0 and seed_count >= max_seeds:
                break
        return seeds[:seed_count]
else:
    _greedy_seed_select_numba = None


# --------------------------------------------------------------------------- #
#  Lane container (replaces FloatLengthLine + PointSelf)
# --------------------------------------------------------------------------- #
class Lane:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.points = []  # list of (x, y, score)
        self._score_sum = 0.0
        self.lane_id = None
        self.lane_role = None
        self.is_ego_boundary = False
        self.lateral_rank = None

    def append(self, x, y, score):
        if isinstance(self.points, np.ndarray):
            self.points = [tuple(map(float, point)) for point in self.points]
        self.points.append((float(x), float(y), float(score)))
        self._score_sum += float(score)

    def reverse(self):
        if isinstance(self.points, np.ndarray):
            self.points = self.points[::-1].copy()
        else:
            self.points.reverse()

    def __len__(self):
        return len(self.points)

    @property
    def score(self):
        if len(self.points) == 0:
            return 0.0
        return self._score_sum / len(self.points)

    def concat(self, other):
        out = Lane(self.width, self.height)
        if isinstance(self.points, np.ndarray) or isinstance(
            other.points, np.ndarray
        ):
            out.points = np.concatenate((
                np.asarray(self.points, dtype=np.float32),
                np.asarray(other.points, dtype=np.float32),
            ))
        else:
            out.points = self.points + other.points
        out._score_sum = self._score_sum + other._score_sum
        return out

    def xy(self):
        points = np.asarray(self.points, dtype=np.float32)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return points[:, :2]

    def iou(self, other, lane_width=15):
        """Rasterize both lanes (cv2.line, width 15) and return mask IoU."""
        im1 = np.zeros((self.height, self.width), np.uint8)
        im2 = np.zeros((self.height, self.width), np.uint8)
        p1 = self.xy().astype(np.int32)
        p2 = other.xy().astype(np.int32)
        for i in range(len(p1) - 1):
            cv2.line(im1, tuple(p1[i]), tuple(p1[i + 1]), 255, lane_width)
        for i in range(len(p2) - 1):
            cv2.line(im2, tuple(p2[i]), tuple(p2[i + 1]), 255, lane_width)
        union = int((cv2.bitwise_or(im1, im2) > 0).sum())
        if union == 0:
            return 0.0
        inter = int((im1 > 0).sum()) + int((im2 > 0).sum()) - union
        return inter / float(union)


# --------------------------------------------------------------------------- #
#  seeding
# --------------------------------------------------------------------------- #
def point_nms(prob, thr=0.5, min_dist=2, max_seeds=1024,
              backend="auto"):
    """Greedy point-NMS: keep highest-prob foreground pixels >= min_dist apart."""
    H, W = prob.shape
    ys, xs = np.where(prob > thr)
    if len(ys) == 0:
        return []
    order = np.argsort(-prob[ys, xs])
    selected_backend = backend
    if backend == "auto":
        selected_backend = (
            "numba" if _greedy_seed_select_numba is not None else "python"
        )
    if selected_backend == "numba":
        limit = -1 if max_seeds is None else int(max_seeds)
        return _greedy_seed_select_numba(
            np.ascontiguousarray(xs[order], dtype=np.int32),
            np.ascontiguousarray(ys[order], dtype=np.int32),
            H, W, int(min_dist), limit,
        )
    if selected_backend != "python":
        raise ValueError("point-NMS backend must be auto, numba, or python")
    taken = np.zeros((H, W), dtype=bool)
    seeds = []
    r = min_dist
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if taken[y, x]:
            continue
        seeds.append((x, y))
        taken[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
        if max_seeds is not None and len(seeds) >= max_seeds:
            break
    return seeds


# --------------------------------------------------------------------------- #
#  relay-chain crawl (port of decode_branch)
# --------------------------------------------------------------------------- #
def decode_branch(cx, cy, semantic_fine, arrow, bound, step_length, seg_threshold):
    H, W = semantic_fine.shape
    arrow_dx, arrow_dy = arrow[0], arrow[1]
    lane = Lane(W, H)
    remain_sq_sum = 0.0
    remain_count = 0
    cx, cy = int(cx), int(cy)

    for index in range(H):
        if semantic_fine[cy, cx] > seg_threshold:
            remain = bound[cy, cx] * 100 / step_length + index
            remain_sq_sum += remain * remain
            remain_count += 1

        dx = arrow_dx[cy, cx]
        dy = arrow_dy[cy, cx]
        norm = np.sqrt(dx * dx + dy * dy)
        if norm == 0:  # guard (not in original): dead transfer vector -> stop
            break
        cx = int(np.floor(cx + dx / norm * step_length))
        cy = int(np.floor(cy + dy / norm * step_length))
        if not (0 <= cx < W and 0 <= cy < H):
            break

        lane.append(cx, cy, semantic_fine[cy, cx])

        if remain_count:
            ret = np.sqrt(remain_sq_sum / remain_count)
        else:
            ret = 1
        if semantic_fine[cy, cx] > seg_threshold:
            continue
        if index > ret * 0.75:
            break
    return lane


def decode_branches_batch(seeds, semantic_fine, arrow, bound, step_length,
                          seg_threshold):
    """Vectorized equivalent of :func:`decode_branch` for many seeds.

    The relay walk is inherently sequential along each lane, but every seed at
    a given crawl step is independent. Advancing all active seeds with NumPy
    gathers removes the expensive Python ``seeds x steps`` nested loop while
    retaining the original stopping rule and integer pixel trajectory.

    Returns:
        ``(points, lengths)`` where ``points`` has shape ``(N, H, 3)`` and each
        valid prefix stores ``(x, y, score)`` exactly like ``Lane.points``.
    """
    H, W = semantic_fine.shape
    seed_array = np.asarray(seeds, dtype=np.int32)
    if seed_array.size == 0:
        return np.empty((0, H, 3), dtype=np.float32), np.zeros(0, np.int32)
    if seed_array.ndim != 2 or seed_array.shape[1] != 2:
        raise ValueError("seeds must have shape (N, 2)")

    count = len(seed_array)
    cx = seed_array[:, 0].copy()
    cy = seed_array[:, 1].copy()
    active = np.ones(count, dtype=bool)
    lengths = np.zeros(count, dtype=np.int32)
    remain_sq_sum = np.zeros(count, dtype=np.float64)
    remain_count = np.zeros(count, dtype=np.int32)
    points = np.empty((count, H, 3), dtype=np.float32)
    arrow_dx, arrow_dy = arrow[0], arrow[1]

    for index in range(H):
        active_ids = np.flatnonzero(active)
        if len(active_ids) == 0:
            break
        current_x = cx[active_ids]
        current_y = cy[active_ids]
        current_score = semantic_fine[current_y, current_x]
        foreground = current_score > seg_threshold
        if np.any(foreground):
            foreground_ids = active_ids[foreground]
            remain = (
                bound[current_y[foreground], current_x[foreground]]
                * 100.0 / step_length + index
            )
            remain_sq_sum[foreground_ids] += remain * remain
            remain_count[foreground_ids] += 1

        dx = arrow_dx[current_y, current_x]
        dy = arrow_dy[current_y, current_x]
        norm = np.sqrt(dx * dx + dy * dy)
        movable = np.isfinite(norm) & (norm != 0.0)
        if np.any(~movable):
            active[active_ids[~movable]] = False
        active_ids = active_ids[movable]
        if len(active_ids) == 0:
            continue
        next_x = np.floor(
            cx[active_ids] + dx[movable] / norm[movable] * step_length
        ).astype(np.int32)
        next_y = np.floor(
            cy[active_ids] + dy[movable] / norm[movable] * step_length
        ).astype(np.int32)
        in_bounds = (
            (next_x >= 0) & (next_x < W) & (next_y >= 0) & (next_y < H)
        )
        if np.any(~in_bounds):
            active[active_ids[~in_bounds]] = False
        active_ids = active_ids[in_bounds]
        if len(active_ids) == 0:
            continue
        next_x = next_x[in_bounds]
        next_y = next_y[in_bounds]
        cx[active_ids] = next_x
        cy[active_ids] = next_y
        next_score = semantic_fine[next_y, next_x]
        points[active_ids, index, 0] = next_x
        points[active_ids, index, 1] = next_y
        points[active_ids, index, 2] = next_score
        lengths[active_ids] = index + 1

        has_remaining = remain_count[active_ids] > 0
        remaining = np.ones(len(active_ids), dtype=np.float64)
        remaining[has_remaining] = np.sqrt(
            remain_sq_sum[active_ids[has_remaining]]
            / remain_count[active_ids[has_remaining]]
        )
        stop = (
            (next_score <= seg_threshold)
            & (index > remaining * 0.75)
        )
        if np.any(stop):
            active[active_ids[stop]] = False
    return points, lengths


if njit is not None:
    @njit(cache=True, parallel=True)
    def _decode_branches_numba_impl(seeds, semantic_fine, arrow, bound,
                                    step_length, seg_threshold):
        """Parallel scalar relay walks compiled to native CPU code."""
        height, width = semantic_fine.shape
        seed_count = len(seeds)
        points = np.empty((seed_count, height, 3), dtype=np.float32)
        lengths = np.zeros(seed_count, dtype=np.int32)
        for seed_index in prange(seed_count):
            cx = int(seeds[seed_index, 0])
            cy = int(seeds[seed_index, 1])
            remain_sq_sum = 0.0
            remain_count = 0
            for index in range(height):
                if semantic_fine[cy, cx] > seg_threshold:
                    remain = (
                        bound[cy, cx] * 100.0 / step_length + index
                    )
                    remain_sq_sum += remain * remain
                    remain_count += 1

                dx = arrow[0, cy, cx]
                dy = arrow[1, cy, cx]
                norm = np.sqrt(dx * dx + dy * dy)
                if norm == 0.0 or not np.isfinite(norm):
                    break
                cx = int(np.floor(cx + dx / norm * step_length))
                cy = int(np.floor(cy + dy / norm * step_length))
                if not (0 <= cx < width and 0 <= cy < height):
                    break

                score = semantic_fine[cy, cx]
                points[seed_index, index, 0] = cx
                points[seed_index, index, 1] = cy
                points[seed_index, index, 2] = score
                lengths[seed_index] = index + 1

                ret = (
                    np.sqrt(remain_sq_sum / remain_count)
                    if remain_count else 1.0
                )
                if score > seg_threshold:
                    continue
                if index > ret * 0.75:
                    break
        return points, lengths


    @njit(cache=True, parallel=True)
    def _candidate_metadata_numba_impl(up_points, up_lengths,
                                       down_points, down_lengths, bin_px):
        seed_count = len(up_lengths)
        scores = np.zeros(seed_count, dtype=np.float64)
        bins = np.zeros(seed_count, dtype=np.int32)
        total_lengths = up_lengths + down_lengths
        for seed_index in prange(seed_count):
            up_length = int(up_lengths[seed_index])
            down_length = int(down_lengths[seed_index])
            total_length = up_length + down_length
            if total_length <= 1:
                continue
            y_values = np.empty(total_length, dtype=np.float32)
            score_sum = 0.0
            position = 0
            for point_index in range(up_length):
                score_sum += up_points[seed_index, point_index, 2]
                y_values[position] = up_points[seed_index, point_index, 1]
                position += 1
            for point_index in range(down_length):
                score_sum += down_points[seed_index, point_index, 2]
                y_values[position] = down_points[seed_index, point_index, 1]
                position += 1
            scores[seed_index] = score_sum / total_length
            median_y = np.median(y_values)
            lower_x = np.empty(total_length, dtype=np.float32)
            lower_count = 0
            for point_index in range(up_length):
                if up_points[seed_index, point_index, 1] >= median_y:
                    lower_x[lower_count] = up_points[
                        seed_index, point_index, 0
                    ]
                    lower_count += 1
            for point_index in range(down_length):
                if down_points[seed_index, point_index, 1] >= median_y:
                    lower_x[lower_count] = down_points[
                        seed_index, point_index, 0
                    ]
                    lower_count += 1
            bins[seed_index] = int(
                np.median(lower_x[:lower_count]) // bin_px
            )
        return total_lengths, scores, bins
else:
    _decode_branches_numba_impl = None
    _candidate_metadata_numba_impl = None


def decode_branches_numba(seeds, semantic_fine, arrow, bound, step_length,
                          seg_threshold):
    if _decode_branches_numba_impl is None:
        raise RuntimeError(
            "Numba crawl requested but numba is not installed; "
            "install requirements.txt or use crawl_backend='numpy'"
        )
    seed_array = np.asarray(seeds, dtype=np.int32)
    if seed_array.size == 0:
        height = semantic_fine.shape[0]
        return (
            np.empty((0, height, 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )
    return _decode_branches_numba_impl(
        seed_array,
        np.ascontiguousarray(semantic_fine, dtype=np.float32),
        np.ascontiguousarray(arrow, dtype=np.float32),
        np.ascontiguousarray(bound, dtype=np.float32),
        float(step_length),
        float(seg_threshold),
    )


def warmup_decode_backend(crawl_backend="auto"):
    """Compile the optional Numba backend before latency measurements."""
    backend = crawl_backend
    if crawl_backend == "auto":
        backend = "numba" if njit is not None else "numpy"
    if backend != "numba":
        return backend
    semantic = np.zeros((8, 8), dtype=np.float32)
    arrow = np.zeros((2, 8, 8), dtype=np.float32)
    arrow[1] = 1.0
    bound = np.zeros((8, 8), dtype=np.float32)
    points, lengths = decode_branches_numba(
        [(4, 4)], semantic, arrow, bound, 1, 0.5
    )
    _candidate_metadata_numba_impl(
        points, lengths, points, lengths, 16
    )
    _greedy_seed_select_numba(
        np.array((4,), dtype=np.int32),
        np.array((4,), dtype=np.int32),
        8, 8, 2, 1,
    )
    return backend


def configure_decode_threads(thread_count=8):
    """Set Numba's relay-crawl worker count and return the applied value."""
    if njit is None:
        return 1
    import numba
    maximum = int(numba.config.NUMBA_NUM_THREADS)
    if not 1 <= int(thread_count) <= maximum:
        raise ValueError(f"decode threads must be in [1, {maximum}]")
    numba.set_num_threads(int(thread_count))
    return numba.get_num_threads()


def _candidate_metadata_numpy(up_points, up_lengths,
                              down_points, down_lengths, bin_px):
    total_lengths = up_lengths + down_lengths
    scores = np.zeros(len(total_lengths), dtype=np.float64)
    bins = np.zeros(len(total_lengths), dtype=np.int32)
    for seed_index, total_length in enumerate(total_lengths):
        if total_length <= 1:
            continue
        up = up_points[seed_index, :up_lengths[seed_index]]
        down = down_points[seed_index, :down_lengths[seed_index]]
        merged = np.concatenate((up, down))
        scores[seed_index] = merged[:, 2].sum(dtype=np.float64) / len(merged)
        median_y = np.median(merged[:, 1])
        lower = merged[merged[:, 1] >= median_y]
        bins[seed_index] = int(np.median(lower[:, 0]) // bin_px)
    return total_lengths, scores, bins


def _preselect_batched_candidates(up_points, up_lengths,
                                  down_points, down_lengths, score_threshold,
                                  max_lanes, backend, bin_px=16):
    metadata = (
        _candidate_metadata_numba_impl
        if backend == "numba" else _candidate_metadata_numpy
    )
    total_lengths, scores, bins = metadata(
        up_points, up_lengths, down_points, down_lengths, bin_px
    )
    valid = np.flatnonzero(
        (total_lengths > 1) & (scores >= score_threshold)
    )
    order = valid[np.argsort(-scores[valid], kind="stable")]
    if max_lanes is None or len(order) <= max_lanes:
        return order

    buckets = {}
    for candidate in order:
        buckets.setdefault(int(bins[candidate]), []).append(int(candidate))
    keys = list(buckets)
    positions = {key: 0 for key in keys}
    selected = []
    while len(selected) < max_lanes:
        progressed = False
        for key in keys:
            position = positions[key]
            if position < len(buckets[key]):
                selected.append(buckets[key][position])
                positions[key] += 1
                progressed = True
                if len(selected) >= max_lanes:
                    break
        if not progressed:
            break
    return np.asarray(
        sorted(selected, key=lambda index: scores[index], reverse=True),
        dtype=np.int32,
    )


def _lines_from_batched_crawls(up_points, up_lengths,
                               down_points, down_lengths, width, height,
                               candidate_indices=None):
    lines = []
    if candidate_indices is None:
        candidate_indices = range(len(up_lengths))
    for seed_index in candidate_indices:
        up_length = int(up_lengths[seed_index])
        down_length = int(down_lengths[seed_index])
        if up_length + down_length <= 1:
            continue
        merged = np.concatenate((
            up_points[seed_index, :up_length][::-1],
            down_points[seed_index, :down_length],
        ))
        lane = Lane(width, height)
        lane.points = merged
        lane._score_sum = float(merged[:, 2].sum(dtype=np.float64))
        lines.append(lane)
    return lines


# --------------------------------------------------------------------------- #
#  post-processing
# --------------------------------------------------------------------------- #
def thresh_line(lines, thr=0.10):
    return [ln for ln in lines if ln.score >= thr]


def _diverse_preselect(order, max_lanes, bin_px=16):
    """Cap the candidate list while keeping spatial diversity.

    `order` is already sorted by score (desc). Taking the top `max_lanes`
    outright lets the single strongest lane monopolize every slot -- its copies
    all score highest -- so genuine but slightly weaker lanes get dropped before
    IoU-NMS ever compares them (this collapsed multi-lane curves to one lane).
    Instead, bucket candidates by their horizontal position on the lower half of
    the line (where lanes are well separated) and pick round-robin across
    buckets, so every lane keeps representatives within the cap.
    """
    if max_lanes is None:
        return order
    if max_lanes <= 0:
        raise ValueError("max_lanes must be positive or None")
    if bin_px <= 0:
        raise ValueError("bin_px must be positive")
    if len(order) <= max_lanes:
        return order
    buckets = {}
    for ln in order:                      # already score-sorted
        xy = ln.xy()
        if len(xy) == 0:
            continue
        low = xy[xy[:, 1] >= np.median(xy[:, 1])]      # lower (near) half
        ref = low if len(low) else xy
        key = int(np.median(ref[:, 0]) // bin_px)
        buckets.setdefault(key, []).append(ln)
    keys = list(buckets.keys())
    idx = {k: 0 for k in keys}
    selected = []
    while len(selected) < max_lanes:
        progressed = False
        for k in keys:
            if idx[k] < len(buckets[k]):
                selected.append(buckets[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(selected) >= max_lanes:
                    break
        if not progressed:
            break
    # NMS is greedy, so restore global score priority after choosing a spatially
    # diverse candidate set.
    return sorted(selected, key=lambda ln: ln.score, reverse=True)


def iou_nms(lines, thr=0.5, max_lanes=128, scale=0.25,
            lane_width=15):
    """Lane IoU NMS with one cached, downscaled mask per candidate.

    Rasterizes each candidate once (downscaled) instead of re-rasterizing both
    masks per pair. The candidate list is capped with `_diverse_preselect` rather
    than a plain top-score cut, so the strongest lane cannot crowd out the others
    before NMS runs.
    """
    if not lines:
        return []
    order = sorted(lines, key=lambda ln: ln.score, reverse=True)
    order = _diverse_preselect(order, max_lanes)
    height = max(1, int(round(order[0].height * scale)))
    width = max(1, int(round(order[0].width * scale)))
    scaled_width = max(1, int(round(lane_width * scale)))
    masks = []
    areas = []
    for line in order:
        mask = np.zeros((height, width), np.uint8)
        points = line.xy() * scale
        if len(points) >= 2:
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.polylines(mask, [points.astype(np.int32)], False, 1,
                          scaled_width)
        masks.append(mask)
        areas.append(int(np.count_nonzero(mask)))

    suppressed = [False] * len(order)
    keep = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        for j in range(i + 1, len(order)):
            if suppressed[j]:
                continue
            inter = int(np.count_nonzero(masks[i] & masks[j]))
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union >= thr:
                suppressed[j] = True
    return keep


# --------------------------------------------------------------------------- #
#  lane identity (left-to-right ordering)
# --------------------------------------------------------------------------- #
def _bottom_x(lane):
    """x where a lane meets its nearest row (largest y). Lanes fan out near the
    camera, so this is the most reliable place to order them left-to-right."""
    xy = lane.xy()
    if len(xy) == 0:
        return float("inf")
    return float(xy[int(np.argmax(xy[:, 1])), 0])


def _ego_reference_x(lane, target_y=None):
    """Estimate where a boundary meets the near-camera reference row.

    On a sharp bend, multiple boundaries can leave through the same image side.
    Comparing their last visible x then becomes ambiguous (both are about 0 or
    ``width - 1``), and a short outer boundary can be mistaken for the ego-lane
    boundary. Extrapolating the lower 40% of each polyline to a common row keeps
    their lateral order after they leave the image.
    """
    xy = lane.xy()
    if len(xy) < 2:
        return _bottom_x(lane)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return _bottom_x(lane)
    if target_y is None:
        target_y = lane.height - 1.0

    cutoff = np.quantile(xy[:, 1], 0.6)
    lower = xy[xy[:, 1] >= cutoff]
    if len(lower) < 2 or np.ptp(lower[:, 1]) < 1.0:
        return _bottom_x(lane)

    ys = lower[:, 1]
    xs = lower[:, 0]
    centered_y = ys - ys.mean()
    denominator = float(np.dot(centered_y, centered_y))
    if denominator <= 1e-6:
        return _bottom_x(lane)
    slope = float(np.dot(centered_y, xs - xs.mean()) / denominator)
    return float(xs.mean() + slope * (float(target_y) - ys.mean()))


def order_lanes(lanes):
    """Sort lanes left-to-right and assign a frame-local index.

    RCLane is anchor-free: `decode` emits lane instances in score order with no
    inherent identity. This helper only establishes spatial order inside one
    frame; it must not be used as a persistent video identity because a missing
    outer lane would shift every following index.
    """
    ordered = sorted(lanes, key=_bottom_x)
    for i, ln in enumerate(ordered):
        ln.lane_id = i
    return ordered


def assign_ego_lane_roles(lanes, ego_x=None):
    """Assign stable semantic IDs relative to the ego vehicle.

    IDs describe a lane boundary's role rather than its position in a variable
    length list:

      * P1: nearest boundary left of ego (current-lane left boundary)
      * P2: nearest boundary right of ego (current-lane right boundary)
      * P0: next boundary to the left
      * P3: next boundary to the right

    Consequently P1/P2 do not become P0/P1 merely because an outer boundary is
    temporarily missing. With the default four-lane cap, returned IDs are in
    ``[0, 3]``. More uncapped lanes continue outward with negative IDs on the
    left and IDs greater than three on the right.
    """
    if not lanes:
        return []
    if ego_x is None:
        ego_x = lanes[0].width / 2.0
    ego_x = float(ego_x)

    for lane in lanes:
        lane.lane_id = None
        lane.lane_role = None
        lane.is_ego_boundary = False
        lane.lateral_rank = None

    reference_x = {lane: _ego_reference_x(lane) for lane in lanes}
    left = sorted(
        (lane for lane in lanes if reference_x[lane] < ego_x),
        key=lambda lane: (abs(reference_x[lane] - ego_x), -lane.score),
    )
    right = sorted(
        (lane for lane in lanes if reference_x[lane] >= ego_x),
        key=lambda lane: (abs(reference_x[lane] - ego_x), -lane.score),
    )

    for rank, lane in enumerate(left, 1):
        lane.lane_id = 2 - rank  # nearest left=P1, next=P0
        lane.lateral_rank = -rank
        lane.is_ego_boundary = rank == 1
        lane.lane_role = "ego_left" if rank == 1 else f"left_{rank}"
    for rank, lane in enumerate(right, 1):
        lane.lane_id = 1 + rank  # nearest right=P2, next=P3
        lane.lateral_rank = rank
        lane.is_ego_boundary = rank == 1
        lane.lane_role = "ego_right" if rank == 1 else f"right_{rank}"

    return sorted(lanes, key=lambda lane: lane.lane_id)


def ego_lane_boundaries(lanes):
    """Return ``(left, right)`` boundaries of the lane containing ego.

    Either value can be ``None`` when that side was not detected.
    """
    left = next(
        (lane for lane in lanes if lane.lane_role == "ego_left"), None
    )
    right = next(
        (lane for lane in lanes if lane.lane_role == "ego_right"), None
    )
    return left, right


def select_ego_lanes(lanes, max_lanes=4, ego_x=None,
                      min_score_ratio=0.5, balance_sides=True):
    """Keep the closest reliable lane boundaries around the ego vehicle.

    The decoder can occasionally return an extra low-confidence crawl in
    addition to the real road boundaries.  When more than ``max_lanes`` are
    present, first prefer candidates whose score is at least
    ``min_score_ratio`` of the best candidate (provided that still leaves enough
    lanes), then select the nearest boundaries using their near-camera x.

    For the usual four-lane output, ``balance_sides`` reserves two slots on
    either side of the camera centre when possible.  Any unfilled slots are
    taken from the remaining closest candidates. The returned IDs are semantic:
    P1/P2 are the current-lane boundaries, while P0/P3 are the adjacent outer
    boundaries. Missing outer lanes therefore do not shift the ego-lane IDs.
    """
    if max_lanes is None:
        return assign_ego_lane_roles(lanes, ego_x)
    if max_lanes <= 0:
        raise ValueError("max_lanes must be positive or None")
    if not 0.0 <= min_score_ratio <= 1.0:
        raise ValueError("min_score_ratio must be in [0, 1]")

    ordered = order_lanes(lanes)
    if not ordered:
        return []
    if ego_x is None:
        ego_x = ordered[0].width / 2.0
    ego_x = float(ego_x)

    # The four-lane semantic contract has exactly two possible boundaries per
    # side: P0/P1 on the left and P2/P3 on the right. If one side is missing,
    # return fewer lanes instead of filling the gap with P4/P-1 farther out.
    if len(ordered) <= max_lanes:
        if balance_sides and max_lanes == 4:
            reference_x = {lane: _ego_reference_x(lane) for lane in ordered}

            def near_ego(lane):
                return (abs(reference_x[lane] - ego_x), -lane.score)

            left = sorted(
                (lane for lane in ordered if reference_x[lane] < ego_x),
                key=near_ego,
            )
            right = sorted(
                (lane for lane in ordered if reference_x[lane] >= ego_x),
                key=near_ego,
            )
            ordered = left[:2] + right[:2]
        return assign_ego_lane_roles(ordered, ego_x)

    best_score = max(lane.score for lane in ordered)
    reliable = [
        lane for lane in ordered
        if lane.score >= best_score * min_score_ratio
    ]
    # Never let the reliability gate force the output below the requested cap.
    pool = reliable if len(reliable) >= max_lanes else ordered

    reference_x = {lane: _ego_reference_x(lane) for lane in pool}

    def proximity_key(lane):
        return (abs(reference_x[lane] - ego_x), -lane.score)

    ranked = sorted(pool, key=proximity_key)
    selected = []
    if balance_sides and max_lanes >= 2:
        left = sorted(
            (lane for lane in pool if reference_x[lane] < ego_x),
            key=proximity_key,
        )
        right = sorted(
            (lane for lane in pool if reference_x[lane] >= ego_x),
            key=proximity_key,
        )
        left_slots = max_lanes // 2
        right_slots = max_lanes - left_slots
        selected.extend(left[:left_slots])
        selected.extend(right[:right_slots])

    if not (balance_sides and max_lanes == 4):
        for lane in ranked:
            if lane not in selected:
                selected.append(lane)
            if len(selected) == max_lanes:
                break

    return assign_ego_lane_roles(selected[:max_lanes], ego_x)


# --------------------------------------------------------------------------- #
#  full decode
# --------------------------------------------------------------------------- #
def decode(seg_prob, up_arrow, down_arrow, up_bound, down_bound,
           step_length=10, seg_threshold=0.5, seed_min_dist=2,
           score_thresh=0.10, iou_thresh=0.5, seed_threshold=None,
           max_seeds=1024, nms_max_lanes=128, nms_scale=0.25,
           sort_lanes=True, max_output_lanes=4, ego_x=None,
           ego_min_score_ratio=0.5, balance_ego_sides=True,
           batch_crawl=True, crawl_backend="auto",
           point_nms_backend="auto"):
    """
    Args:
        seg_prob: (H, W) foreground probability.
        up_arrow, down_arrow, up_bound, down_bound: (2, H, W).
        seg_threshold: foreground test used while crawling a chain.
        seed_threshold: threshold for picking seeds (defaults to seg_threshold).
            RCLane seg maps are low-magnitude (OHEM 15:1), so seeds often sit below
            0.5 -- set this lower (e.g. 0.3) for under-trained models.
        max_output_lanes: final ego-centric lane cap. Defaults to four; pass
            ``None`` to preserve every lane surviving NMS.
    Returns:
        list of Lane. Use `lane.xy()` for the (N, 2) point array and `lane.score`.
    """
    H, W = seg_prob.shape
    if seed_threshold is None:
        seed_threshold = seg_threshold
    seeds = point_nms(
        seg_prob, seed_threshold, seed_min_dist, max_seeds,
        backend=point_nms_backend,
    )
    ub0, db0 = up_bound[0], down_bound[0]  # bound channel 0 (both channels equal)

    if batch_crawl:
        backend = crawl_backend
        if crawl_backend == "auto":
            backend = "numba" if njit is not None else "numpy"
        if backend not in ("numba", "numpy"):
            raise ValueError("crawl_backend must be auto, numba, or numpy")
        crawl = (
            decode_branches_numba if backend == "numba"
            else decode_branches_batch
        )
        up_points, up_lengths = crawl(
            seeds, seg_prob, up_arrow, ub0, step_length, seg_threshold
        )
        down_points, down_lengths = crawl(
            seeds, seg_prob, down_arrow, db0, step_length, seg_threshold
        )
        candidate_indices = _preselect_batched_candidates(
            up_points, up_lengths, down_points, down_lengths,
            score_thresh, nms_max_lanes, backend,
        )
        lines = _lines_from_batched_crawls(
            up_points, up_lengths, down_points, down_lengths, W, H,
            candidate_indices,
        )
    else:
        lines = []
        for (x, y) in seeds:
            up = decode_branch(
                x, y, seg_prob, up_arrow, ub0, step_length, seg_threshold
            )
            down = decode_branch(
                x, y, seg_prob, down_arrow, db0, step_length, seg_threshold
            )
            up.reverse()
            full = up.concat(down)
            if len(full) > 1:
                lines.append(full)

    lines = thresh_line(lines, score_thresh)
    lines = iou_nms(lines, iou_thresh, max_lanes=nms_max_lanes,
                    scale=nms_scale)
    if max_output_lanes is not None:
        lines = select_ego_lanes(
            lines,
            max_lanes=max_output_lanes,
            ego_x=ego_x,
            min_score_ratio=ego_min_score_ratio,
            balance_sides=balance_ego_sides,
        )
    elif sort_lanes:
        lines = order_lanes(lines)
    return lines


def decode_predictions(pred_dict, **kwargs):
    """Convenience: decode a batch of network outputs (torch tensors, B,2,H,W).

    Returns a list (len B) of lists of Lane. Requires torch only for the input.
    """
    import torch  # local import so the module stays torch-free otherwise

    seg = torch.softmax(pred_dict["seg_map"], dim=1)[:, 1]  # (B,H,W) fg prob
    seg = seg.detach().cpu().numpy()
    ua = pred_dict["up_arrow"].detach().cpu().numpy()
    da = pred_dict["down_arrow"].detach().cpu().numpy()
    ub = pred_dict["up_bound"].detach().cpu().numpy()
    db = pred_dict["down_bound"].detach().cpu().numpy()
    return [decode(seg[b], ua[b], da[b], ub[b], db[b], **kwargs) for b in range(seg.shape[0])]


# --------------------------------------------------------------------------- #
#  smoke test -- encode/decode ROUND TRIP
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from encode import encode, IMG_SIZE

    H, W = IMG_SIZE

    # ground-truth lane (an S-curve), points in 800x320 space
    ys = np.linspace(20, 300, 40)
    xs = 400 + 150 * np.sin(ys / 300 * np.pi)
    gt_lane = np.stack([xs, ys], axis=1)

    gt = encode([list(map(tuple, gt_lane))])

    # decode straight from the GT maps -> should reconstruct the lane.
    # use a coarser seed spacing to keep the test fast.
    lanes = decode(
        gt["seg_map"], gt["up_arrow"], gt["down_arrow"], gt["up_bound"], gt["down_bound"],
        seed_min_dist=12,
    )
    print(f"decoded {len(lanes)} lane(s) from GT maps")
    assert len(lanes) >= 1, "round trip produced no lanes!"

    # take the longest recovered lane, measure how far its points sit from the GT curve
    best = max(lanes, key=len)
    pred_xy = best.xy()
    print(f"longest lane: {len(best)} points, score={best.score:.2f}")

    def dist_to_gt(pt):
        d = np.hypot(gt_lane[:, 0] - pt[0], gt_lane[:, 1] - pt[1])
        return d.min()

    errs = np.array([dist_to_gt(p) for p in pred_xy])
    print(f"mean dist to GT curve = {errs.mean():.2f}px, max = {errs.max():.2f}px")
    assert errs.mean() < 8.0, "reconstructed lane strays too far from GT!"

    print("OK -- encode/decode round trip reconstructs the lane.")

    # Regression: a high-scoring lane may have hundreds of near-duplicate
    # crawls. The cap must still retain weaker candidates from other lanes,
    # while greedy NMS must receive candidates in descending score order.
    def vertical_lane(x, score):
        lane = Lane(W, H)
        lane.append(x, 200, score)
        lane.append(x, 300, score)
        return lane

    candidates = [
        vertical_lane(100 + index % 2, 0.99 - index * 0.001)
        for index in range(24)
    ]
    candidates += [vertical_lane(350, 0.90), vertical_lane(650, 0.89)]
    candidates.sort(key=lambda lane: lane.score, reverse=True)
    selected = _diverse_preselect(candidates, max_lanes=8, bin_px=16)
    selected_bins = {
        int(np.median(lane.xy()[:, 0]) // 16) for lane in selected
    }
    expected_bins = {100 // 16, 350 // 16, 650 // 16}
    assert expected_bins <= selected_bins, "spatial preselection dropped a lane"
    selected_scores = [lane.score for lane in selected]
    assert selected_scores == sorted(selected_scores, reverse=True), (
        "spatial preselection changed greedy NMS score priority"
    )
    print("OK -- diverse NMS preselection retains spatially distinct lanes.")

    # Regression: cap the final output around the ego vehicle without keeping a
    # weak extra crawl merely because its endpoint is slightly closer laterally.
    ego_candidates = [
        vertical_lane(8, 0.93),
        vertical_lane(20, 0.24),  # spurious fifth crawl
        vertical_lane(95, 0.94),
        vertical_lane(748, 0.89),
        vertical_lane(796, 0.81),
    ]
    ego_lanes = select_ego_lanes(ego_candidates, max_lanes=4)
    ego_xs = [int(_bottom_x(lane)) for lane in ego_lanes]
    assert ego_xs == [8, 95, 748, 796], (
        f"ego selector kept the wrong lanes: {ego_xs}"
    )
    assert [lane.lane_id for lane in ego_lanes] == [0, 1, 2, 3]
    ego_left, ego_right = ego_lane_boundaries(ego_lanes)
    assert ego_left is not None and int(_bottom_x(ego_left)) == 95
    assert ego_right is not None and int(_bottom_x(ego_right)) == 748

    # Semantic IDs must not shift when an outer lane disappears. P1/P2 remain
    # the current-lane boundaries and P3 remains the next boundary on the right.
    missing_outer_left = select_ego_lanes(
        [vertical_lane(95, 0.94), vertical_lane(748, 0.89),
         vertical_lane(796, 0.81)],
        max_lanes=4,
    )
    assert [lane.lane_id for lane in missing_outer_left] == [1, 2, 3]
    missing_outer_right = select_ego_lanes(
        [vertical_lane(8, 0.93), vertical_lane(95, 0.94),
         vertical_lane(748, 0.89)],
        max_lanes=4,
    )
    assert [lane.lane_id for lane in missing_outer_right] == [0, 1, 2]

    right_only = select_ego_lanes(
        [vertical_lane(500, 0.95), vertical_lane(600, 0.90),
         vertical_lane(700, 0.80)],
        max_lanes=4,
    )
    assert [lane.lane_id for lane in right_only] == [2, 3]

    # Two right boundaries can both leave through x=width on a sharp curve.
    # The longer/nearer curve must remain P2 even if its last visible x is a
    # little farther right than the short outer curve's last x.
    near_right = Lane(800, 320)
    outer_right = Lane(800, 320)
    for x, y in ((700, 200), (740, 225), (780, 250)):
        near_right.append(x, y, 0.9)
    for x, y in ((700, 125), (750, 137.5), (790, 147.5)):
        outer_right.append(x, y, 0.8)
    curved_right = assign_ego_lane_roles([outer_right, near_right], ego_x=400)
    assert near_right.lane_id == 2 and near_right.lane_role == "ego_right"
    assert outer_right.lane_id == 3 and outer_right.lane_role == "right_2"
    assert [lane.lane_id for lane in curved_right] == [2, 3]
    print("OK -- ego post-processing keeps four reliable nearby lanes.")
