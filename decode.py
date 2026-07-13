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


# --------------------------------------------------------------------------- #
#  Lane container (replaces FloatLengthLine + PointSelf)
# --------------------------------------------------------------------------- #
class Lane:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.points = []  # list of (x, y, score)
        self._score_sum = 0.0

    def append(self, x, y, score):
        self.points.append((float(x), float(y), float(score)))
        self._score_sum += float(score)

    def reverse(self):
        self.points.reverse()

    def __len__(self):
        return len(self.points)

    @property
    def score(self):
        if not self.points:
            return 0.0
        return self._score_sum / len(self.points)

    def concat(self, other):
        out = Lane(self.width, self.height)
        out.points = self.points + other.points
        out._score_sum = self._score_sum + other._score_sum
        return out

    def xy(self):
        return np.array([(p[0], p[1]) for p in self.points], dtype=float)

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
def point_nms(prob, thr=0.5, min_dist=2, max_seeds=1024):
    """Greedy point-NMS: keep highest-prob foreground pixels >= min_dist apart."""
    H, W = prob.shape
    ys, xs = np.where(prob > thr)
    if len(ys) == 0:
        return []
    order = np.argsort(-prob[ys, xs])
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


# --------------------------------------------------------------------------- #
#  post-processing
# --------------------------------------------------------------------------- #
def thresh_line(lines, thr=0.10):
    return [ln for ln in lines if ln.score >= thr]


def iou_nms(lines, thr=0.5, max_lanes=128, scale=0.25,
            lane_width=15):
    """Lane IoU NMS with one cached, downscaled mask per candidate.

    The previous implementation re-rasterized two full 800x320 masks for every
    pair, making untrained-model eval effectively quadratic in both lane count
    and pixel count. Real road scenes contain far fewer than 128 lanes, so keep
    the highest-scoring candidates and rasterize each only once.
    """
    if not lines:
        return []
    order = sorted(lines, key=lambda ln: ln.score, reverse=True)
    if max_lanes is not None:
        order = order[:max_lanes]
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
#  full decode
# --------------------------------------------------------------------------- #
def decode(seg_prob, up_arrow, down_arrow, up_bound, down_bound,
           step_length=10, seg_threshold=0.5, seed_min_dist=2,
           score_thresh=0.10, iou_thresh=0.5, seed_threshold=None,
           max_seeds=1024, nms_max_lanes=128, nms_scale=0.25):
    """
    Args:
        seg_prob: (H, W) foreground probability.
        up_arrow, down_arrow, up_bound, down_bound: (2, H, W).
        seg_threshold: foreground test used while crawling a chain.
        seed_threshold: threshold for picking seeds (defaults to seg_threshold).
            RCLane seg maps are low-magnitude (OHEM 15:1), so seeds often sit below
            0.5 -- set this lower (e.g. 0.3) for under-trained models.
    Returns:
        list of Lane. Use `lane.xy()` for the (N, 2) point array and `lane.score`.
    """
    H, W = seg_prob.shape
    if seed_threshold is None:
        seed_threshold = seg_threshold
    seeds = point_nms(seg_prob, seed_threshold, seed_min_dist, max_seeds)
    ub0, db0 = up_bound[0], down_bound[0]  # bound channel 0 (both channels equal)

    lines = []
    for (x, y) in seeds:
        up = decode_branch(x, y, seg_prob, up_arrow, ub0, step_length, seg_threshold)
        down = decode_branch(x, y, seg_prob, down_arrow, db0, step_length, seg_threshold)
        up.reverse()
        full = up.concat(down)
        if len(full) > 1:
            lines.append(full)

    lines = thresh_line(lines, score_thresh)
    lines = iou_nms(lines, iou_thresh, max_lanes=nms_max_lanes,
                    scale=nms_scale)
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
