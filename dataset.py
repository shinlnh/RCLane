"""
Shared dataset machinery for RCLane.

`LaneEncodeDataset` holds everything common to any annotation format: image resize +
normalization, running `encode` to build the 5 GT maps, and caching that GT to disk in
SPARSE form (foreground pixels only) so we don't recompute the slow shapely encode every
epoch, nor store hundreds of GB of dense maps.

A concrete dataset only implements `_load(idx)`, returning the raw image and lane
polylines in the original image coordinate space. See `dataset_curvelanes.py`.
"""

import os
import hashlib

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from encode import encode

# ImageNet normalization (lets us plug in pretrained MiT later)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def normalize_image(img_bgr, W, H):
    """BGR uint8 -> resized, ImageNet-normalized (3, H, W) float tensor."""
    img_r = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    x = img_r[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
    x = (x - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def sparse_from_dense(gt):
    """Keep only foreground pixels of the dense GT maps."""
    ys, xs = np.where(gt["seg_map"] > 0.5)
    return dict(
        ys=ys.astype(np.int32),
        xs=xs.astype(np.int32),
        up_arrow=gt["up_arrow"][:, ys, xs].T.astype(np.float32),   # (N,2)
        down_arrow=gt["down_arrow"][:, ys, xs].T.astype(np.float32),
        up_bound=gt["up_bound"][0, ys, xs].astype(np.float32),     # (N,)
        down_bound=gt["down_bound"][0, ys, xs].astype(np.float32),
    )


def dense_from_sparse(s, H, W):
    """Scatter sparse foreground values back to dense (2,H,W) maps."""
    seg = np.zeros((H, W), np.float32)
    ua = np.zeros((2, H, W), np.float32)
    da = np.zeros((2, H, W), np.float32)
    ub = np.zeros((2, H, W), np.float32)
    db = np.zeros((2, H, W), np.float32)
    ys, xs = s["ys"], s["xs"]
    if len(ys) > 0:
        seg[ys, xs] = 1.0
        ua[:, ys, xs] = s["up_arrow"].T
        da[:, ys, xs] = s["down_arrow"].T
        ub[:, ys, xs] = s["up_bound"]
        db[:, ys, xs] = s["down_bound"]
    return dict(seg_map=seg, up_arrow=ua, down_arrow=da, up_bound=ub, down_bound=db)


class LaneEncodeDataset(Dataset):
    """Base class: resize + normalize + encode(+cache). Subclass provides `_load`."""

    def __init__(self, img_size=(320, 800), cache_dir=None):
        self.H, self.W = img_size
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ---- to be implemented by subclasses ----
    def _load(self, idx):
        """Return (img_bgr, lanes_orig, ow, oh, cache_key).

        lanes_orig: list of lanes, each a list of (x, y) in the ORIGINAL image space.
        cache_key : a unique string per sample (usually the image path).
        """
        raise NotImplementedError

    # ---- shared machinery ----
    def _scale(self, lanes, ow, oh):
        sx, sy = self.W / ow, self.H / oh
        return [[(x * sx, y * sy) for x, y in lane] for lane in lanes]

    def _cache_path(self, key):
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, h + ".npz")

    def _get_gt(self, key, lanes_orig, ow, oh):
        if self.cache_dir:
            cp = self._cache_path(key)
            if os.path.exists(cp):
                try:
                    with np.load(cp) as sparse:
                        return dense_from_sparse(sparse, self.H, self.W)
                except (OSError, ValueError, EOFError):
                    # A job killed during an older, non-atomic cache write may
                    # leave a truncated npz. Rebuild it instead of killing a
                    # long multi-GPU run.
                    pass
        gt = encode(self._scale(lanes_orig, ow, oh), img_size=(self.H, self.W))
        if self.cache_dir:
            # Many DataLoader/DDP processes can discover the same missing key
            # at once (DistributedSampler may pad one sample). Write privately
            # and atomically publish the completed archive so readers never see
            # a half-written npz.
            tmp = f"{cp}.{os.getpid()}.tmp.npz"
            try:
                np.savez(tmp, **sparse_from_dense(gt))
                os.replace(tmp, cp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        return gt

    def __getitem__(self, idx):
        img_bgr, lanes_orig, ow, oh, key = self._load(idx)
        gt = self._get_gt(key, lanes_orig, ow, oh)
        x = normalize_image(img_bgr, self.W, self.H)
        target = {
            "seg_map": torch.from_numpy(gt["seg_map"]).long(),
            "up_arrow": torch.from_numpy(gt["up_arrow"]).float(),
            "down_arrow": torch.from_numpy(gt["down_arrow"]).float(),
            "up_bound": torch.from_numpy(gt["up_bound"]).float(),
            "down_bound": torch.from_numpy(gt["down_bound"]).float(),
        }
        return x, target


def collate(batch):
    """Stack images and each target map along a new batch dim."""
    imgs = torch.stack([b[0] for b in batch], 0)
    keys = batch[0][1].keys()
    targets = {k: torch.stack([b[1][k] for b in batch], 0) for k in keys}
    return imgs, targets
