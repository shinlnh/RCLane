"""
CULane dataset for RCLane.

CULane layout:
    <root>/driver_*/…/<frame>.jpg            images (1640x590)
    <root>/driver_*/…/<frame>.lines.txt      annotation: one lane per line,
                                             "x1 y1 x2 y2 …" in 1640x590 space
    <root>/list/train_gt.txt                 "/img.jpg /seglabel.png f f f f"
    <root>/list/{val,test}.txt               "/img.jpg" per line

RCLane does not use the provided seg labels -- it generates its own GT via `encode`,
so only the images + `.lines.txt` are needed. Lanes are read in the original 1640x590
space; the base class handles resize to 800x320 and encoding.
"""

import os

import cv2

from dataset import LaneEncodeDataset


class CULaneDataset(LaneEncodeDataset):
    def __init__(self, list_file, data_root, img_size=(320, 800),
                 cache_dir=None, max_samples=None):
        super().__init__(img_size, cache_dir)
        self.data_root = data_root
        with open(list_file) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        # first whitespace-separated token is the image path (leading '/')
        self.img_paths = [ln.split()[0].lstrip("/") for ln in lines]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _parse_lines_txt(path):
        """Parse a CULane .lines.txt into a list of lanes [[(x,y), ...], ...]."""
        lanes = []
        if not os.path.exists(path):
            return lanes
        with open(path) as f:
            for ln in f:
                nums = ln.split()
                if len(nums) < 4:
                    continue
                coords = list(map(float, nums))
                pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)
                       if coords[i] >= 0]  # CULane uses valid x >= 0
                if len(pts) >= 2:
                    lanes.append(pts)
        return lanes

    def _load(self, idx):
        img_rel = self.img_paths[idx]
        img_path = os.path.join(self.data_root, img_rel)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        oh, ow = img.shape[:2]
        anno = os.path.join(self.data_root, os.path.splitext(img_rel)[0] + ".lines.txt")
        lanes = self._parse_lines_txt(anno)
        return img, lanes, ow, oh, img_rel


# --------------------------------------------------------------------------- #
#  smoke test -- builds a tiny FAKE CULane sample (no real data needed) and
#  verifies the loader parses annotations and produces GT tensors.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="fake_culane_")
    drv = os.path.join(tmp, "driver_1", "seq")
    os.makedirs(drv, exist_ok=True)
    os.makedirs(os.path.join(tmp, "list"), exist_ok=True)

    # a fake 1640x590 image
    img = np.zeros((590, 1640, 3), np.uint8)
    cv2.imwrite(os.path.join(drv, "00000.jpg"), img)

    # a .lines.txt with 2 lanes (points in 1640x590 space)
    with open(os.path.join(drv, "00000.lines.txt"), "w") as f:
        ys = list(range(250, 590, 20))
        lane1 = " ".join(f"{600 + (y-250)*0.3:.1f} {y}" for y in ys)
        lane2 = " ".join(f"{1000 - (y-250)*0.3:.1f} {y}" for y in ys)
        f.write(lane1 + "\n")
        f.write(lane2 + "\n")

    with open(os.path.join(tmp, "list", "train.txt"), "w") as f:
        f.write("/driver_1/seq/00000.jpg\n")

    ds = CULaneDataset(
        list_file=os.path.join(tmp, "list", "train.txt"),
        data_root=tmp,
        cache_dir=None,
    )
    print("dataset size:", len(ds))
    lanes = ds._parse_lines_txt(os.path.join(drv, "00000.lines.txt"))
    print("parsed lanes:", len(lanes), "| lane0 pts:", len(lanes[0]))
    x, tgt = ds[0]
    print("image:", tuple(x.shape), "| seg fg pixels:", int((tgt["seg_map"] > 0).sum()))
    for k, v in tgt.items():
        assert tuple(v.shape)[-2:] == (320, 800), f"bad shape for {k}"
    assert int((tgt["seg_map"] > 0).sum()) > 0, "no foreground -- parse/encode failed"
    print("OK -- CULane loader parses .lines.txt and produces GT.")
