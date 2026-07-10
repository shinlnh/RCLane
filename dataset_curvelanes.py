"""
CurveLanes dataset for RCLane.

CurveLanes layout (official):
    <root>/train/images/<name>.jpg
    <root>/train/labels/<name>.lines.json
    <root>/train/train.txt              list of image paths (relative to <root>)
    <root>/valid/…                      same structure

Label JSON:
    {"Lines": [ [ {"x": "123.4", "y": "567.8"}, ... ], ... ]}
    -> a list of lanes, each a list of {x, y} points (values may be strings).

CurveLanes images have VARYING resolution (2560x1440, 1570x660, 1280x720, ...),
which is fine: lanes are read in each image's own pixel space and the base class
resizes to 800x320 per image. RCLane generates its own GT via `encode`, so only the
images + `.lines.json` are needed.
"""

import os
import json

import cv2

from dataset import LaneEncodeDataset


def _label_path(img_rel):
    """train/images/x.jpg -> train/labels/x.lines.json"""
    p = img_rel.replace("/images/", "/labels/")
    base, _ = os.path.splitext(p)
    return base + ".lines.json"


class CurveLanesDataset(LaneEncodeDataset):
    def __init__(self, list_file, data_root, img_size=(320, 800),
                 cache_dir=None, max_samples=None):
        super().__init__(img_size, cache_dir)
        self.data_root = data_root
        with open(list_file) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.img_paths = [ln.split()[0].lstrip("/") for ln in lines]
        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _parse_lines_json(path):
        """Parse a CurveLanes .lines.json into [[(x,y), ...], ...]."""
        lanes = []
        if not os.path.exists(path):
            return lanes
        with open(path) as f:
            data = json.load(f)
        for line in data.get("Lines", []):
            pts = [(float(p["x"]), float(p["y"])) for p in line]
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
        anno = os.path.join(self.data_root, _label_path(img_rel))
        lanes = self._parse_lines_json(anno)
        return img, lanes, ow, oh, img_rel


# --------------------------------------------------------------------------- #
#  smoke test -- builds a tiny FAKE CurveLanes sample (no real data needed) and
#  verifies the loader parses annotations and produces GT tensors.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="fake_curvelanes_")
    img_dir = os.path.join(tmp, "train", "images")
    lbl_dir = os.path.join(tmp, "train", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # a fake 2560x1440 image (CurveLanes' common resolution)
    cv2.imwrite(os.path.join(img_dir, "000000.jpg"), np.zeros((1440, 2560, 3), np.uint8))

    # 2 lanes as {"Lines": [[{x,y}, ...], ...]} (x,y as strings like the real data)
    ys = list(range(600, 1440, 40))
    line1 = [{"x": str(1000 + (y - 600) * 0.4), "y": str(y)} for y in ys]
    line2 = [{"x": str(1600 - (y - 600) * 0.4), "y": str(y)} for y in ys]
    with open(os.path.join(lbl_dir, "000000.lines.json"), "w") as f:
        json.dump({"Lines": [line1, line2]}, f)

    with open(os.path.join(tmp, "train", "train.txt"), "w") as f:
        f.write("train/images/000000.jpg\n")

    ds = CurveLanesDataset(
        list_file=os.path.join(tmp, "train", "train.txt"),
        data_root=tmp,
        cache_dir=None,
    )
    print("dataset size:", len(ds))
    lanes = ds._parse_lines_json(os.path.join(lbl_dir, "000000.lines.json"))
    print("parsed lanes:", len(lanes), "| lane0 pts:", len(lanes[0]))
    x, tgt = ds[0]
    print("image:", tuple(x.shape), "| seg fg pixels:", int((tgt["seg_map"] > 0).sum()))
    for k, v in tgt.items():
        assert tuple(v.shape)[-2:] == (320, 800), f"bad shape for {k}"
    assert int((tgt["seg_map"] > 0).sum()) > 0, "no foreground -- parse/encode failed"
    print("OK -- CurveLanes loader parses .lines.json and produces GT.")
