"""
CARLA dataset for RCLane (BanVienCorp/dataset_laneatt_fullmap, LaneATT format).

Annotation is JSON-lines:
  - first line: {"Ys": [72 y-anchors, 0..1065 step 15]}
  - each record: {"lines": [[x per anchor, -2 = missing], ...],
                  "types": [...], "image": "TownXX/.../frame.jpg"}
Images are 1920x1080.

Lanes are read in the original 1920x1080 space; the base class resizes to 800x320
and runs `encode` to build the GT. RCLane generates its own GT, so no seg labels needed.
"""

import os
import json

import cv2

from dataset import LaneEncodeDataset


class CarlaLaneDataset(LaneEncodeDataset):
    _MISSING = -2

    def __init__(self, label_json, data_root, img_size=(320, 800),
                 cache_dir=None, max_samples=None):
        super().__init__(img_size, cache_dir)
        self.data_root = data_root
        with open(label_json) as f:
            lines = f.read().splitlines()
        self.Ys = json.loads(lines[0])["Ys"]
        self.records = [r for r in (json.loads(ln) for ln in lines[1:])
                        if "image" in r and "lines" in r]
        if max_samples is not None:
            self.records = self.records[:max_samples]

    def __len__(self):
        return len(self.records)

    def _load(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.data_root, rec["image"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        oh, ow = img.shape[:2]
        lanes = []
        for lane in rec["lines"]:
            pts = [(x, self.Ys[i]) for i, x in enumerate(lane) if x != self._MISSING]
            if len(pts) >= 2:
                lanes.append(pts)
        return img, lanes, ow, oh, rec["image"]


# --------------------------------------------------------------------------- #
#  smoke test -- builds a tiny FAKE CARLA sample (no real data needed) and
#  verifies the loader parses annotations and produces GT tensors.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="fake_carla_")
    img_rel = "Town01/clear_noon/000000.jpg"
    os.makedirs(os.path.join(tmp, os.path.dirname(img_rel)), exist_ok=True)
    cv2.imwrite(os.path.join(tmp, img_rel), np.zeros((1080, 1920, 3), np.uint8))

    ys = list(range(0, 1080, 15))                       # 72 anchors, like the real data
    lane1 = [700 + i * 6 for i in range(len(ys))]       # x per anchor
    lane2 = [1200 - i * 6 for i in range(len(ys))]
    label = os.path.join(tmp, "label_train.json")
    with open(label, "w") as f:
        f.write(json.dumps({"Ys": ys}) + "\n")
        f.write(json.dumps({"lines": [lane1, lane2], "types": [0, 0], "image": img_rel}) + "\n")

    ds = CarlaLaneDataset(label_json=label, data_root=tmp, cache_dir=None)
    print("dataset size:", len(ds))
    x, tgt = ds[0]
    print("image:", tuple(x.shape), "| seg fg pixels:", int((tgt["seg_map"] > 0).sum()))
    for k, v in tgt.items():
        assert tuple(v.shape)[-2:] == (320, 800), f"bad shape for {k}"
    assert int((tgt["seg_map"] > 0).sum()) > 0, "no foreground -- parse/encode failed"
    print("OK -- CARLA loader parses JSONL and produces GT.")
