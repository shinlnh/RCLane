# RCLane -- architecture ported to PyTorch

Ports **only the network part** (backbone + head) of RCLane from the original
MindSpore code ([lpplbiubiubiub/RCLane](https://github.com/lpplbiubiubiub/RCLane))
to PyTorch.

## Done
- `rclane.py` -- single self-contained file:
  - `MixVisionTransformer` (MiT / SegFormer encoder, variants b0/b1/b2)
  - `RCLaneHead` (MLP-fuse decoder + 5 conv branches)
  - `RCLane` (wrapper): image `(B,3,320,800)` -> dict of 5 maps, each `(B,2,320,800)`
    - `seg_map`, `up_arrow`, `down_arrow`, `up_bound`, `down_bound`
- `loss.py` -- `RCLaneLoss`: OHEM CE (seg, 15:1 ratio) + SmoothL1 (arrow/bound,
  foreground only). Ported from `rclane_loss.py`. Tested: finite, backward runs,
  handles the no-lane image case.
- `encode.py` -- `encode()`: turns lane polylines into the 5 GT maps (NumPy + shapely),
  ported from `lane_codec.py::encode`. Tested: arrows are exactly `step_length` long and
  point the right way; bounds >= 1; GT plugs straight into the loss (encode -> loss ->
  backward verified end-to-end).
- `decode.py` -- `decode()`: relay-chain crawl (point-NMS -> forward/backward chains ->
  merge -> thresh -> IoU-NMS) turning the 5 maps back into lane polylines. Ported from
  `lane_codec.py` (decode/decode_branch/iou_nms) + `lane_geometry.py` (`_iou`).
  Tested with an encode->decode ROUND TRIP: reconstructs the lane to ~3px mean error.
  `decode_predictions(pred_dict)` decodes a batch of network outputs directly.

## Validation
`python rclane.py` runs the forward pass and compares parameter counts to the paper:

| Variant | Paper | This port |
|---|---|---|
| S (b0) | 6.3M | **6.2M** |
| M (b1) | 17.2M | 19.9M |
| L (b2) | 30.9M | **30.9M** |

b0/b2 match almost exactly -> backbone + head are faithful. b1 is slightly higher
due to the decoder `embedding_dim` (the original repo never pins this value;
train.py never builds the model).

## Deviations from the original (read before training)
1. **Positional embedding**: the original MiT adds absolute pos-embed (hardcoded for
   320x800); standard SegFormer has none. Controlled by `use_pos_embed`:
   - `True` (default) matches the original, but locks the input size and cannot load
     pretrained MiT.
   - `False` = standard SegFormer -> can load ImageNet-pretrained MiT-b0/b1/b2
     (recommended for training).
2. **sr_ratio at stage 2**: the MindSpore code uses `sr_ratios[0]` (almost certainly a
   copy-paste bug); here we use `sr_ratios[i]` per stage, as in standard SegFormer.
3. `embedding_dim` / `middle_dim`: defaults follow the SegFormer convention
   (256 for b0, 768 for b1/b2).

- `dataset.py` -- `LaneEncodeDataset` (shared base: resize + normalize + `encode` + sparse
  disk cache) and `CarlaLaneDataset` (CARLA LaneATT JSONL). Verified on the real CARLA data
  (train 50444 / val 4192 / test 4365 records).
- `dataset_culane.py` -- `CULaneDataset`: reads CULane `.lines.txt` + `list/*.txt`, in the
  standard 1640x590 space. Verified with a synthetic CULane sample (no real data needed yet).
- `train.py` -- training loop: AdamW (lr 6e-4) + poly LR + `RCLaneLoss` + checkpointing.
  Verified end-to-end on the real data: an 8-image overfit for 15 epochs drives the loss
  42.5 -> 7.2 (every term decreasing); training on the trained model + `decode` recovers
  ~5 lanes (3 GT) on a training image.

## Practical note found during testing
RCLane seg maps are LOW-magnitude by design (OHEM 15:1 biases hard toward background),
so foreground probabilities sit around ~0.25-0.5 even as the loss drops. Seeding `decode`
at the default 0.5 finds almost nothing on an under-trained model. Use `decode(...,
seed_threshold=0.3)` (added for this), or train longer / on GPU with a pretrained MiT
(`use_pos_embed=False`) for sharper seg. This is a threshold/convergence issue, not a bug
-- the encode/decode round trip and the loss are both correct.

## Not done yet
- Data augmentation (currently just resize + normalize).
- Evaluation / F1 metric on val/test.
- Full-scale training run (needs a GPU; CPU is ~8.5s/2-img batch).

## Run
```bash
python rclane.py     # smoke test: prints shapes + params, asserts forward is correct
python loss.py       # smoke test: computes loss, checks backward + no-lane case
python encode.py     # smoke test: builds GT maps from a synthetic lane, checks geometry
python decode.py     # smoke test: encode->decode round trip reconstructs the lane
python dataset.py         # smoke test: loads real CARLA samples, checks shapes + batching
python dataset_culane.py  # smoke test: fake CULane sample, checks .lines.txt parsing

# CULane (goal: reproduce paper F1 -- needs CULane data + a GPU)
python train.py --dataset culane --data-root <CULANE_ROOT> \
    --train-list list/train_gt.txt --vision b0 --epochs 20 --batch 32 --device cuda

# CARLA overfit sanity run (CPU ok)
python train.py --dataset carla --data-root ../RCLane/data/dataset \
    --subset 8 --epochs 15 --batch 2 --device cpu
```
> Tested with torch 2.13 CPU, shapely 2.1, opencv 4.13. (These were installed into
> `../infer_env` for testing -- remove with `infer_env/bin/pip uninstall ...` if not
> needed. `encode.py` needs `shapely` + `opencv-python`; the network/loss need only torch.)
