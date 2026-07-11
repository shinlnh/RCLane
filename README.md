# RCLane -- PyTorch port

A faithful PyTorch reimplementation of **RCLane** (relay-chain lane detection,
ECCV 2022), ported from the original MindSpore code:
[lpplbiubiubiub/RCLane](https://github.com/lpplbiubiubiub/RCLane).

RCLane predicts, at every foreground pixel, a transfer vector to the next relay
point and a distance to the lane end. Decoding then crawls those relay chains
forward and backward into full lane polylines, which makes the method suitable
for curved lanes, Y-shapes and near-horizontal lanes.

## Project layout

**Core**
- `rclane.py` - MixVisionTransformer / SegFormer-style backbone, RCLane head and
  model wrapper.
- `loss.py` - OHEM cross entropy for segmentation plus SmoothL1 losses for relay
  arrows and bounds.
- `encode.py` - lane polylines to RCLane supervision maps.
- `decode.py` - relay-chain outputs back to lane polylines.
- `dataset.py` - shared lane dataset base with resize, normalization and GT cache.
- `train.py` - CUDA/AMP training loop, dataset selector, checkpoint resume,
  validation loss and CULane-style F1 validation.

**Dataset loaders**
- `dataset_carla.py` - CARLA LaneATT JSONL.
- `dataset_culane.py` - CULane `.lines.txt`.
- `dataset_curvelanes.py` - CurveLanes `.lines.json`.

## Datasets

Download the dataset you need and point `--data-root` at its extracted folder.

| Dataset | Download |
|---|---|
| **CULane** | [xingangpan.github.io/projects/CULane.html](https://xingangpan.github.io/projects/CULane.html) - official release. Download every `driver_*` archive plus `laneseg_label_w16.tar.gz` and `list.tar.gz`. |
| **CurveLanes** | [github.com/SoulmateB/CurveLanes](https://github.com/SoulmateB/CurveLanes) - official release. |
| **CARLA** | [huggingface.co/datasets/BanVienCorp/dataset_laneatt_fullmap](https://huggingface.co/datasets/BanVienCorp/dataset_laneatt_fullmap/tree/feat%2Fadd-dataset-laneatt-fulltown-clean) - CARLA-simulated lane data. |

Large datasets, generated GT caches and checkpoints are intentionally ignored by
git. Keep them under `data/`, `gt_cache_*` and `checkpoints/` locally.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run quick smoke tests:

```bash
python rclane.py
python loss.py
python encode.py
python decode.py
```

## Train

CULane with CUDA, AMP, 8 dataloader workers, GT cache, validation loss and
CULane-style F1:

```bash
python train.py --dataset culane --data-root data/CULane \
    --train-list list/train_gt.txt --eval-list list/val_gt.txt --eval-f1 \
    --vision b0 --epochs 20 --batch 4 --eval-batch 4 \
    --workers 8 --prefetch 4 --device cuda --amp \
    --cache-dir ./gt_cache_culane --out ./checkpoints/f1_resume \
    --log-every 10
```

Resume from the latest checkpoint:

```bash
python train.py --dataset culane --data-root data/CULane \
    --train-list list/train_gt.txt --eval-list list/val_gt.txt --eval-f1 \
    --vision b0 --epochs 20 --batch 4 --eval-batch 4 \
    --workers 8 --prefetch 4 --device cuda --amp \
    --cache-dir ./gt_cache_culane --out ./checkpoints/f1_resume \
    --resume ./checkpoints/f1_resume/last.pth --log-every 10
```

Other datasets use the same trainer:

```bash
python train.py --dataset carla --data-root <CARLA_ROOT> --device cuda
python train.py --dataset curvelanes --data-root <CURVELANES_ROOT> \
    --train-list train/train.txt --device cuda
```

## Checkpoints

`train.py` saves full resumable checkpoints containing the model, optimizer,
AMP scaler, epoch counters, monitor score, arguments and RNG state.

- `last.pth` - latest epoch checkpoint.
- `best.pth` - best checkpoint for the active monitor.
- `rclane_<vision>_e<epoch>.pth` - per-epoch checkpoint.

When `--eval-f1` is enabled, `best.pth` tracks `val_f1` in max mode. With only
`--eval-list`, it tracks `val_loss` in min mode. Without validation, it tracks
training loss.

## Validation

`python rclane.py` runs a forward pass and compares parameter counts with the
paper:

| Variant | Paper | This port |
|---|---:|---:|
| S (b0) | 6.3M | 6.2M |
| M (b1) | 17.2M | 19.9M |
| L (b2) | 30.9M | 30.9M |

The current validation path computes CULane-style lane-level precision, recall
and F1 by decoding predicted lanes, rasterizing lanes with the CULane line width
convention, matching by IoU, and reporting the final F1 score.

## Credits

Method: *RCLane: Relay Chain Prediction for Lane Detection*, Xu et al., ECCV
2022. Original MindSpore implementation by lpplbiubiubiub. Licensed under MIT
(see `LICENSE`).
