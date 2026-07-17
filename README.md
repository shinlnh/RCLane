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
- `train.py` - CUDA/AMP/DDP training loop, dataset selector, checkpoint resume,
  validation loss and parallel CULane-style F1 validation.
- `eval_checkpoints.py` - rank one checkpoint or a checkpoint glob by lane-IoU
  F1, writing results after every model.
- `export_onnx.py` - export a checkpoint to ONNX with a dynamic batch dimension
  and stable names for all five prediction maps.
- `test_video_onnx.py` - run ONNX Runtime inference on a video and render lane
  identities plus per-stage runtime statistics.
- `hf_train_carla.sh` - launch the optimized dual-GPU CARLA workflow on HF Jobs.

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

For two GPUs, launch one DDP process per GPU. Worker counts are per process:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --dataset carla --data-root <CARLA_ROOT> \
    --label label_train.json --eval-list label_val.json --eval-f1 \
    --vision b0 --epochs 20 --batch 64 --workers 21 --warm-cache \
    --eval-batch 64 --eval-workers 2 --eval-decode-workers 18 \
    --device cuda --amp --amp-dtype bfloat16
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

Rank a set of CARLA checkpoints without recomputing dense GT loss maps:

```bash
python eval_checkpoints.py --data-root <CARLA_ROOT> \
    --eval-list label_val.json \
    --checkpoints 'checkpoints/*.pth' \
    --output eval_results/checkpoint_eval.json
```

Add `--with-loss` when validation loss is also required.

## ONNX inference

Export a checkpoint and verify its outputs with ONNX Runtime:

```bash
python export_onnx.py --checkpoint checkpoints/rclane_b0_e19.pth \
    --output exports/rclane_b0_e19.onnx --check-runtime

# Build/cache a device-specific TensorRT FP16 engine and compare it to CUDA FP32
python build_tensorrt_engine.py --model exports/rclane_b0_e19.onnx \
    --cache-dir exports/trt_cache

# Benchmark the sequential production core (preprocess + engine + 1024-seed
# decode + raw-model BEV) without visualization/video-I/O
python benchmark_realtime.py --model exports/rclane_b0_e19.onnx \
    --video raw_Town04_Opt_20260714_093110.mp4 --provider tensorrt \
    --trt-cache-dir exports/trt_cache --cpu-threads 8 --max-seeds 1024 \
    --max-frames 300 --report runs/realtime_benchmark_1024seeds.json

# Render raw decoded lanes in BEV and export one cubic per detected lane.
# No lane is synthesized or forced parallel; cubics are clipped to camera FOV.
python test_video_bev_onnx.py --model exports/rclane_b0_e19.onnx \
    --video raw_Town04_Opt_20260714_093110.mp4 --provider tensorrt \
    --trt-cache-dir exports/trt_cache --decode-cpu-threads 8 \
    --decode-max-seeds 1024 --output runs/video_bev_e19.mp4
```

TensorRT cache files are tied to the TensorRT version and GPU compute
capability. Rebuild the cache on a different deployment GPU. The real-time
benchmark deliberately reports camera/video acquisition and visualization
separately: its core latency is the latency relevant to the ADAS algorithm,
whereas rendering and MP4 encoding are offline diagnostics.

For CUDA inference, use the CUDA 13 ONNX Runtime build pinned in
`requirements.txt`. Disable TF32 when exact lane decisions matter:

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "exports/rclane_b0_e19.onnx",
    providers=[
        ("CUDAExecutionProvider", {"device_id": "0", "use_tf32": "0"}),
        "CPUExecutionProvider",
    ],
)
```

The exported `seg_map` contains logits. Apply softmax over the channel dimension
and pass foreground channel 1 to the relay-chain decoder.

## Credits

Method: *RCLane: Relay Chain Prediction for Lane Detection*, Xu et al., ECCV
2022. Original MindSpore implementation by lpplbiubiubiub. Licensed under MIT
(see `LICENSE`).
