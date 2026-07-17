# RCLane -- PyTorch port

A faithful PyTorch port of RCLane (relay-chain lane detection), reimplemented from the
original MindSpore code ([lpplbiubiubiub/RCLane](https://github.com/lpplbiubiubiub/RCLane)).

## Layout

**Core (shared, dataset-agnostic)**
- `rclane.py` -- `MixVisionTransformer` (MiT/SegFormer encoder, b0/b1/b2), `RCLaneHead`
  (MLP-fuse decoder + 5 conv branches), and the `RCLane` wrapper: image `(B,3,320,800)`
  -> dict of 5 maps `seg_map`, `up_arrow`, `down_arrow`, `up_bound`, `down_bound`.
- `loss.py` -- `RCLaneLoss`: OHEM CE (seg, 15:1) + SmoothL1 (arrow/bound, foreground only).
- `encode.py` -- `encode()`: lane polylines -> the 5 GT maps (NumPy + shapely).
- `decode.py` -- `decode()`: relay-chain crawl -> lane polylines (point-NMS, forward/backward
  chains, IoU-NMS). Verified by an encode->decode round trip (~3px error).
- `dataset.py` -- `LaneEncodeDataset`: shared base (resize + normalize + `encode` + sparse
  GT cache). A concrete dataset only implements `_load`.
- `train.py` -- training loop (AdamW, lr 6e-4, poly LR). Selects the dataset via
  `--dataset {carla,culane,curvelanes}`, importing each loader lazily.
- `eval_checkpoints.py` -- rank one checkpoint or a glob of checkpoints with the
  lane-IoU F1 metric; writes results after every model.
- `export_onnx.py` -- export a checkpoint to ONNX with a dynamic batch dimension
  and stable names for all five prediction maps.

**Datasets (one file each, added per branch / merged into `dev`)**
- `dataset_carla.py` -- CARLA LaneATT JSONL (the primary target).
- `dataset_culane.py` -- CULane `.lines.txt` (paper F1 comparison).
- `dataset_curvelanes.py` -- CurveLanes `.lines.json`.

## Validation
`python rclane.py` runs the forward pass and compares parameters to the paper:

| Variant | Paper | This port |
|---|---|---|
| S (b0) | 6.3M | **6.2M** |
| M (b1) | 17.2M | 19.9M |
| L (b2) | 30.9M | **30.9M** |

Backbone + head are faithful (b0/b2 near-exact). The pipeline was verified end-to-end with
an overfit run: the loss drops 42.5 -> 7.2 over 15 epochs and `decode` recovers the lanes.

## Deviations from the original (read before training)
1. **Positional embedding**: the original MiT adds absolute pos-embed (hardcoded for 320x800).
   `use_pos_embed=True` (default) matches it but locks the input size and blocks pretrained
   MiT; `False` = standard SegFormer, so ImageNet-pretrained MiT-b0/b1/b2 can be loaded.
2. **sr_ratio at stage 2**: the MindSpore code uses `sr_ratios[0]` (likely a bug); here it is
   `sr_ratios[i]` per stage.
3. `embedding_dim`/`middle_dim` default to the SegFormer convention (256 for b0, 768 else).

## Note found during testing
RCLane seg maps are low-magnitude by design (OHEM 15:1 biases toward background), so
foreground probabilities sit around ~0.25-0.5. Seed `decode` with `seed_threshold=0.3` on an
under-trained model, or train longer / on GPU with a pretrained MiT for sharper seg.

## Not done yet
- Data augmentation (currently resize + normalize only).

## Run
```bash
python rclane.py  # smoke test: shapes + params
python loss.py    # smoke test: loss + backward
python encode.py  # smoke test: GT geometry
python decode.py  # smoke test: encode->decode round trip

# CARLA (primary target)
python train.py --dataset carla --data-root ../RCLane/data/dataset \
    --vision b0 --epochs 20 --batch 32 --device cuda

# CULane (paper F1 comparison)
python train.py --dataset culane --data-root <CULANE_ROOT> \
    --train-list list/train_gt.txt --vision b0 --epochs 20 --batch 32 --device cuda

# CurveLanes
python train.py --dataset curvelanes --data-root <CURVELANES_ROOT> \
    --train-list train/train.txt --vision b0 --epochs 20 --batch 32 --device cuda

# Rank CARLA checkpoints by F1 (add --with-loss only when validation loss is needed)
python eval_checkpoints.py --data-root data/dataset \
    --eval-list ../rawimages/Town04_Opt/clear_sunset/label_raw_train.json \
    --checkpoints 'job_artifacts/<job-id>/carla-b0/*.pth' \
    --output eval_results/town04_clear_sunset.json \
    --eval-batch 16 --eval-workers 1 --eval-decode-workers 9

# Export a checkpoint and verify its outputs with ONNX Runtime
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

The exported `seg_map` contains logits. Apply softmax over channel dimension and
pass foreground channel 1 to the relay-chain decoder.

> Needs `torch`; `encode.py`/datasets also need `shapely` + `opencv-python`.
