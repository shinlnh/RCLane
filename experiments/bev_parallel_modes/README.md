# Archived parallel BEV modes

This directory preserves the two experimental BEV post-processing modes that
were evaluated before the native C++ runtime became the active path. The files
are isolated from the production Python/C++ pipeline: nothing here changes raw
model decoding or the default perspective-to-BEV projection.

Both modes alter only the cubic curves rendered/exported in BEV. The decoded
camera-view lanes remain the model's raw output.

## Mode 1: parallel detected lanes, camera funnel only

This mode keeps only lane IDs detected in the current frame, rebuilds their BEV
cubics as offsets of one reference curve, and clips the result to the camera
funnel. It never synthesizes a missing lane.

```bash
.venv/bin/python -u experiments/bev_parallel_modes/test_video_bev_parallel_legacy.py \
  --model exports/rclane_b0_e19.onnx \
  --video raw_Town04_Opt_20260714_093110.mp4 \
  --output runs/bev_parallel_visible.mp4 \
  --provider tensorrt \
  --always-parallel-repair
```

## Mode 2: always complete four parallel lanes

This mode always exports P0-P3. Missing lane IDs are synthesized as parallel
offsets, and the BEV curves may extend outside the camera funnel.

```bash
.venv/bin/python -u experiments/bev_parallel_modes/test_video_bev_parallel_legacy.py \
  --model exports/rclane_b0_e19.onnx \
  --video raw_Town04_Opt_20260714_093110.mp4 \
  --output runs/bev_parallel_four.mp4 \
  --provider tensorrt \
  --complete-four-parallel-lanes
```

Use `--max-frames N` for a short smoke test. This archived runner processes one
frame at a time and is intended for reproducibility, not deployment benchmarking.
