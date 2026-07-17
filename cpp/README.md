# RCLane native C++ runtime

This directory is the sequential deployment runtime. One frame completes
preprocessing, TensorRT inference, 1024-seed decode and raw-model BEV export
before the next frame begins. There is no inter-frame overlap. Rendering,
source-video decoding and video writing are outside the measured core latency.

The BEV output is a one-to-one projection of decoded model lanes. It does not
force curves to be parallel and does not synthesize missing lanes. The funnel
guard may only shorten a cubic's valid X domain; it never changes coefficients.

The build intentionally consumes TensorRT/CUDA SDK headers and shared libraries
from the target machine; serialized TensorRT engines must be rebuilt per GPU.

```bash
cmake -S cpp -B cpp/build -G "Unix Makefiles" \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT/include \
  -DTENSORRT_LIBRARY=/path/to/libnvinfer.so \
  -DCUDA_INCLUDE_DIR=/path/to/cuda/include \
  -DCUDART_LIBRARY=/path/to/libcudart.so
cmake --build cpp/build -j8
```

TensorRT and CUDA are auto-discovered in standard x86/Jetson locations. The
explicit `-D` paths above are useful for Python-wheel installations.

## One-frame parity harness

The runtime accepts either a preprocessed float32 NCHW tensor or a raw
1920x1080 BGR frame. It can dump maps, decoded image lanes and metric BEV
cubics for comparison with Python:

```bash
cpp/build/rclane_runtime \
  --engine exports/trt_cache/model.engine \
  --input-bgr /tmp/frame.bgr \
  --dump-prefix /tmp/cpp \
  --lanes-json /tmp/cpp_lanes.json \
  --bev-json /tmp/cpp_bev.json \
  --threads 8
```

## Sequential full-video benchmark

Pipe decoded BGR frames from FFmpeg. The reported `core_pipeline` contains
only preprocess + TensorRT (including transfers) + decode + BEV/cubic/funnel.

```bash
ffmpeg -loglevel error -i raw_Town04_Opt_20260714_093110.mp4 \
  -f rawvideo -pix_fmt bgr24 - | \
cpp/build/rclane_runtime \
  --engine exports/trt_cache/model.engine \
  --raw-bgr-stdin --source-width 1920 --source-height 1080 \
  --threads 8 --warmup 10 --timing-warmup 5 \
  --frames-jsonl runs/cpp_final_lanes.jsonl \
  --report runs/cpp_benchmark_final_full.json
```

Render those saved C++ results later without rerunning inference or affecting
the measured pipeline latency:

```bash
python cpp/render_cpp_results.py \
  --video raw_Town04_Opt_20260714_093110.mp4 \
  --results runs/cpp_final_lanes.jsonl \
  --benchmark-report runs/cpp_benchmark_final_full.json \
  --output runs/cpp_final_render_h264.mp4
```

On the development RTX 3050/i5-13420H machine, the complete 1768-frame final
video measured 13.39 ms median and 14.15 ms p95 core latency (74.67 FPS median)
with 1024 seeds and 8 CPU threads. TensorRT engines are GPU-architecture
specific and must be rebuilt on the deployment target.
