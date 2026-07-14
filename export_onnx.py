"""Export a trained RCLane checkpoint to a single ONNX model file.

The exported model accepts normalized NCHW float32 images at 320x800 and emits
the five raw RCLane prediction maps. Only the batch dimension is dynamic because
the model's learned positional embeddings are tied to the training resolution.

Example:
    python export_onnx.py \
        --checkpoint checkpoints/rclane_b0_e19.pth \
        --output exports/rclane_b0_e19.onnx \
        --check-runtime
"""

import argparse
import os
from pathlib import Path

import torch
from torch import nn

from rclane import RCLane


IMAGE_HEIGHT = 320
IMAGE_WIDTH = 800
OUTPUT_NAMES = (
    "seg_map",
    "up_arrow",
    "down_arrow",
    "up_bound",
    "down_bound",
)


class RCLaneOnnxWrapper(nn.Module):
    """Give the dictionary model output a stable ONNX output contract."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def _checkpoint_args(checkpoint):
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if isinstance(args, dict):
        return args
    try:
        return vars(args)
    except TypeError:
        return {}


def load_model(checkpoint_path, vision=None):
    """Load a training checkpoint or a raw state dict on CPU."""
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dict")

    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint['model'] must be a state dict")
    if state_dict and all(name.startswith("module.") for name in state_dict):
        state_dict = {
            name.removeprefix("module."): value
            for name, value in state_dict.items()
        }

    saved_args = _checkpoint_args(checkpoint)
    saved_vision = saved_args.get("vision")
    if vision is not None and saved_vision is not None and vision != saved_vision:
        raise ValueError(
            f"--vision={vision!r} does not match checkpoint vision "
            f"{saved_vision!r}"
        )
    vision = vision or saved_vision or "b0"

    model = RCLane(
        vision=vision,
        img_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        use_pos_embed=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint, vision


def check_onnx(path):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX validation requires the 'onnx' package; install requirements.txt"
        ) from exc

    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    return graph


def check_runtime(path, wrapper, example):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "runtime verification requires 'onnxruntime' "
            "(pip install onnxruntime)"
        ) from exc

    with torch.no_grad():
        torch_outputs = [output.cpu().numpy() for output in wrapper(example)]
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    ort_outputs = session.run(None, {"images": example.cpu().numpy()})
    if len(ort_outputs) != len(OUTPUT_NAMES):
        raise RuntimeError(
            f"ONNX Runtime returned {len(ort_outputs)} outputs; "
            f"expected {len(OUTPUT_NAMES)}"
        )

    comparisons = []
    for name, expected, actual in zip(OUTPUT_NAMES, torch_outputs, ort_outputs):
        if expected.shape != actual.shape:
            raise RuntimeError(
                f"{name} shape mismatch: PyTorch {expected.shape}, "
                f"ONNX Runtime {actual.shape}"
            )
        max_abs = float(np.max(np.abs(expected - actual)))
        comparisons.append((name, actual.shape, max_abs))
        if not np.allclose(expected, actual, rtol=2e-4, atol=2e-4):
            raise RuntimeError(
                f"{name} numerical mismatch (max abs error {max_abs:.6g})"
            )
    return comparisons


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="training checkpoint or raw RCLane state dict")
    parser.add_argument("--output", default=None,
                        help="output .onnx path; defaults beside the checkpoint")
    parser.add_argument("--vision", choices=["b0", "b1", "b2"], default=None,
                        help="model variant; defaults to checkpoint args, then b0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="example batch used while tracing")
    parser.add_argument("--static-batch", action="store_true",
                        help="do not make the ONNX batch dimension dynamic")
    parser.add_argument("--check-runtime", action="store_true",
                        help="compare PyTorch and ONNX Runtime outputs on CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.opset < 17:
        raise ValueError("--opset must be at least 17")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = Path(args.output).expanduser() if args.output else (
        checkpoint_path.with_suffix(".onnx")
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    model, checkpoint, vision = load_model(checkpoint_path, args.vision)
    wrapper = RCLaneOnnxWrapper(model).eval()
    torch.manual_seed(0)
    example = torch.randn(
        args.batch_size, 3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32
    )

    dynamic_axes = None if args.static_batch else {
        "images": {0: "batch"},
        **{name: {0: "batch"} for name in OUTPUT_NAMES},
    }
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    print(
        f"exporting checkpoint={checkpoint_path.name} epoch={epoch} "
        f"vision={vision} opset={args.opset}"
    )
    print(
        f"input=images[{args.batch_size},3,{IMAGE_HEIGHT},{IMAGE_WIDTH}] "
        f"dynamic_batch={not args.static_batch}"
    )

    try:
        torch.onnx.export(
            wrapper,
            (example,),
            str(temporary_path),
            input_names=["images"],
            output_names=list(OUTPUT_NAMES),
            opset_version=args.opset,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
            external_data=False,
        )
        graph = check_onnx(temporary_path)
        if args.check_runtime:
            comparisons = check_runtime(temporary_path, wrapper, example)
            for name, shape, max_abs in comparisons:
                print(f"verified {name}: shape={shape} max_abs={max_abs:.6g}")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    size_mib = output_path.stat().st_size / (1024 ** 2)
    print(
        f"ONNX OK: {output_path} ({size_mib:.1f} MiB, "
        f"nodes={len(graph.graph.node)})"
    )
    print("outputs=" + ",".join(OUTPUT_NAMES))


if __name__ == "__main__":
    main()
