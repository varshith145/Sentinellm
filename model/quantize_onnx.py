#!/usr/bin/env python3
"""
Apply dynamic INT8 quantization to the exported ONNX model.

"Dynamic" here means weights are quantized to INT8 ahead of time (offline,
by this script) but activation ranges are computed per-inference at
runtime — unlike static quantization, this needs no calibration dataset,
at the cost of a small per-op overhead computing those ranges on the fly.
Uses the `arm64` quantization preset (not `avx512_vnni`, which is x86-only)
to match this project's dev/deploy hardware.

Input is the graph model/export_onnx.py already produced, not the original
PyTorch model — quantization operates on the ONNX graph. Output directory
gets both the quantized model.onnx and a full copy of the tokenizer/config
files (ORTQuantizer.quantize copies these itself), so it's a complete,
independently loadable model directory — same shape as model/onnx/.

Usage:
    python model/export_onnx.py            # if model/onnx/ doesn't exist yet
    python model/quantize_onnx.py

Requires: pip install optimum-onnx onnxruntime
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_ONNX_DIR = REPO_ROOT / "model" / "onnx"
DEFAULT_OUTPUT = REPO_ROOT / "model" / "onnx-int8"


def quantize(onnx_dir: Path, output_dir: Path) -> Path:
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    if not (onnx_dir / "model.onnx").exists():
        raise SystemExit(
            f"{onnx_dir}/model.onnx not found. Run `python model/export_onnx.py` first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    # file_suffix=None -> writes exactly "model.onnx", matching what
    # gateway/app/detectors/semantic.py's onnx/triton backends expect,
    # instead of the library's default "model_quantized.onnx".
    quantizer.quantize(
        save_dir=str(output_dir), quantization_config=qconfig, file_suffix=None
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", default=str(DEFAULT_ONNX_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    output = Path(args.out)
    print(f"Quantizing {onnx_dir} -> {output}")
    quantize(onnx_dir, output)

    before_mb = (onnx_dir / "model.onnx").stat().st_size / 1024 / 1024
    after_mb = (output / "model.onnx").stat().st_size / 1024 / 1024
    print(f"model.onnx: {before_mb:.1f} MB -> {after_mb:.1f} MB")
    print("Wrote:")
    for f in sorted(output.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:30s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
