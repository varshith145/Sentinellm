#!/usr/bin/env python3
"""
Build the Triton model repository layout for SentinelLM's semantic detector.

Copies an ONNX export into <model repo>/1/model.onnx — Triton's required
<model>/<version>/model.<ext> layout. Each model's config.pbtxt lives one
level up (e.g. triton_deploy/models/sentinellm/config.pbtxt) and is
committed directly since it's small and doesn't change per export.

Two models by default:
  - sentinellm       <- model/onnx/model.onnx        (full precision)
  - sentinellm_int8  <- model/onnx-int8/model.onnx    (dynamic INT8, see
                        model/quantize_onnx.py) — skipped if that directory
                        doesn't exist, so this script still works for
                        anyone who hasn't run quantization.

Usage:
    python model/export_onnx.py             # if model/onnx/ doesn't exist yet
    python model/quantize_onnx.py            # optional, for the int8 model
    python triton_deploy/build_model_repo.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MODELS_DIR = REPO_ROOT / "triton_deploy" / "models"

# (Triton model name, source ONNX directory) — source dir must contain model.onnx
MODELS = [
    ("sentinellm", REPO_ROOT / "model" / "onnx"),
    ("sentinellm_int8", REPO_ROOT / "model" / "onnx-int8"),
]


def build_one(name: str, source_dir: Path) -> bool:
    source = source_dir / "model.onnx"
    if not source.exists():
        print(f"Skipping {name}: {source} not found.")
        return False

    dest = MODELS_DIR / name / "1" / "model.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    print(f"Copied {source} -> {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


def main() -> None:
    built = [name for name, source_dir in MODELS if build_one(name, source_dir)]
    if not built:
        raise SystemExit("No models built — run `python model/export_onnx.py` first.")


if __name__ == "__main__":
    main()
