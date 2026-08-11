#!/usr/bin/env python3
"""
Build the Triton model repository layout for SentinelLM's semantic detector.

Copies the ONNX export (model/onnx/model.onnx, produced by
model/export_onnx.py) into triton_deploy/models/sentinellm/1/model.onnx —
Triton's required <model>/<version>/model.<ext> layout. config.pbtxt lives
one level up (triton_deploy/models/sentinellm/config.pbtxt) and is committed
directly since it's small and doesn't change per export.

Usage:
    python model/export_onnx.py          # if model/onnx/ doesn't exist yet
    python triton_deploy/build_model_repo.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SOURCE = REPO_ROOT / "model" / "onnx" / "model.onnx"
DEST = REPO_ROOT / "triton_deploy" / "models" / "sentinellm" / "1" / "model.onnx"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(
            f"{SOURCE} not found. Run `python model/export_onnx.py` first."
        )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, DEST)
    print(f"Copied {SOURCE} -> {DEST} ({DEST.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
