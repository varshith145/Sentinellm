#!/usr/bin/env python3
"""
Export the fine-tuned DistilBERT NER model to ONNX.

Resolution order matches gateway/app/detectors/semantic.py's SemanticDetector:
a local model/trained/ directory wins if present and non-empty, otherwise
falls back to the Hugging Face Hub id the gateway itself uses in production
(model/trained/ is gitignored — 40GB of training checkpoints — so this is
the only source available in CI or a fresh checkout).

Usage:
    python model/export_onnx.py [--source PATH_OR_HUB_ID] [--out DIR]

Requires: pip install optimum-onnx onnxruntime
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_LOCAL_DIR = REPO_ROOT / "model" / "trained"
DEFAULT_HUB_ID = "varshith145/sentinellm-pii-ner"  # mirrors gateway/app/config.py's semantic_model_id
DEFAULT_OUTPUT = REPO_ROOT / "model" / "onnx"


def resolve_source(override: str | None) -> str:
    if override:
        return override
    if DEFAULT_LOCAL_DIR.exists() and any(DEFAULT_LOCAL_DIR.iterdir()):
        return str(DEFAULT_LOCAL_DIR)
    return DEFAULT_HUB_ID


def export(source: str, output: Path) -> Path:
    from optimum.exporters.onnx import main_export

    output.mkdir(parents=True, exist_ok=True)
    main_export(model_name_or_path=source, output=output, task="token-classification")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help="Local model dir or HF Hub id (default: model/trained if present, else the Hub id)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    source = resolve_source(args.source)
    output = Path(args.out)
    print(f"Exporting {source} -> {output}")
    export(source, output)

    print("Wrote:")
    for f in sorted(output.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:30s} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
