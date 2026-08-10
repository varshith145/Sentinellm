"""
Parity test: ONNX export vs the original PyTorch model, for the fine-tuned
DistilBERT NER model (model/export_onnx.py).

Runs the same 50 fixed inputs (a deterministic slice of the real training
data — model/data/synthetic_obfuscated.jsonl and hard_negatives.jsonl, both
tracked in git) through PyTorch and through the ONNX export, one input at a
time with no padding — matching gateway/app/detectors/semantic.py's
_detect_sync, and avoiding the spurious argmax mismatches padding would
introduce on pad positions.

Requires optimum-onnx + onnxruntime (not part of gateway/requirements.txt —
these are export/CI-only tooling, kept out of the runtime image). Skips
cleanly if they aren't installed, so `pytest tests/` in the main CI job
(which doesn't install them) doesn't fail collection; the dedicated
`onnx-parity` CI job installs them and actually runs this.

model/trained/ is gitignored (40GB of checkpoints), so this pulls the
fine-tuned model from the Hugging Face Hub id the gateway itself falls back
to in production — see export_onnx.py's resolve_source().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("optimum.exporters.onnx")
pytest.importorskip("onnxruntime")

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "model"))

from export_onnx import DEFAULT_HUB_ID, export

MAX_LOGIT_DIFF = 1e-4


def _load_test_inputs() -> list[str]:
    data_dir = REPO_ROOT / "model" / "data"
    texts: list[str] = []
    for filename, count in [
        ("synthetic_obfuscated.jsonl", 40),
        ("hard_negatives.jsonl", 10),
    ]:
        with open(data_dir / filename) as f:
            lines = [json.loads(line)["text"] for line in f]
        texts.extend(lines[:count])
    return texts


TEST_INPUTS = _load_test_inputs()
assert len(TEST_INPUTS) == 50
assert len(set(TEST_INPUTS)) == 50, (
    "duplicate inputs in the fixed 50 — coverage would be inflated"
)


@pytest.fixture(scope="module")
def onnx_model_dir(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("onnx_export")
    return export(DEFAULT_HUB_ID, out_dir)


@pytest.fixture(scope="module")
def pytorch_model():
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HUB_ID)
    model = AutoModelForTokenClassification.from_pretrained(DEFAULT_HUB_ID)
    model.eval()
    return tokenizer, model


def test_onnx_matches_pytorch(onnx_model_dir, pytorch_model):
    import torch
    from optimum.onnxruntime import ORTModelForTokenClassification

    tokenizer, pt_model = pytorch_model
    onnx_model = ORTModelForTokenClassification.from_pretrained(onnx_model_dir)

    max_diff = 0.0
    mismatches: list[str] = []
    for text in TEST_INPUTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            pt_logits = pt_model(**inputs).logits
        onnx_logits = onnx_model(**inputs).logits

        max_diff = max(max_diff, (pt_logits - onnx_logits).abs().max().item())

        if (
            torch.argmax(pt_logits, dim=-1).tolist()
            != torch.argmax(onnx_logits, dim=-1).tolist()
        ):
            mismatches.append(text)

    assert not mismatches, (
        f"argmax label mismatch on {len(mismatches)} inputs: {mismatches}"
    )
    assert max_diff < MAX_LOGIT_DIFF, (
        f"max abs logit diff {max_diff} >= {MAX_LOGIT_DIFF}"
    )
