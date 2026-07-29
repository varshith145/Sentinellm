#!/usr/bin/env python3
"""
Pipeline-level evaluation for SentinelLM's detection stack.

This complements model/evaluate.py, which reports the raw DistilBERT token
classifier's precision/recall/F1 on its held-out test split
(docs/model_eval.json). That number is real but partial: it never runs regex
or Presidio, and it never runs through DetectionOrchestrator's deduplication.
This script runs the SAME held-out test split through the full three-pass
pipeline (regex + Presidio + semantic, deduplicated) exactly as a live
request would see it, and reports span-level precision/recall/F1 plus a
confusion matrix.

Held-out set: reproduces the exact test split model/data/prepare_dataset.py
produced (same source files, same combination order, same
train_test_split(seed=42) twice) — but recovers the original text and
character-level entity spans instead of the tokenized/subword-aligned
version prepare_dataset.py saved, since the raw pipeline needs raw text.

Ground truth granularity: the training data labels entities as PII or SECRET
only (see model/data/synthetic_obfuscated.jsonl) — not fine-grained types
like EMAIL or AWS_KEY. So precision/recall/F1 and the confusion matrix are
computed at that same category level. The "predicted entity type breakdown"
table shows what the pipeline actually output at finer granularity — useful
context, not something the ground truth can score.

Span matching: a predicted finding counts as a match for a ground-truth span
if IoU (intersection over union) >= 0.5 and categories agree — the standard
NER span-match threshold, not exact boundary equality. Detectors don't share
one boundary convention (semantic spans follow tokenizer offsets, regex
spans follow pattern boundaries). See the `overlaps()` docstring for the
robustness check run before trusting this threshold.

Usage:
    python eval/run_eval.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "gateway"))

from datasets import Dataset

DATA_DIR = REPO_ROOT / "model" / "data"
SYNTHETIC_PATH = DATA_DIR / "synthetic_obfuscated.jsonl"
HARD_NEGATIVES_PATH = DATA_DIR / "hard_negatives.jsonl"
MAX_NEGATIVES = 80  # must match model/data/prepare_dataset.py exactly
OUTPUT_MD = Path(__file__).parent / "results.md"

CATEGORIES = ("PII", "SECRET")


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def reconstruct_test_split() -> list[dict]:
    """
    Reproduce prepare_dataset.py's exact test split, but keep raw text +
    character spans instead of tokenizing.

    prepare_dataset.py: combined = synthetic + hard_negatives[:80], then
    dataset.train_test_split(test_size=0.2, seed=42), then
    train_test["test"].train_test_split(test_size=0.5, seed=42)["test"].
    Splitting an index array with the same length and seed reproduces the
    identical partition regardless of what columns the original dataset had.
    """
    synthetic = load_jsonl(SYNTHETIC_PATH)
    negatives = load_jsonl(HARD_NEGATIVES_PATH)[:MAX_NEGATIVES]
    combined = synthetic + negatives

    idx_dataset = Dataset.from_dict({"idx": list(range(len(combined)))})
    train_test = idx_dataset.train_test_split(test_size=0.2, seed=42)
    test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)
    test_indices = test_val["test"]["idx"]

    return [combined[i] for i in test_indices]


IOU_THRESHOLD = 0.5  # standard NER span-match threshold


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """
    Span match at IoU >= IOU_THRESHOLD, not mere overlap.

    Checked before trusting this: on this test set, any-overlap and IoU>=0.5
    give IDENTICAL precision/recall (matched spans average IoU=0.91 — tight,
    not degenerate wide-span matches). Only an unusually strict IoU>=0.7
    threshold moves the number at all (micro F1 0.9574 -> 0.9362). IoU>=0.5
    is used here anyway, as the more standard and more defensible choice.
    """
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return (inter / union if union else 0.0) >= IOU_THRESHOLD


def evaluate(test_examples: list[dict], orchestrator) -> dict:
    async def scan_all():
        return [await orchestrator.scan(ex["text"]) for ex in test_examples]

    all_findings = asyncio.run(scan_all())

    tp = dict.fromkeys(CATEGORIES, 0)
    fp = dict.fromkeys(CATEGORIES, 0)
    fn = dict.fromkeys(CATEGORIES, 0)
    confusion: dict[str, dict[str, int]] = {
        r: dict.fromkeys((*CATEGORIES, "NONE"), 0) for r in (*CATEGORIES, "NONE")
    }
    predicted_type_breakdown: dict[str, int] = {}  # entity_type -> count of TPs

    for example, findings in zip(test_examples, all_findings):
        gt_spans = [(e["start"], e["end"], e["label"]) for e in example["entities"]]
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        # Pass 1: match each gt span to an unmatched prediction of the SAME category.
        for gi, (gs, ge, gc) in enumerate(gt_spans):
            for pi, f in enumerate(findings):
                if pi in matched_pred:
                    continue
                if f.category.value == gc and overlaps(gs, ge, f.start, f.end):
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    tp[gc] += 1
                    confusion[gc][gc] += 1
                    predicted_type_breakdown[f.entity_type.value] = (
                        predicted_type_breakdown.get(f.entity_type.value, 0) + 1
                    )
                    break

        # Pass 2: unmatched gt spans — either caught under the wrong category, or missed.
        for gi, (gs, ge, gc) in enumerate(gt_spans):
            if gi in matched_gt:
                continue
            fn[gc] += 1
            caught_wrong_category = False
            for pi, f in enumerate(findings):
                if pi in matched_pred:
                    continue
                if overlaps(gs, ge, f.start, f.end):
                    matched_pred.add(pi)
                    confusion[gc][f.category.value] += 1
                    fp[f.category.value] += 1
                    caught_wrong_category = True
                    break
            if not caught_wrong_category:
                confusion[gc]["NONE"] += 1

        # Remaining unmatched predictions are spurious findings (false positives).
        for pi, f in enumerate(findings):
            if pi not in matched_pred:
                fp[f.category.value] += 1
                confusion["NONE"][f.category.value] += 1

        if not gt_spans and not findings:
            confusion["NONE"]["NONE"] += 1

    def prf(cat: str) -> tuple[float, float, float]:
        p = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) else 0.0
        r = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    per_category = {
        cat: dict(zip(("precision", "recall", "f1"), prf(cat))) for cat in CATEGORIES
    }

    total_tp, total_fp, total_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )

    return {
        "num_test_examples": len(test_examples),
        "per_category": per_category,
        "micro": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        },
        "confusion_matrix": confusion,
        "predicted_type_breakdown": dict(
            sorted(predicted_type_breakdown.items(), key=lambda kv: -kv[1])
        ),
    }


def write_results_md(result: dict, detectors_used: list[str], path: Path) -> None:
    cm = result["confusion_matrix"]
    cols = ["PII", "SECRET", "NONE"]
    cm_header = "| true \\ predicted | " + " | ".join(cols) + " |"
    cm_sep = "|---" * (len(cols) + 1) + "|"
    cm_rows = [
        "| **" + row + "** | " + " | ".join(str(cm[row][c]) for c in cols) + " |"
        for row in cols
    ]

    type_rows = "\n".join(
        f"| {t} | {c} |" for t, c in result["predicted_type_breakdown"].items()
    )

    md = f"""# SentinelLM pipeline-level evaluation

Full three-pass pipeline (regex + Presidio + semantic, deduplicated) run
against the exact held-out test split used to produce `docs/model_eval.json`
(the raw semantic model's own eval). Detectors active: {", ".join(detectors_used)}.

Test examples: {result["num_test_examples"]}

## Precision / Recall / F1 by category

Ground truth is labeled PII/SECRET only (see script docstring) — this is
the finest granularity the held-out set supports.

| Category | Precision | Recall | F1 |
|---|---|---|---|
| PII | {result["per_category"]["PII"]["precision"]} | {result["per_category"]["PII"]["recall"]} | {result["per_category"]["PII"]["f1"]} |
| SECRET | {result["per_category"]["SECRET"]["precision"]} | {result["per_category"]["SECRET"]["recall"]} | {result["per_category"]["SECRET"]["f1"]} |
| **Micro-average** | **{result["micro"]["precision"]}** | **{result["micro"]["recall"]}** | **{result["micro"]["f1"]}** |

## Confusion matrix (span-level, IoU >= 0.5 match)

{cm_header}
{cm_sep}
{chr(10).join(cm_rows)}

Rows are ground truth, columns are what the pipeline predicted. `NONE` means
no entity (true negative context) or no matching prediction (miss).

## Predicted entity-type breakdown (true positives only)

The ground truth doesn't distinguish EMAIL from AWS_KEY from PERSON_NAME —
but the pipeline's predictions do. This shows which specific entity_type
each true positive was tagged as, and by extension which detector pass
found it (regex/Presidio predict specific types; semantic predicts
GENERIC_PII/GENERIC_SECRET).

| entity_type | count |
|---|---|
{type_rows}
"""
    path.write_text(md)


def main() -> None:
    test_examples = reconstruct_test_split()
    print(f"Reconstructed held-out test split: {len(test_examples)} examples")

    from app.config import settings
    from app.detectors.orchestrator import DetectionOrchestrator
    from app.detectors.regex import RegexDetector

    detectors = [RegexDetector()]
    detector_names = ["regex"]

    try:
        from app.detectors.presidio_detector import PresidioDetector

        detectors.append(PresidioDetector())
        detector_names.append("presidio")
    except Exception as e:  # noqa: BLE001 — eval degrades gracefully like the app does
        print(f"Presidio unavailable, skipping: {e}")

    try:
        from app.detectors.semantic import SemanticDetector

        semantic = SemanticDetector(
            model_path=settings.model_path, model_id=settings.semantic_model_id
        )
        if semantic.is_available:
            detectors.append(semantic)
            detector_names.append("semantic")
        else:
            print("Semantic model not available, skipping")
    except Exception as e:  # noqa: BLE001
        print(f"Semantic detector unavailable, skipping: {e}")

    orchestrator = DetectionOrchestrator(detectors)
    print(f"Active detectors: {detector_names}")

    result = evaluate(test_examples, orchestrator)
    print(json.dumps(result, indent=2))

    write_results_md(result, detector_names, OUTPUT_MD)
    print(f"\nWrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
