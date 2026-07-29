# SentinelLM pipeline-level evaluation

Full three-pass pipeline (regex + Presidio + semantic, deduplicated) run
against the exact held-out test split used to produce `docs/model_eval.json`
(the raw semantic model's own eval). Detectors active: regex, presidio, semantic.

Test examples: 61

## Precision / Recall / F1 by category

Ground truth is labeled PII/SECRET only (see script docstring) — this is
the finest granularity the held-out set supports.

| Category | Precision | Recall | F1 |
|---|---|---|---|
| PII | 1.0 | 0.9643 | 0.9818 |
| SECRET | 1.0 | 0.8571 | 0.9231 |
| **Micro-average** | **1.0** | **0.9184** | **0.9574** |

## Confusion matrix (span-level, IoU >= 0.5 match)

| true \ predicted | PII | SECRET | NONE |
|---|---|---|---|
| **PII** | 27 | 0 | 1 |
| **SECRET** | 0 | 18 | 3 |
| **NONE** | 0 | 0 | 12 |

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
| GENERIC_PII | 27 |
| GENERIC_SECRET | 18 |
