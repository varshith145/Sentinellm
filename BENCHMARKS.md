# SentinelLM benchmarks

Generated: 2026-08-10 16:49 UTC
Git SHA: `ffc7bbc` (dirty tree — uncommitted changes present)
Hardware: arm64, 10 cores, 16GB RAM, Darwin 25.6.0

## Reproduce

```bash
python benchmarks/loadtest.py --url http://localhost:8000
```

Requires a running gateway (see README for startup). Deterministic by
construction: fixed payload set, fixed duration per level (20.0s),
payload order seeded with `--seed 42` (default 42) and round-robined
per worker — same inputs always produce the same request sequence.

## Gateway config at time of run

| Setting | Value |
|---|---|
| Workers | 4 (uvicorn, per gateway/Dockerfile default) |
| Active detectors | regex, presidio, semantic |
| Policy ID | default-v1 |
| DB backend | sqlite (aiosqlite) |

## Load test: `/scan` (detection pipeline, no LLM in the path)

| Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|
| 1 | 139.8 req/s | 9.81ms | 11.3ms | 11.92ms | 2796 | 0 |
| 10 | 259.1 req/s | 32.32ms | 78.73ms | 87.6ms | 5191 | 0 |
| 50 | 267.0 req/s | 152.32ms | 432.08ms | 674.66ms | 5389 | 0 |
| 100 | 269.0 req/s | 348.98ms | 636.5ms | 769.52ms | 5459 | 0 |

This measures `/scan` (regex + Presidio + semantic NER) over real HTTP —
not `/v1/chat/completions` end-to-end, which also pays an LLM round-trip.
See [`bench/results.md`](bench/results.md) for the prior single-point
optimization investigation (short-circuit + multi-worker) this baseline
was measured on top of.

## Eval: ensemble micro-F1

```bash
python eval/run_eval.py
```

| Metric | Value |
|---|---|
| Micro-F1 | 0.9574 |
| Precision | 1.0 |
| Recall | 0.9184 |

Read from `eval/results.md` at write time — run the command above first so
that file is current before running this sweep. Full methodology in
[`eval/results.md`](eval/results.md).
