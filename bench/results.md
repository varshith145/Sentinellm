# SentinelLM load test results

Generated: 2026-07-29, Apple M4, 10 cores, 16GB RAM.

**This measures `/scan` (detection pipeline: regex + Presidio + semantic NER)
over real HTTP, not `/v1/chat/completions` end-to-end.** The LLM call is a
separate, network- and model-dependent cost that this number does not include
— see the script docstring for why that's the honest thing to measure here.

## Current numbers (4 workers, semantic short-circuit)

```bash
python bench/load_test.py --url http://localhost:8000 --concurrency 100 --duration 60
```

| Concurrency | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 10 | 238.8 req/s | 34.69ms | 87.21ms | 98.87ms |
| 100 | 258.5 req/s | 362.54ms | 732.37ms | 989.32ms |

`gateway/Dockerfile`'s `CMD` now runs `--workers 4` by default (env-driven via
`SENTINELLM_WORKERS`) against this same detection pipeline, with one
architectural change: the semantic pass only fires when the fast passes
(regex, Presidio) didn't already reach a confident conclusion — see below.

## What changed, and what didn't (in the order they were tried)

The original single-worker measurement (below) showed identical throughput
at concurrency 10 and 100 (~130 req/s either way) — proof the process was
saturated, not that requests were individually slow. Diagnosing *why*
before changing anything:

1. **Isolated the cost.** Running the same load test with
   `SENTINELLM_SEMANTIC_MODEL_ENABLED=false` (regex + Presidio only) jumped
   throughput from 126.6 to 521.5 req/s — the semantic pass (fine-tuned
   DistilBERT) is the dominant cost, not Presidio/spaCy.
2. **`torch.set_num_threads(1)`, tried first because it's free.** No
   measurable change (127.3 req/s, within noise of baseline). Confirmed via
   `top` during the run: the process used ~2.5–3.7 of 10 cores with the
   system 60% idle — this was never a thread-oversubscription problem, so
   pinning thread count fixed nothing. Kept anyway (see
   `gateway/app/detectors/semantic.py`) since it's a correct default with no
   downside, just not the fix.
3. **4 uvicorn worker processes alone** (no other change): 160.0 req/s —
   only 1.26x, and p95 got *worse* (1005.78ms vs 893.53ms). The idle-CPU
   headroom from step 2 suggested more processes should help more than
   this; the per-process ceiling — most likely GIL contention around the
   CPU-bound inference post-processing, though the `ThreadPoolExecutor(
   max_workers=2)` cap in `presidio_detector.py` wasn't ruled out as a
   contributor either — meant throwing more processes at a still-expensive
   per-request cost mostly added scheduling contention instead of scaling
   cleanly.
4. **Semantic short-circuit, single worker:** 202.9 req/s (1.6x) — skip the
   ~150ms semantic pass when a regex match (Luhn-validated card number,
   exact email/SSN/API-key shape) already gives a confident answer. This
   reduces total work instead of just spreading the same work across more
   processes, which is why it outperformed step 3 alone.
5. **Short-circuit + 4 workers combined:** 258.5 req/s (2.04x over
   baseline), p95 down from 913.53ms to 732.37ms (–20%). The two compound:
   fewer requests need the expensive pass, and the ones that do are spread
   across processes.
6. **ONNX + int8 quantization: not done.** Marked optional going in; skipped
   once steps 4–5 already produced a real, explained number without adding
   a new inference runtime to validate.

**The short-circuit's accuracy cost was checked, not assumed:** gating on
*any* confident fast-pass finding (including Presidio's contextual
PERSON_NAME) dropped `eval/run_eval.py`'s micro-F1 from 0.9574 to 0.8298 —
a name mention next to obfuscated content that only semantic can see was
enough to wrongly skip it. Restricting the skip to regex-only findings
(exact structural matches, not contextual ones) restored F1 to exactly
0.9574 — zero measured accuracy cost. Full reasoning in
`gateway/app/detectors/orchestrator.py`.

## Original single-worker baseline (before this optimization pass)

| Concurrency | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 10 | 132.6 req/s | 74.79ms | 85.84ms | 95.62ms |
| 100 | 127.2 req/s | 764.81ms | 913.53ms | 1031.57ms |

Throughput was identical at both concurrency levels (~130 req/s) — that's
the real ceiling of one `--workers 1` process doing CPU-bound Presidio/spaCy
+ DistilBERT inference in a thread pool. This is the number that motivated
the investigation above rather than being quoted as-is.
