# SentinelLM load test results

Generated: 2026-07-28 02:33 UTC
Command: `python bench/load_test.py --url http://127.0.0.1:8123 --concurrency 100 --duration 60.2`

**This measures `/scan` (detection pipeline: regex + Presidio + semantic NER)
over real HTTP, not `/v1/chat/completions` end-to-end.** The LLM call is a
separate, network- and model-dependent cost that this number does not include
— see the script docstring for why that's the honest thing to measure here.

| Metric | Value |
|---|---|
| Concurrency | 100 |
| Duration | 60.2s |
| Total requests | 7659 |
| Errors | 0 |
| Throughput | 127.2 req/s |
| p50 latency | 764.81 ms |
| p95 latency | 913.53 ms |
| p99 latency | 1031.57 ms |
| max latency | 1338.34 ms |

Decision breakdown: {'ALLOW': 2400, 'MASK': 3259, 'BLOCK': 2000}

**Hardware:** Apple M4, 10 cores, 16GB RAM. Gateway run as `uvicorn app.main:app --workers 1`
— the same single-worker configuration `gateway/Dockerfile`'s `CMD` uses in
production. All three detectors active (regex + Presidio/spaCy + fine-tuned
DistilBERT), model loaded from `model/trained`.

## Concurrency 100 vs 10 — why p95 is 913ms, not 86ms

A second run at `--concurrency 10 --duration 20` on the same process:

| Metric | Concurrency 10 | Concurrency 100 |
|---|---|---|
| Throughput | 132.6 req/s | 127.2 req/s |
| p50 latency | 74.79 ms | 764.81 ms |
| p95 latency | 85.84 ms | 913.53 ms |
| p99 latency | 95.62 ms | 1031.57 ms |

Throughput is identical across both runs (~130 req/s) — that's the real
ceiling of one `--workers 1` process doing CPU-bound Presidio/spaCy +
DistilBERT inference in a thread pool. At concurrency 10, that ceiling isn't
hit, so latency is what a single request actually costs: **p95 ≈ 86ms**. At
concurrency 100, the extra 90 requests queue behind the ~130/s ceiling
(Little's Law), which is why p95 balloons to ~913ms — that's *queueing delay*
on an overloaded single process, not per-request detection cost going up.

**What this means for the resume claim:** "p95 < 50ms" was written before
this was ever measured. The honest numbers are: **p95 ≈ 86ms per-request cost
at low concurrency**, and **~130 req/s is the ceiling of one worker process**.
Above that ceiling, add `--workers N` (one per core is a reasonable start —
the work is CPU-bound, not I/O-bound, so more workers should scale close to
linearly up to core count) rather than quoting a p95 that only holds below
the concurrency where queueing kicks in.
