# SentinelLM benchmarks

## Reproduce

```bash
python benchmarks/loadtest.py --url http://localhost:8000
```

Requires a running gateway (see README for startup). Deterministic by
construction: fixed payload set, fixed duration per level (20.0s),
payload order seeded with `--seed 42` (default 42) and round-robined
per worker — same inputs always produce the same request sequence. The
script refuses to run on a dirty working tree (`--allow-dirty` to override)
so every number below is tied to a commit someone else can check out and
reproduce exactly.

## Gateway config at time of run

| Setting | Value |
|---|---|
| Workers | 4 (uvicorn, per gateway/Dockerfile default) |
| Active detectors | regex, presidio, semantic |
| Policy ID | default-v1 |
| DB backend | sqlite (aiosqlite) |

## Load test: `/scan` (detection pipeline, no LLM in the path)

Two independent runs, same machine, same clean-tree requirement — kept as
two rows rather than overwritten, since the spread between them is itself
the signal: this pipeline's tail latency has real run-to-run variance under
load, not just backend-to-backend variance (see the ONNX section below,
which hit the same thing).

**Run 1** — generated 2026-08-10 16:49 UTC, SHA `ffc7bbc` (dirty tree at the
time; superseded by run 2's clean SHA below, kept for the variance record):

| Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|
| 1 | 139.8 req/s | 9.81ms | 11.3ms | 11.92ms | 2796 | 0 |
| 10 | 259.1 req/s | 32.32ms | 78.73ms | 87.6ms | 5191 | 0 |
| 50 | 267.0 req/s | 152.32ms | 432.08ms | 674.66ms | 5389 | 0 |
| 100 | 269.0 req/s | 348.98ms | 636.5ms | 769.52ms | 5459 | 0 |

**Run 2 (clean tree)** — generated 2026-08-11 00:15 UTC, SHA `9436851`
(clean — `git status --porcelain` empty, verified by the script itself
before running):

| Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|
| 1 | 140.6 req/s | 9.65ms | 11.5ms | 12.59ms | 2812 | 0 |
| 10 | 290.5 req/s | 27.19ms | 80.35ms | 90.23ms | 5816 | 0 |
| 50 | 260.7 req/s | 174.58ms | 386.66ms | 718.45ms | 5253 | 0 |
| 100 | 265.9 req/s | 346.4ms | 707.92ms | 1246.92ms | 5420 | 0 |

Throughput at c=1/10/50/100 agrees within ~10% across the two runs; p99 at
c=50/100 swings more (674ms→718ms, 769ms→1247ms) — same pattern already
seen in the ONNX section's paired runs, on the same shared dev machine.
Treat any single p99 tail number in this file as ±30-40% noise at
concurrency ≥50 unless it's from a paired same-session comparison (like the
ONNX delta below); throughput and p50/p95 are the stable numbers.

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

## ONNX backend: PyTorch vs ONNX Runtime

Git SHA at measurement time: `e85181f` + uncommitted changes — those
changes were committed unmodified immediately afterward as `9b3ed79`, so
that's the SHA that actually reproduces this section (now further
superseded by `9436851`, which only touched `.gitignore` and
`benchmarks/loadtest.py`'s dirty-tree guard — no semantic-detector code
changed after `9b3ed79`).
Hardware: arm64, 10 cores, 16GB RAM, Darwin 25.6.0 (same machine as the
headline table above, but **not the same run** — see note below)

`SENTINELLM_INFERENCE_BACKEND` now selects the semantic detector's inference
path: `torch` (default, unchanged) or `onnx`, which runs the graph exported
by `model/export_onnx.py` through `onnxruntime.InferenceSession` —
`intra_op_num_threads` set explicitly to 1, same rationale as the existing
`torch.set_num_threads(1)`: each request already gets its own thread-pool
worker, so leaving ONNX Runtime's own intra-op parallelism at its default
(every core, per call) would have concurrent requests thrash instead of
scale. Both backends run the same BIO-decode logic (`SemanticDetector._decode`)
so this isolates the model execution cost specifically — see
[`tests/test_onnx_parity.py`](tests/test_onnx_parity.py) for the numeric
proof the two backends agree (max abs logit diff ~1.3e-05), and
`eval/run_eval.py` re-run after the `_decode` refactor to confirm the torch
path's own output didn't move (still micro-F1 0.9574, byte-identical
`eval/results.md`).

**Paired-run note:** the headline table earlier in this file was captured
~7 hours before this section, a different process, at a different point in
this machine's uptime — not a fair baseline for a 2x claim. So both rows
below come from two back-to-back gateway restarts run minutes apart in the
same session (torch: 2026-08-10 23:49 UTC, onnx: 2026-08-10 23:43 UTC),
identical config each time (4 uvicorn workers, sqlite, all three detectors
active) — only `SENTINELLM_INFERENCE_BACKEND` changed between them.

Reproduce (requires `python model/export_onnx.py` first — no ONNX build is
published to the Hub, only PyTorch):

```bash
SENTINELLM_INFERENCE_BACKEND=torch python benchmarks/loadtest.py --url http://localhost:8000 --out /tmp/torch.md
SENTINELLM_INFERENCE_BACKEND=onnx python benchmarks/loadtest.py --url http://localhost:8000 --out /tmp/onnx.md
```

(Restart the gateway between the two so the backend env var actually takes
effect — it's read once at startup.)

| Backend | Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|---|
| torch | 1 | 138.3 req/s | 9.82ms | 11.5ms | 12.17ms | 2768 | 0 |
| torch | 10 | 274.4 req/s | 31.5ms | 79.53ms | 90.12ms | 5496 | 0 |
| torch | 50 | 261.1 req/s | 186.86ms | 327.93ms | 381.24ms | 5263 | 0 |
| torch | 100 | 247.8 req/s | 377.82ms | 738.6ms | 1352.06ms | 5042 | 0 |
| onnx | 1 | 173.5 req/s | 7.04ms | 8.82ms | 9.64ms | 3471 | 0 |
| onnx | 10 | 499.9 req/s | 17.96ms | 36.28ms | 40.44ms | 10011 | 0 |
| onnx | 50 | 468.7 req/s | 58.35ms | 222.02ms | 305.02ms | 9427 | 0 |
| onnx | 100 | 511.9 req/s | 188.27ms | 252.78ms | 303.88ms | 10315 | 0 |

### Latency delta (ONNX vs PyTorch, paired runs above)

| Concurrency | Throughput (torch → onnx) | Speedup | p99 (torch → onnx) | p99 reduction |
|---|---|---|---|---|
| 1 | 138.3 → 173.5 req/s | 1.25x | 12.17ms → 9.64ms | -21% |
| 10 | 274.4 → 499.9 req/s | 1.82x | 90.12ms → 40.44ms | -55% |
| 50 | 261.1 → 468.7 req/s | 1.80x | 381.24ms → 305.02ms | -20% |
| 100 | 247.8 → 511.9 req/s | 2.07x | 1352.06ms → 303.88ms | -78% |

ONNX Runtime is consistently faster at every concurrency level tested here —
1.25x to 2.1x throughput, with the biggest tail-latency win at c=100 (torch's
p99 blew out to 1.35s under load in this run; onnx held at 304ms). The c=100
torch p99 is higher here than in the headline table's original torch run
(769ms) — normal run-to-run variance on a shared dev machine, not a
regression; the point of the paired setup is that both rows in this section
came from back-to-back runs, so the *delta* between them is meaningful even
though the absolute torch number moved between sessions.
