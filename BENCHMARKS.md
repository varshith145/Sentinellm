# SentinelLM benchmarks

## Reproduce

```bash
python benchmarks/loadtest.py --url http://localhost:8000
```

Requires a running gateway (see README for startup). Deterministic by
construction: fixed payload set, fixed duration per level (20.0s),
payload order seeded with `--seed 42` (default 42) and round-robined
per worker — same inputs always produce the same request sequence. The
script refuses to run on a dirty working tree (`--allow-dirty` to override).
Every section below through "Eval: ensemble micro-F1" is tied to a commit
someone else can check out and reproduce exactly. The "Triton Inference
Server backend", "Dynamic INT8 quantization", and "Summary: tradeoff table"
sections further down were run with `--allow-dirty` against work still in
progress — see the SHA note at the start of the Triton section.

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

## Triton Inference Server backend (gRPC)

Git SHA at measurement time: `094d2ff` + uncommitted changes (the
`SENTINELLM_INFERENCE_BACKEND=triton` implementation — semantic.py,
config.py, main.py, gateway/requirements.txt — plus the INT8 quantization
work below: model/quantize_onnx.py, eval/run_eval.py's new CLI flags, the
sentinellm_int8 Triton model repo entry, and this section's test/benchmark
scripts). Not yet committed at time of writing; see repo state when reading
this.
Hardware: arm64, 10 cores, 16GB RAM, Darwin 25.6.0 (same machine as the rest
of this file).

`SENTINELLM_INFERENCE_BACKEND=triton` sends the same ONNX graph to a Triton
Inference Server over **gRPC** (`tritonclient.grpc`, not HTTP — see below for
why) via `gateway/app/detectors/semantic.py::_infer_triton`. Tokenization
still happens in-process; Triton only serves the model graph. Regex/Presidio
still run first and the orchestrator's confidence short-circuit
(`DetectionOrchestrator._is_conclusive`) still applies identically regardless
of backend — it operates on detector output, not on how the semantic
detector gets its logits — verified concretely in
[`tests/test_triton_backend.py`](tests/test_triton_backend.py), which runs
the same golden-path scenario as `tests/test_golden_path.py` (create a
BLOCK policy, trip it, check the audit record, check the Prometheus
counter) with the semantic model actually enabled and Triton-backed, and
additionally reads Triton's own `nv_inference_exec_count` metric
before/after the regex-conclusive request and asserts it didn't move.
`tests/test_golden_path.py` itself was also run directly against this
backend (`SENTINELLM_INFERENCE_BACKEND=triton python -m pytest
tests/test_golden_path.py`) — it passes, though it disables the semantic
model entirely, so that run only proves the option doesn't break startup;
`test_triton_backend.py` is what actually exercises the triton call path
end-to-end.

### HTTP vs gRPC client transport (direct to Triton, no gateway)

```bash
python triton_deploy/compare_http_grpc.py --n 200
```

Sequential single-request calls (no concurrency — isolates transport
overhead, not batching), same process, same machine, interleaved (HTTP
block then gRPC block, both after a warmup):

| Transport | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| HTTP | 35.800ms | 33.056ms | 61.468ms | 72.816ms |
| gRPC | 32.559ms | 30.812ms | 53.928ms | 60.022ms |

gRPC was ~1.10x faster on mean latency and noticeably tighter at the tail
(p99 72.8ms → 60.0ms) — the persistent HTTP/2 connection and protobuf
framing avoid a per-call handshake and text-header parse that the HTTP/REST
client pays every time. This is why the gateway's own triton backend uses
`tritonclient.grpc`, not `tritonclient.http`.

### Load test: torch (in-process) baseline vs Triton (gRPC)

**Correction:** this section originally published a torch/triton comparison
measured at ~110 req/s / ~56 req/s respectively, with a note concluding
"this machine was genuinely slower during this session" based on two
independent torch runs agreeing with each other at ~110 req/s. That
conclusion was wrong. A third torch run, minutes later, came back at 291
req/s — back in line with this file's other torch numbers. The actual cause
of the earlier two slow runs: `avconferenced` / `cameracaptured` /
`VTEncoderXPCService` — an active video call on this dev machine — were
consuming 50%+ CPU at the time, confirmed via `ps -Ao pid,pcpu,comm -r` and
gone by the third run. Two independent runs agreeing with each other is
*not* the same as two runs independent of a shared confound — both were
contaminated by the same still-running call. The numbers below are from
after that call ended, verified clean the same way (checked `ps` for
CPU hogs immediately before each run, and confirmed the torch number matched
this file's other sessions within 3%).

Three-way comparison this time — torch, Triton (fp32, dynamic_batching
configured), and Triton (INT8, dynamic_batching configured, see the
quantization section below) — all measured back-to-back in one continuous
session, gateway restarted between each, only `SENTINELLM_INFERENCE_BACKEND`
/ `SENTINELLM_TRITON_MODEL_NAME` changed. The Triton container (serving
both models) was left running for all three rows, including the torch one —
confirmed idle at 0.08% CPU when not serving a request, so this costs
nothing and removes "different number of resident processes" as a confound
between rows.

```bash
python triton_deploy/build_model_repo.py   # builds both sentinellm and sentinellm_int8
./triton_deploy/run.sh &                   # left running for all three rows below

SENTINELLM_INFERENCE_BACKEND=torch python benchmarks/loadtest.py --url http://localhost:8000 --concurrencies 10,50,100 --out /tmp/torch.md

SENTINELLM_INFERENCE_BACKEND=triton SENTINELLM_ONNX_MODEL_PATH=$(pwd)/model/onnx SENTINELLM_TRITON_URL=localhost:8101 SENTINELLM_TRITON_MODEL_NAME=sentinellm python benchmarks/loadtest.py --url http://localhost:8000 --concurrencies 10,50,100 --out /tmp/triton_fp32.md

SENTINELLM_INFERENCE_BACKEND=triton SENTINELLM_ONNX_MODEL_PATH=$(pwd)/model/onnx-int8 SENTINELLM_TRITON_URL=localhost:8101 SENTINELLM_TRITON_MODEL_NAME=sentinellm_int8 python benchmarks/loadtest.py --url http://localhost:8000 --concurrencies 10,50,100 --out /tmp/triton_int8.md
```

| Backend | Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|---|
| torch | 10 | 291.1 req/s | 24.31ms | 74.47ms | 85.07ms | 5866 | 0 |
| torch | 50 | 272.3 req/s | 177.18ms | 329.69ms | 394.12ms | 5483 | 0 |
| torch | 100 | 255.1 req/s | 355.47ms | 802.49ms | 1289.45ms | 5176 | 0 |
| triton (fp32) | 10 | 152.1 req/s | 51.24ms | 148.62ms | 180.68ms | 3048 | 0 |
| triton (fp32) | 50 | 146.0 req/s | 239.23ms | 892.3ms | 1277.89ms | 2989 | 0 |
| triton (fp32) | 100 | 144.0 req/s | 329.87ms | 2288.74ms | 2471.42ms | 3012 | 0 |
| triton (int8) | 10 | 268.9 req/s | 34.98ms | 67.89ms | 85.91ms | 5387 | 0 |
| triton (int8) | 50 | 251.2 req/s | 176.89ms | 419.05ms | 813.68ms | 5057 | 0 |
| triton (int8) | 100 | 225.7 req/s | 371.23ms | 1123.91ms | 1970.99ms | 4588 | 0 |

**Baseline p95 at concurrency 100 was 802.49ms (torch, in-process); Triton
fp32 (gRPC, dynamic batching configured) p95 at concurrency 100 is
2288.74ms — slower, not faster. Triton INT8 p95 at concurrency 100 is
1123.91ms — still behind torch, but recovers most of the gap: 2.85x worse
than torch at fp32, only 1.40x worse at int8.**

### Why: dynamic_batching barely engaged under real gateway traffic, at either precision

Triton's `dynamic_batching` can only combine requests that share the exact
same input shape. The gateway tokenizes each request unpadded, to its own
natural length (see `_infer_triton` / `_infer_onnx`) — so concurrent
requests almost never share a shape, and Triton executes them essentially
one at a time regardless of the `max_queue_delay_microseconds: 5000`
config, at either precision. Measured directly from Triton's own metrics
after the sweep above (cumulative since container start, one sweep per
model — no other traffic hit either model in between):

```bash
curl -s http://localhost:8102/metrics | grep -E "nv_inference_(exec_count|request_success)"
# nv_inference_request_success{model="sentinellm",version="1"}      4633
# nv_inference_exec_count{model="sentinellm",version="1"}           3629
# nv_inference_request_success{model="sentinellm_int8",version="1"} 7516
# nv_inference_exec_count{model="sentinellm_int8",version="1"}      6386
```

Average batch size: fp32 ≈ 4633/3629 ≈ **1.28**, int8 ≈ 7516/6386 ≈ **1.18**
— both barely above 1:1, nowhere near the ≈6.9–8.0 measured in
[`triton_deploy/verify_client.py`](triton_deploy/verify_client.py)'s probe,
which deliberately pads every request to the *same* fixed length before
firing them concurrently. That gap is the whole story for both rows:
`dynamic_batching` is correctly configured and demonstrably works
(verify_client.py proves it), but this workload — one always-different-length
text per request — doesn't hand it many same-shape requests to batch,
regardless of model precision. INT8's throughput recovery over fp32 comes
from the model itself being ~4x smaller and cheaper to run per call (see
quantization section below), not from batching engaging any better.

This isn't a Triton misconfiguration; it's a mismatch between Triton's
batching precondition (uniform shape) and this gateway's tokenization
strategy (unpadded, natural length) — already flagged as a caveat in
`triton_deploy/models/sentinellm/config.pbtxt` when the deployment was
first built. Making Triton a net win here would mean either padding
gateway requests to a fixed length before sending them (adding compute to
save on batching, a real tradeoff) or batching multiple *user* requests
together server-side before tokenizing — both out of scope for adding the
backend option itself.

## Dynamic INT8 quantization (`model/quantize_onnx.py`)

```bash
python model/export_onnx.py       # if model/onnx/ doesn't exist yet
python model/quantize_onnx.py     # writes model/onnx-int8/
```

Uses `optimum.onnxruntime.ORTQuantizer` with the `arm64` preset
(`AutoQuantizationConfig.arm64(is_static=False, per_channel=False)`) —
`avx512_vnni` is x86-only and doesn't apply to this hardware. "Dynamic"
means weights are quantized to INT8 offline by this script; activation
ranges are computed per-inference at runtime, so no calibration dataset is
needed (unlike static quantization). Model size: **253.3 MB → 63.7 MB**
(~4.0x smaller).

### F1 delta — recorded honestly

```bash
python eval/run_eval.py --backend onnx --onnx-model-path model/onnx-int8 --out eval/results_int8.md
```

Same held-out test split, same three-pass pipeline (regex + Presidio +
semantic), only the semantic model's weights changed:

| Backend | Precision | Micro-F1 | Precision | Recall |
|---|---|---|---|---|
| torch / onnx / triton(fp32) | FP32 | 0.9574 | 1.0 | 0.9184 |
| onnx / triton(int8) | INT8 (dynamic) | **0.9462** | 1.0 | 0.898 |

F1 dropped **0.0112** (−1.2% relative), entirely from recall — precision
held at a perfect 1.0 in both cases, so quantization didn't introduce any
*new* false positives on this test set, it just missed one true positive it
previously caught (26 PII true positives vs 27; see
[`eval/results_int8.md`](eval/results_int8.md) for the full confusion
matrix). Small test set (49 examples) — a single missed span is what a 1.2%
F1 move looks like here, not evidence of a stable-at-scale number, but the
direction (precision preserved, recall softened) is the expected signature
of aggressive weight quantization and matches what dynamic INT8 quantization
generally does to transformer classifiers.

### Load test: ONNX Runtime in-process, fp32 vs INT8

```bash
SENTINELLM_INFERENCE_BACKEND=onnx SENTINELLM_ONNX_MODEL_PATH=$(pwd)/model/onnx python benchmarks/loadtest.py --url http://localhost:8000 --concurrencies 10,50,100 --out /tmp/onnx_fp32.md

SENTINELLM_INFERENCE_BACKEND=onnx SENTINELLM_ONNX_MODEL_PATH=$(pwd)/model/onnx-int8 python benchmarks/loadtest.py --url http://localhost:8000 --concurrencies 10,50,100 --out /tmp/onnx_int8.md
```

Same session as the torch/triton rows above (checked clean before each
run — see the correction note there for why that check matters).

| Backend | Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|---|
| onnx (fp32) | 10 | 506.0 req/s | 18.57ms | 35.76ms | 40.12ms | 10133 | 0 |
| onnx (fp32) | 50 | 446.2 req/s | 80.14ms | 234.73ms | 345.37ms | 8966 | 0 |
| onnx (fp32) | 100 | 372.7 req/s | 197.13ms | 774.76ms | 1244.66ms | 7521 | 0 |
| onnx (int8) | 10 | 615.1 req/s | 15.84ms | 23.57ms | 26.42ms | 12310 | 0 |
| onnx (int8) | 50 | 421.9 req/s\* | 75.91ms | 370.36ms | 570.29ms | 8478 | 0 |
| onnx (int8) | 100 | 516.2 req/s | 162.02ms | 478.89ms | 781.57ms | 10395 | 0 |

\* **Confirmed noisy, not replaced.** The int8 c=50 throughput dipping below
both its own c=10 and c=100 neighbors was originally suspected to be a
background file-sync spike (`FPCKService`, ~98% CPU for ~20s right before
this sweep) bleeding into the run. Re-ran c=50 alone afterward, machine
confirmed clean via `ps` immediately before (no CPU hogs, load average
1.64) — and it came back at a **third** number, 300.4 req/s, p95 318.67ms
(lower throughput than the original, not higher) — still below the c=100
row, i.e. still non-monotonic. Two independent readings under confirmed-clean
conditions that don't agree with each other means this isn't residual
contamination from that file-sync spike; c=50 is just noisy for this
specific backend (the fastest of the five measured, so plausibly the one
most sensitive to something concurrency-scheduling-related at that exact
load level — not chased further). Per instruction, this is annotated as
confirmed-noisy rather than silently replaced with whichever rerun looked
nicer: **421.9 req/s (original) and 300.4 req/s (rerun) are both real
measurements that disagree; treat onnx-int8's c=50 cell as ±30% uncertain,
c=10 and c=100 for this row as reliable.**

## Summary: tradeoff table

The task that produced this section originally asked for four rows
(PyTorch / ONNX / Triton+batching / Triton+INT8) with the ONNX row reused
from an earlier session's paired torch/onnx measurement, on the reasoning
that the two sessions' torch numbers agreed within 8%. That reasoning was
directly undercut a few paragraphs up in this same file — two independently
agreeing torch runs earlier in *this* session were both silently
contaminated by a video call, and agreement turned out not to imply a clean
measurement. So the ONNX row was re-measured on today's machine instead of
reused, and a fifth row (ONNX INT8, in-process — the "natural next
comparison" this section originally deferred) was added in the same pass,
since it was one gateway restart away and turned out to matter. Every row
below is from today, one continuous session, CPU checked clean via
`ps -Ao pid,pcpu,comm -r` immediately before each run:

| Backend | Micro-F1 | p95 @ c=100 | req/s @ c=100 |
|---|---|---|---|
| PyTorch (baseline) | 0.9574 | 802ms | 255.1 |
| ONNX Runtime, fp32 (in-process) | 0.9574 | 775ms | 372.7 |
| ONNX Runtime, INT8 (in-process) | 0.9462 | 479ms | 516.2 |
| Triton + dynamic_batching, fp32 | 0.9574 | 2289ms | 144.0 |
| Triton + dynamic_batching, INT8 | 0.9462 | 1124ms | 225.7 |

No single row wins on both axes, but one is close:

- **ONNX Runtime INT8, in-process, is the best measured option overall** —
  highest throughput of anything tested (516.2 req/s, 2.0x torch), lowest
  p95 (479ms), no extra infrastructure, at the same honestly-measured
  −1.2% F1 cost quantization carries everywhere else in this file. If
  accuracy at the third decimal place isn't the constraint, this is the
  answer.
- **ONNX Runtime fp32, in-process, is the right choice if INT8's accuracy
  cost is unacceptable** — full precision, still 1.46x torch's throughput,
  zero extra infrastructure.
- **Triton (either precision) is a throughput loss on this specific
  workload** — same accuracy as its in-process counterpart at each
  precision, but both Triton rows are slower than the corresponding
  in-process ONNX row, because this workload structurally can't hand
  `dynamic_batching` anything to batch (see above) while still paying a
  gRPC hop and a second process's CPU. Only justified by reasons other than
  raw latency: decoupling the inference tier from the gateway tier, GPU
  deployment, multi-model serving.
- **INT8 is worth it within either deployment shape** — it improves
  throughput over fp32 whether in-process (372.7 → 516.2 req/s, +38%) or
  through Triton (144.0 → 225.7 req/s, +57%), for the same fixed F1 cost
  either way. The quantized graph itself is what's doing the work here, not
  which serving path it's reached through.
- **Direct per-example parity, in-process ONNX INT8 vs Triton INT8**: now
  checked (`tests/test_triton_backend.py::test_triton_matches_onnx[int8]`,
  same 50 fixed inputs as `tests/test_onnx_parity.py`, parametrized
  alongside the existing fp32 check) — not just inferred from the two
  aggregate F1 numbers agreeing. Result: **49/50 exact match, 1 genuine
  deterministic disagreement** (reproduced 3x, not flaky), traced to ONNX
  Runtime version skew — the local pip install is 1.28.0, Triton 24.09
  bundles 1.19.2 (`libonnxruntime.so.1.19.2` inside the container). Max abs
  logit diff on the disagreeing example: 0.988, on a borderline case the
  in-process path calls SECRET at confidence 0.9633 and Triton doesn't call
  at all — consistent with INT8 requantization being more sensitive to
  small cross-version fp32 accumulation differences than fp32 inference is
  (a rounding-boundary effect, not random noise). The fp32 parity check, run
  through the exact same two ORT versions, has zero mismatches on the same
  50 inputs — isolating this to an INT8-specific interaction, not a general
  cross-version inference difference. Marked `xfail(strict=True)` in the
  test so it's visible in test output rather than either silently passing
  or silently blocking the suite, and so an accidental fix (e.g. a future
  Triton image with a newer bundled ORT) gets flagged instead of the marker
  going stale. Practical takeaway: aggregate F1 agreement was directionally
  right here but is not a substitute for per-example checking — this is
  exactly the failure mode per-example parity testing exists to catch, and
  it caught one, even though it's small enough (1/50, on a single
  near-threshold example) not to change the eval F1 numbers reported above.
