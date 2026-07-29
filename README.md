---
title: SentinelLM
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SentinelLM 🛡️

> **A self-hosted AI gateway that intercepts LLM traffic, detects PII and secrets across three detection passes, enforces configurable policies, and logs every decision — without sending your data anywhere.**

### 🔴 [**Live demo →**](https://huggingface.co/spaces/varshith145/sentinellm) &nbsp;paste text and watch detection run (direct app: https://varshith145-sentinellm.hf.space)

The public demo runs the **detection pipeline only** (demo mode — LLM proxy disabled). Try the obfuscated example `reach me at john dot smith at company dot com` to see the fine-tuned model catch what regex can't.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/DistilBERT-NER%20F1%3D0.849-orange?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-298%20passed-brightgreen?logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## The Problem

Regex catches `john@example.com`. It does not catch `"reach me at john dot smith at company dot com"`.

Organizations using LLMs have three unsolved risks:

- **Employees paste PII into prompts** — emails, SSNs, credit cards, passwords
- **Obfuscated data bypasses scanners** — spoken phone numbers, spelled-out SSNs, phonetic API keys
- **No audit trail** — security teams have no visibility into what gets sent to the model

SentinelLM solves all three with a layered detection pipeline, configurable policy engine, and tamper-evident audit log.

---

## How It Works

SentinelLM sits between your application and any LLM backend (Ollama, OpenAI-compatible). Every prompt and every response passes through a three-pass detection pipeline before anything reaches the model.

```
  App / Client
       │
       │  POST /v1/chat/completions  ← OpenAI-compatible
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    SentinelLM Gateway                        │
│                                                              │
│  ┌───────────────────── Input Scan ──────────────────────┐   │
│  │                                                       │   │
│  │  Pass 1 · Regex          Pass 2 · Presidio            │   │
│  │  < 1ms                   ~ 30ms                       │   │
│  │  Emails, SSNs, cards     Names, addresses             │   │
│  │  AWS keys, JWTs          Contextual PII               │   │
│  │  GitHub tokens                                        │   │
│  │                          Pass 3 · Semantic NER        │   │
│  │                          ~ 150ms                      │   │
│  │  ─────── asyncio.gather ─ Fine-tuned DistilBERT ───── │   │
│  │                                                       │   │
│  │              Detection Orchestrator                   │   │
│  │         Merge findings · Deduplicate overlaps         │   │
│  │         Prefer highest confidence per span            │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│               ┌─────────▼──────────┐                        │
│               │   Policy Engine    │   ← DB (seeded once     │
│               │  ALLOW / MASK / BLOCK    from default.yaml)  │
│               └─────────┬──────────┘                        │
│                         │                                    │
│           ┌─────────────┴──────────────┐                    │
│       MASK/ALLOW                    BLOCK                    │
│     Redact → LLM                 Return 403                  │
│           │                      (LLM never sees it)        │
│     Output Scan                                              │
│     Audit Log → PostgreSQL                                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
  Streamlit Admin Dashboard  (http://localhost:8501)
```

---

## What the Semantic Pass Catches

This is the differentiator. The fine-tuned DistilBERT model catches obfuscated PII that regex and Presidio are completely blind to:

| Input | Detector | Decision |
|-------|----------|----------|
| `john@example.com` | Regex | MASK |
| `reach me at john dot smith at company dot com` | **Semantic** | MASK |
| `my social is four five six dash seven eight dash nine zero one two` | **Semantic** | MASK |
| `AKIAIOSFODNN7EXAMPLE` | Regex | BLOCK |
| `the secret key is hunter two dont tell anyone` | **Semantic** | BLOCK |
| `my password for the server is the name of my dog followed by the year i was born` | **Semantic** | BLOCK |
| `we use AWS for cloud hosting and S3 for storage` | — | ALLOW ✓ |

---

## Detection Pipeline Details

| Pass | Engine | Latency | Entity Types |
|------|--------|---------|--------------|
| **1 · Regex** | Compiled patterns + Luhn validation | < 1ms | EMAIL, PHONE, SSN, CREDIT_CARD, AWS_KEY, GITHUB_TOKEN, JWT |
| **2 · Presidio** | Microsoft NLP + spaCy `en_core_web_lg` | ~30ms | PERSON_NAME, contextual PII, addresses |
| **3 · Semantic NER** | Fine-tuned DistilBERT (BIO tagging) | ~150ms | GENERIC_PII, GENERIC_SECRET — obfuscated and informal |

Regex and Presidio run in parallel via `asyncio.gather`. The semantic pass — by far the most expensive — only runs when those two didn't already reach a confident conclusion (see Design Notes); when it does run, it's the same `asyncio.gather` step. The orchestrator then merges results, deduplicates overlapping spans (keeping highest confidence), and applies detector priority: `semantic > presidio > regex` on ties.

---

## Model Performance

The semantic NER model is trained on 410+ synthetic obfuscated examples with 120 hard negatives, using a custom `WeightedLossTrainer` to correct class imbalance:

| Metric | Score |
|--------|-------|
| **F1** | **0.849** |
| Precision | 0.789 |
| Recall | 0.918 |
| Test examples | 61 |

Class weights `[O=0.3, B-PII=10.0, I-PII=10.0, B-SECRET=10.0, I-SECRET=10.0]` ensure the model doesn't collapse to predicting all-`O` on the heavily imbalanced label distribution. Confidence scores are derived from real softmax probabilities per token span — genuine secrets score ~1.00, uncertain detections (e.g. cloud service names) score ~0.85, enabling clean threshold separation at 0.90.

The number above is the *raw token classifier* in isolation (`model/evaluate.py`, `docs/model_eval.json`). The number that actually ships is the full three-pass pipeline — see below.

---

## Measured Performance (reproducible — every number below is one command)

**Detection pipeline, full stack (regex + Presidio + semantic, deduplicated), same 61-example held-out split as above, IoU ≥ 0.5 span matching:**

```bash
python eval/run_eval.py
```

| Category | Precision | Recall | F1 |
|---|---|---|---|
| PII | 1.0000 | 0.9643 | 0.9818 |
| SECRET | 1.0000 | 0.8571 | 0.9231 |
| **Micro-average** | **1.0000** | **0.9184** | **0.9574** |

Higher than the raw model's 0.849 F1 — expected, since regex and Presidio also
run and the eval is span-overlap-based rather than strict per-token BIO
matching. Precision=1.0000 was checked, not assumed: it's identical under
any-overlap, IoU≥0.3, and IoU≥0.5 matching (spans average IoU=0.91 — tight
matches, not degenerate wide-span luck), and the 12 hard-negative examples in
the split produced zero false positives. Full methodology and the robustness
check in [`eval/results.md`](eval/results.md).

**Load test, `/scan` (detection only, no LLM in the path), 4 `uvicorn`
worker processes — the same config `gateway/Dockerfile` runs in production
(docker-compose's Postgres stack; the HF Spaces demo is a separate,
single-container SQLite deployment — see Design Notes), Apple M4:**

```bash
python bench/load_test.py --url http://localhost:8000 --concurrency 100 --duration 60
```

| Concurrency | Throughput | p50 | p95 | p99 |
|---|---|---|---|---|
| 10 | 238.8 req/s | 34.69ms | 87.21ms | 98.87ms |
| 100 | 258.5 req/s | 362.54ms | 732.37ms | 989.32ms |

That's 2.04x the throughput and 20% lower p95 than a single `--workers 1`
process running the original unconditional 3-pass pipeline (127.2 req/s,
913.53ms p95) — from two changes: the orchestrator now skips the ~150ms
semantic pass once regex already gives a confident answer, and 4 worker
processes spread whatever's left across cores. See
[`bench/results.md`](bench/results.md) for the full investigation — what
was tried, measured, and ruled out before landing here — and Design Notes
below for why the accuracy cost of the short-circuit had to be checked
against `eval/run_eval.py`, not assumed.

---

## API Response

SentinelLM is a drop-in OpenAI-compatible proxy. Every response includes a `ppg` metadata block with the full decision trace:

```json
{
  "choices": [...],
  "ppg": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "input_decision": "MASK",
    "output_decision": "ALLOW",
    "input_redactions": { "EMAIL": 1 },
    "output_redactions": {},
    "policy_id": "default-v1",
    "detectors_used": ["regex", "presidio", "semantic"],
    "latency_ms": {
      "detection": 162,
      "llm": 2341,
      "total": 2514
    }
  }
}
```

When a request is blocked:

```json
{
  "error": {
    "message": "Request blocked by SentinelLM policy",
    "type": "policy_violation",
    "reasons": [
      "BLOCK: AWS_KEY detected (confidence=0.95, detector=regex)"
    ]
  },
  "ppg": {
    "input_decision": "BLOCK",
    ...
  }
}
```

---

## Console API

A read/write REST surface under `/api/v1/` — separate from the proxied
`/v1/*` traffic — for authoring policy, searching the audit log, and reading
metrics. This is the backend the [React/TS console](#console-react--typescript)
below is built against — every endpoint here is also covered by
`tests/test_console_api.py`.

| Endpoint | What it does |
|---|---|
| `GET/POST /api/v1/policies` | List / create policy rules |
| `GET/PATCH/DELETE /api/v1/policies/{id}` | Read, edit, or remove one rule |
| `POST /api/v1/policies/{id}/dry-run` | Test a rule against sample text — no audit record written |
| `GET /api/v1/audit` | Search audit records: time range, decision, entity type, free text — cursor-paginated |
| `GET /api/v1/audit/{id}` | Full record for one request |
| `GET /api/v1/stats/summary` | Totals, decision breakdown, p50/p95/p99, top triggered entity types |
| `GET /api/v1/stats/timeseries` | Same, bucketed over a time window |

Auth: `X-API-Key` header, checked against `SENTINELLM_CONSOLE_API_KEY` — empty
(the default) disables the check for local dev. CORS is an explicit allowlist
via `SENTINELLM_CONSOLE_CORS_ORIGINS`, never `*`.

**Flip a policy at runtime and watch it take effect immediately:**

```bash
# EMAIL masks by default — flip it to BLOCK
curl -s http://localhost:8000/api/v1/policies | python3 -m json.tool  # find the EMAIL rule's id
curl -s -X PATCH http://localhost:8000/api/v1/policies/{id} \
  -H "Content-Type: application/json" -d '{"action": "block"}'

curl -s -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"text": "contact me at a@b.com"}'
# decision is now BLOCK — no restart, no redeploy
```

The `policies` table is the source of truth at runtime.
`gateway/policies/default.yaml` is only read once, to seed that table if it's
empty — it is **not** hot-reloaded the way it was before this API existed
(see Design Notes below for why that tradeoff was made deliberately).

---

## Console (React + TypeScript)

`console/` — Vite + React 18 + TypeScript (strict) + Tailwind + TanStack
Query + React Router + Recharts, talking to the Console API above through a
client generated by `openapi-typescript`/`openapi-fetch` (no hand-written
API types to drift from the backend).

Its visual identity, **"The Instrument Room,"** is a purpose-built system
(not a default UI kit reskin): a seismograph/polygraph recording-desk
metaphor — warm paper ground, ink-black structure, exactly two severity
inks (vermillion for BLOCK, ochre for MASK — never used as decoration),
IBM Plex Mono for gateway-produced data vs. system sans for UI chrome, and
a hand-stamped decision badge as the one deliberately irregular shape in an
otherwise rectilinear, flat-by-construction system. Full rationale and
token values in [`DESIGN.md`](DESIGN.md).

Three pages:

- **`/policies`** — table, create form, optimistic enable/disable toggle,
  delete with confirm, and a dry-run panel (paste text, see the decision,
  findings, and redacted output inline, no audit record written).
- **`/audit`** — filters (time range, decision, entity type, free text)
  synced to the URL query string, cursor-paginated ("Load more" against the
  API's real `next_cursor`, not an offset hack), row click opens a detail
  drawer with the full record.
- **`/metrics`** — 4 cards (total requests, block rate, p95 latency, top
  entity type), a stacked decisions-over-time chart, a p95-latency-over-time
  chart, a 1h/24h/7d window selector, 10s polling that pauses when the tab
  is hidden.

11 Vitest + React Testing Library tests (MSW-mocked, no backend needed —
`npm run test`), `tsc --noEmit --strict` clean, `npm run lint` clean.

### Local dev

```bash
cd console
npm install
npm run dev          # http://localhost:5173, talks to VITE_API_BASE_URL (.env.local, defaults to :8000)
```

Needs the gateway running (`docker compose up gateway db` or
`uvicorn app.main:app` from `gateway/`) for anything beyond the empty-state
screens. Regenerate the typed API client after changing any `/api/v1/*`
response shape:

```bash
npm run generate:types    # hits a running gateway's /openapi.json
```

### Deploying (Vercel)

Not yet deployed as of this writing — these are the steps, not a claim
that a live link exists:

1. New Vercel project, **Root Directory = `console/`** (this is a
   monorepo; the repo root has no `package.json`).
2. Set `VITE_API_BASE_URL` in the Vercel dashboard. `console/.env.production`
   already points it at the HF Spaces demo (`https://varshith145-sentinellm.hf.space`),
   used automatically on `vercel --prod` builds.
3. `console/vercel.json` has the SPA rewrite (`/(.*)` → `/index.html`) React
   Router needs — without it, a direct link to `/audit` 404s on Vercel's
   static file routing.
4. On the gateway side (HF Spaces repository variables, not a file this repo
   controls — see `HF_DEPLOY_HANDOFF.md`): set
   `SENTINELLM_CONSOLE_READ_ONLY=true` (writes 403, reads and dry-run stay
   open — see the read-only guard in `console_api.py`) and
   `SENTINELLM_CONSOLE_CORS_ORIGINS` to the Vercel URL once it exists.

No Vercel-side secrets are needed — the read-only guard lives server-side on
HF Spaces, so there's nothing sensitive in the public frontend bundle.

---

## Quickstart

**Prerequisites:** Docker + Docker Compose v2, [Ollama](https://ollama.ai/) running locally.

```bash
# 1. Clone
git clone https://github.com/varshith145/sentinellm.git
cd sentinellm

# 2. Pull a small model into Ollama (on your host machine)
ollama pull qwen2.5:0.5b

# 3. Start the full stack
docker compose up --build
```

| Service | URL |
|---------|-----|
| Gateway (OpenAI-compatible) | http://localhost:8000 |
| Console API + docs | http://localhost:8000/docs |
| Admin Dashboard (Streamlit) | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana (5-panel dashboard, provisioned) | http://localhost:3001 |
| Jaeger (traces) | http://localhost:16686 |

**Test PII masking:**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "Summarize this note from john@acme.com about the Alpha project."}]
  }' | python3 -m json.tool
```

`john@acme.com` is replaced with `[REDACTED_EMAIL]` before reaching the model.

**Test secret blocking:**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "Use key AKIAIOSFODNN7EXAMPLE to access S3"}]
  }' | python3 -m json.tool
```

Returns 403. The LLM never sees the key.

---

## Observability

`docker compose up` also starts Prometheus, Grafana, and Jaeger — zero
manual setup, all provisioned from files committed in this repo
(`prometheus/`, `grafana/provisioning/`, `grafana/dashboards/sentinellm.json`).

**Grafana** (http://localhost:3001, anonymous viewer access) ships one
dashboard, five panels:

1. Request rate, by decision
2. Request latency p50 / p95 / p99 (`histogram_quantile` over `sentinellm_request_latency_secs`)
3. Decision breakdown, stacked over time
4. Top 10 triggered entity types
5. Detector confidence distribution (heatmap, `sentinellm_detection_confidence`)

**Jaeger** (http://localhost:16686) shows one trace per request with child
spans for `detection_pipeline`, `policy_evaluation`, and `audit_write` — the
full path a request takes, not just one opaque HTTP span. Off by default
(`SENTINELLM_OTEL_ENABLED=false`) so a plain `uvicorn` run or `pytest` never
spends time retrying a collector that isn't there; docker-compose turns it on.

---

## Policy Configuration

`gateway/policies/default.yaml` defines the starting rule set. It's read
**once**, at first boot, to seed the `policies` DB table if it's empty —
after that, the DB is authoritative and edits go through the
[Console API](#console-api) (`PATCH /api/v1/policies/{id}`), not this file.
This is what makes runtime policy authoring possible at all: a file can't be
edited by a running service without a restart, a DB row can.

```yaml
rules:
  - entity_type: EMAIL
    action: MASK
    min_confidence: 0.7

  - entity_type: AWS_KEY
    action: BLOCK
    min_confidence: 0.5        # Block at any reasonable confidence

  - entity_type: GENERIC_SECRET
    action: BLOCK
    min_confidence: 0.90       # Calibrated: genuine secrets score ~1.00
                               # Uncertain matches (cloud service names) ~0.85

default_action: ALLOW

output_scanning:
  enabled: true
  secret_action: MASK          # Don't block the response, just redact
```

**Actions:**
- `ALLOW` — pass through unchanged
- `MASK` — replace the matched span with a typed token (`[REDACTED_EMAIL]`, `[REDACTED_SSN]`, etc.)
- `BLOCK` — return 403 immediately, LLM never invoked

**Decision priority:** `BLOCK > MASK > ALLOW`. If any finding triggers BLOCK, the entire request is blocked.

---

## Training the Semantic Model

The semantic model is optional (the gateway degrades gracefully to regex + Presidio without it) but significantly improves obfuscation detection.

```bash
# One command — builds a training container, trains, evaluates, saves to model/trained/
bash train.sh
```

Or step by step:

```bash
# 1. Download base model (DistilBERT)
python3 model/download_base_model.py

# 2. Generate synthetic training data
python3 model/data/generate_training_data.py

# 3. Prepare tokenized dataset
python3 model/data/prepare_dataset.py

# 4. Train (12 epochs, weighted loss, lr=3e-5)
python3 model/train.py

# 5. Evaluate on held-out test set
python3 model/evaluate.py
```

The trained model is automatically mounted into the gateway container via Docker volume — no rebuild required after retraining.

---

## Test Suite

```bash
cd gateway && python3 -m pytest ../tests/ -v
```

```
287 passed in 22.18s
```

| File | Tests | Coverage |
|------|-------|---------|
| `test_regex.py` | 82 | Luhn algorithm, all 7 regex patterns, offsets, false-positive guards |
| `test_policy.py` | 59 | All entity types, every confidence threshold boundary, output scanning |
| `test_redact.py` | 31 | All 11 entity types + tokens, positions, adjacency, count aggregation |
| `test_orchestrator.py` | 28 | Deduplication, overlap handling, detector priority, failure resilience, semantic short-circuit |
| `test_integration.py` | 20 | Full pipeline: detect → evaluate → redact, end-to-end per scenario |
| `test_streaming.py` | 29 | SSE chunk assembly, redacted re-emission, clean pass-through, error paths |
| `test_console_api.py` | 28 | Policy CRUD, dry-run, live-reload, audit filters + cursor pagination, stats |
| `test_scan.py` | 6 | `/scan` endpoint, demo-mode LLM gating, demo page |
| `test_sqlite_fallback.py` | 3 | DB URL defaults to SQLite when unset |
| `test_golden_path.py` | 1 | Console creates a BLOCK policy → gateway request trips it → audit view shows it → metric increments |

Counted on Python 3.12 with the semantic detector disabled, matching CI's
`SENTINELLM_SEMANTIC_MODEL_ENABLED=false` job. Plus 11 Vitest/RTL tests for
the console (see [Console](#console-react--typescript) above) — **298
tests total**, which is what the badge at the top counts.

---

## Project Structure

```
SentinelLM/
├── gateway/                         # FastAPI gateway
│   ├── app/
│   │   ├── main.py                  # Entrypoint + request pipeline (streaming + non-streaming)
│   │   ├── config.py                # Env-based settings (pydantic-settings)
│   │   ├── policy.py                # DB-backed policy engine (seeded from YAML once)
│   │   ├── console_api.py           # /api/v1/* — policy CRUD, dry-run, audit search, stats
│   │   ├── console_models.py        # Pydantic models for the console API
│   │   ├── redact.py                # Typed-token redaction
│   │   ├── audit.py                 # Async audit log writer
│   │   ├── db.py                    # SQLAlchemy async — AuditLog + Policy tables
│   │   ├── metrics.py               # Prometheus counters, histograms, gauges
│   │   ├── tracing.py               # OpenTelemetry setup + shared tracer
│   │   └── detectors/
│   │       ├── base.py              # EntityType, Finding, BaseDetector
│   │       ├── regex.py             # Pass 1: compiled patterns + Luhn
│   │       ├── presidio_detector.py # Pass 2: Microsoft Presidio
│   │       ├── semantic.py          # Pass 3: DistilBERT NER
│   │       └── orchestrator.py      # asyncio.gather + deduplication
│   └── policies/
│       └── default.yaml             # Seed-only: read once if `policies` table is empty
├── model/
│   ├── data/
│   │   ├── generate_training_data.py # 410+ synthetic obfuscated examples
│   │   ├── prepare_dataset.py        # Tokenize + BIO label alignment
│   │   ├── synthetic_obfuscated.jsonl
│   │   └── hard_negatives.jsonl
│   ├── train.py                      # WeightedLossTrainer, 12 epochs
│   └── evaluate.py                   # seqeval F1/precision/recall — raw model only
├── eval/
│   └── run_eval.py                   # Full-pipeline eval on the same held-out split
├── bench/
│   └── load_test.py                  # Load test against /scan, writes bench/results.md
├── console/                          # React/TS console (Vite, strict TS, Tailwind)
│   ├── src/
│   │   ├── api/                      # openapi-typescript-generated client + TanStack Query hooks
│   │   ├── components/
│   │   ├── routes/                   # PoliciesPage, AuditPage, MetricsPage
│   │   └── test/                     # Vitest setup + MSW mocks
│   └── vercel.json                   # SPA rewrite for client-side routing
├── prometheus/
│   └── prometheus.yml                # Scrape config
├── grafana/
│   ├── provisioning/                 # Datasource + dashboard provisioning (zero-click)
│   └── dashboards/sentinellm.json    # 5-panel dashboard, committed
├── admin/
│   └── streamlit_app.py              # Audit dashboard with charts
├── tests/
│   ├── conftest.py
│   ├── test_regex.py
│   ├── test_policy.py
│   ├── test_redact.py
│   ├── test_orchestrator.py
│   ├── test_integration.py
│   ├── test_streaming.py
│   └── test_console_api.py
├── docker-compose.yml                # gateway + db + admin + prometheus + grafana + jaeger
├── Dockerfile.train
├── train.sh
└── Makefile
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTINELLM_LLM_BACKEND` | `ollama` | `ollama` or `openai` |
| `SENTINELLM_OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API URL |
| `SENTINELLM_OLLAMA_MODEL` | `qwen2.5:0.5b` | Default model |
| `SENTINELLM_OPENAI_API_KEY` | `""` | OpenAI API key (if using OpenAI backend) |
| `SENTINELLM_POLICY_PATH` | `/app/policies/default.yaml` | Seed file, read once if `policies` table is empty |
| `SENTINELLM_MODEL_PATH` | `/app/model/trained` | Semantic model path |
| `SENTINELLM_SEMANTIC_MODEL_ENABLED` | `true` | Enable/disable semantic detector |
| `SENTINELLM_DEBUG` | `false` | Debug logging |
| `SENTINELLM_CONSOLE_API_KEY` | `""` | `X-API-Key` required on `/api/v1/*`; empty disables auth |
| `SENTINELLM_CONSOLE_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowlist for `/api/v1/*` |
| `SENTINELLM_OTEL_ENABLED` | `false` | Auto-instrument FastAPI + export traces via OTLP/HTTP |
| `SENTINELLM_OTEL_EXPORTER_ENDPOINT` | `http://jaeger:4318/v1/traces` | OTLP/HTTP traces endpoint |
| `SENTINELLM_WORKERS` | `4` | `gateway/Dockerfile`'s `uvicorn --workers` count (not a `Settings` field — read directly in the container's `CMD`). Detection is CPU-bound, so this is what actually scales throughput; see Measured Performance. Lower it on constrained hardware — each worker loads its own copy of spaCy + DistilBERT. |

---

## Useful Commands

```bash
make up              # Start full stack (gateway + db + dashboard)
make down            # Stop everything
make logs            # Stream logs
make test            # Run the test suite
make restart-gateway # Restart gateway after model/policy update
bash train.sh        # Full training pipeline via Docker
```

---

## Design Notes

A few decisions worth explaining, not just stating:

**Why policy evaluation is in-line, not a sidecar — and why its store
became the DB, killing YAML hot-reload.** The industry-standard shape for
"a service needs authorization decisions" is a policy sidecar (OPA/Rego is
the canonical example): a separate process the request calls out to. That
buys you a general-purpose policy language and independent deployment, at
the cost of a network hop per decision. SentinelLM evaluates policy in-line
— a plain Python function call inside the same request — because the rule
shape here is narrow (`entity_type` / `action` / `min_confidence`, nothing
more expressive is needed) and the hot-path budget is already tight: regex
alone runs in <1ms, so adding an RPC to a sidecar for a decision this simple
would often cost more than the detection it's gating, and it's one more
container to run and secure for what's meant to be a simple self-hosted
deployment. The storage side of that same component did move, though: the
original design let you edit `default.yaml` and see the change take effect
with no rebuild — a nice property for local iteration. But it's fundamentally
incompatible with a console (or this console API) actually *authoring*
policy: a running process can't be made to notice a file it didn't open
changed without polling or a restart, while a DB row it queries is live by
construction. Once "developers can create/edit/delete rules through an API"
became a real requirement, the YAML had to become a one-time seed rather than
the source of truth. The workflow cost is real — you can no longer `vim` a
threshold and have it apply — and worth naming rather than hiding.

**Why `Histogram`, not `Summary`, for latency metrics.** A `Summary`
computes quantiles client-side, per process — you cannot average or combine
them across replicas, and Grafana's `histogram_quantile()` doesn't work on
one. A `Histogram` exports raw bucket counts, so Prometheus can aggregate
across every gateway instance and compute a real p95 in the query layer. The
cost is you choose bucket boundaries up front; get them wrong (as the first
pass here did — buckets that topped out at 1.0s, before the load test proved
real p95 hits 913ms under load) and `histogram_quantile` silently returns
garbage for the tail. Fixed once the actual load test existed to catch it.

**Why cursor pagination, not `OFFSET`, on the audit endpoint.** The audit
table grows with every request. `OFFSET 50000` means the database still
walks and discards 50,000 rows before returning page 501 — a real,
user-visible stall once the table isn't small. Keyset pagination on
`(created_at, id)` costs a slightly odd cursor token in the API instead, and
stays O(page size) regardless of how deep you page.

**What actually fixed the ~130 req/s ceiling — and what didn't.** One
`--workers 1` process showed identical throughput at concurrency 10 and 100
(~130 req/s either way), proving the process was saturated rather than
requests being individually slow. Diagnosing before changing anything:
disabling the semantic detector alone jumped throughput 126.6 → 521.5 req/s,
isolating DistilBERT inference as the dominant cost. `torch.set_num_threads(1)`
— the obvious, zero-cost fix for thread oversubscription — changed nothing
measurably (127.3 req/s); `top` during the run showed only ~2.5–3.7 of 10
cores in use with the system 60% idle, so oversubscription was never the
problem. Adding 4 worker processes alone helped little (160.0 req/s, and
p95 got *worse*) — most likely a per-process GIL ceiling around the
CPU-bound inference (not ruled out as a contributor: Presidio's own
2-worker thread pool cap), so more processes just added scheduling
contention on top of the same per-request cost. What worked was
reducing that cost: the orchestrator now runs regex + Presidio first and
only escalates to semantic when neither already gave a confident answer,
which cut real requests' need for the ~150ms pass roughly in half. Combined
with 4 workers, that reached 258.5 req/s (2.04x baseline) with p95 down 20%
(913ms → 732ms) — see `bench/results.md` for the full step-by-step
measurements and `gateway/app/detectors/orchestrator.py` for why the skip
condition is gated on regex findings specifically, not any fast-pass
finding (the naive version cost real accuracy — see below).

**Why the console (once built) won't get a live link.** Railway dropped its
free tier in 2023 (now a one-time trial credit, then a paid Hobby plan); Fly
requires a card on file even for its free allowance. Neither is free with
zero strings. The demo stays on Hugging Face Spaces — genuinely free, no
card, already deployed — which is the right call for `/scan` (stateless,
detection-only) but can't hold a persistent audit trail across the sleep
cycles a free Space goes through. So the console gets a local
`docker compose up` walkthrough and a screen recording in the README instead
of a live URL, once it exists. A dead or misleading "live" link on a resume
is worse than an honest local-only demo.

---

## License

MIT — see [LICENSE](LICENSE)

---

**Built by [Varshith Peddineni](https://github.com/varshith145)**
