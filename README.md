# SentinelLM 🛡️

> **A self-hosted AI gateway that intercepts LLM traffic, detects PII and secrets across three detection passes, enforces configurable policies, and logs every decision — without sending your data anywhere.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/DistilBERT-NER%20F1%3D0.849-orange?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-244%20passed-brightgreen?logo=pytest&logoColor=white" />
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
  Your App / Client
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
│               │   Policy Engine    │   ← YAML config        │
│               │  ALLOW / MASK / BLOCK                        │
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

All three passes run in parallel via `asyncio.gather`. The orchestrator merges results, deduplicates overlapping spans (keeping highest confidence), and applies detector priority: `semantic > presidio > regex` on ties.

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
| Admin Dashboard | http://localhost:8501 |
| API docs | http://localhost:8000/docs |

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

## Policy Configuration

Policies live in `gateway/policies/default.yaml` and are **hot-reloaded** — no Docker rebuild needed when you change thresholds.

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
244 passed in 0.51s
```

| File | Tests | Coverage |
|------|-------|---------|
| `test_regex.py` | 39 | Luhn algorithm, all 7 regex patterns, offsets, false-positive guards |
| `test_policy.py` | 32 | All entity types, every confidence threshold boundary, output scanning |
| `test_redact.py` | 21 | All 11 entity types + tokens, positions, adjacency, count aggregation |
| `test_orchestrator.py` | 23 | Deduplication, overlap handling, detector priority, failure resilience |
| `test_integration.py` | 20 | Full pipeline: detect → evaluate → redact, end-to-end per scenario |
| `test_streaming.py` | 29 | SSE chunk assembly, redacted re-emission, clean pass-through, error paths |

---

## Project Structure

```
SentinelLM/
├── gateway/                         # FastAPI gateway
│   ├── app/
│   │   ├── main.py                  # Entrypoint + request pipeline (streaming + non-streaming)
│   │   ├── config.py                # Env-based settings (pydantic-settings)
│   │   ├── policy.py                # YAML policy engine
│   │   ├── redact.py                # Typed-token redaction
│   │   ├── audit.py                 # Async audit log writer
│   │   ├── db.py                    # SQLAlchemy async + PostgreSQL
│   │   ├── metrics.py               # Prometheus counters, histograms, gauges
│   │   └── detectors/
│   │       ├── base.py              # EntityType, Finding, BaseDetector
│   │       ├── regex.py             # Pass 1: compiled patterns + Luhn
│   │       ├── presidio_detector.py # Pass 2: Microsoft Presidio
│   │       ├── semantic.py          # Pass 3: DistilBERT NER
│   │       └── orchestrator.py      # asyncio.gather + deduplication
│   └── policies/
│       └── default.yaml             # Live-mounted policy (hot reload)
├── model/
│   ├── data/
│   │   ├── generate_training_data.py # 410+ synthetic obfuscated examples
│   │   ├── prepare_dataset.py        # Tokenize + BIO label alignment
│   │   ├── synthetic_obfuscated.jsonl
│   │   └── hard_negatives.jsonl
│   ├── train.py                      # WeightedLossTrainer, 12 epochs
│   └── evaluate.py                   # seqeval F1/precision/recall
├── admin/
│   └── streamlit_app.py              # Audit dashboard with charts
├── tests/
│   ├── conftest.py
│   ├── test_regex.py
│   ├── test_policy.py
│   ├── test_redact.py
│   ├── test_orchestrator.py
│   ├── test_integration.py
│   └── test_streaming.py
├── docker-compose.yml
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
| `SENTINELLM_POLICY_PATH` | `/app/policies/default.yaml` | Policy file path |
| `SENTINELLM_MODEL_PATH` | `/app/model/trained` | Semantic model path |
| `SENTINELLM_SEMANTIC_MODEL_ENABLED` | `true` | Enable/disable semantic detector |
| `SENTINELLM_DEBUG` | `false` | Debug logging |

---

## Useful Commands

```bash
make up              # Start full stack (gateway + db + dashboard)
make down            # Stop everything
make logs            # Stream logs
make test            # Run test suite (244 tests)
make restart-gateway # Restart gateway after model/policy update
bash train.sh        # Full training pipeline via Docker
```

---

## License

MIT — see [LICENSE](LICENSE)

---

**Built by [Varshith Peddineni](https://github.com/varshith145)**
