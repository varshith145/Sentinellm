# SentinelLM 🛡️

**An AI gateway that detects and redacts sensitive data in LLM traffic using a three-pass detection pipeline (regex + Presidio + fine-tuned semantic NER model), with full audit logging and an admin dashboard.**

[![CI](https://github.com/varshith145/sentinellm/actions/workflows/ci.yml/badge.svg)](https://github.com/varshith145/sentinellm/actions/workflows/ci.yml)

---

## 🔍 What is SentinelLM?

SentinelLM is a **self-hosted AI gateway** (reverse proxy) that sits between any application and an LLM backend. It inspects every prompt and every response using a **three-pass detection pipeline** to find sensitive data, enforces configurable policies (allow, mask, or block), and logs every decision to a tamper-evident audit store.

### The Problem

Organizations adopting LLMs face three risks:
1. **Data Leakage** — Employees paste PII and secrets into LLM prompts
2. **No Visibility** — Security teams can't see what's being sent to models
3. **Obfuscation Bypasses** — "john at gmail dot com" bypasses regex scanning

### The Solution: Three-Pass Detection

| Pass | Technology | What It Catches | Latency |
|------|-----------|----------------|---------|
| 1️⃣ **Regex** | Compiled patterns | Structured PII (emails, SSNs, cards) + Secrets (AWS keys, JWTs) | <1ms |
| 2️⃣ **Presidio** | Microsoft Presidio + spaCy | Contextual PII (names, addresses, IDs) | ~30ms |
| 3️⃣ **Semantic NER** | Fine-tuned DistilBERT | Obfuscated PII ("john at gmail dot com") | ~100-200ms |

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Client / App   │
└────────┬────────┘
         │ POST /v1/chat/completions
         ▼
┌─────────────────────────────────────────────┐
│          SentinelLM Gateway (FastAPI)        │
│                                             │
│  Input Scanning → Policy Engine → Redact    │
│       ↓                                     │
│  Forward to LLM (Ollama / OpenAI)           │
│       ↓                                     │
│  Output Scanning → Return + ppg metadata    │
│       ↓                                     │
│  Audit Log (PostgreSQL)                     │
└─────────────────────────────────────────────┘
         │
         ▼
┌────────────────┐     ┌──────────────────┐
│  PostgreSQL    │◄────│  Streamlit Admin  │
│  (audit_log)   │     │  Dashboard        │
└────────────────┘     └──────────────────┘
```

---

## 🚀 Quickstart

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2
- [Ollama](https://ollama.ai/) running locally (or an OpenAI API key)

### 1. Clone and Configure

```bash
git clone https://github.com/varshith145/sentinellm.git
cd sentinellm
cp .env.example .env
# Edit .env if needed (defaults work with local Ollama)
```

### 2. Start the Stack

```bash
docker compose up --build
```

This starts:
- **Gateway** on `http://localhost:8000`
- **PostgreSQL** on `localhost:5432`
- **Admin Dashboard** on `http://localhost:8501`

### 3. Send a Test Request

**Test PII masking (email gets redacted):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Summarize this email from john@acme.com about project Alpha."}],
    "stream": false,
    "user": "demo-user"
  }' | python -m json.tool
```

Expected: The email `john@acme.com` is replaced with `[REDACTED_EMAIL]` before being sent to the LLM.

**Test secret blocking (request is blocked entirely):**

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Use this key AKIAIOSFODNN7EXAMPLE to access S3"}],
    "stream": false
  }' | python -m json.tool
```

Expected: Returns 403 — the LLM never receives the AWS key.

### 4. View the Dashboard

Open [http://localhost:8501](http://localhost:8501) to see the admin dashboard with audit records.

---

## 📁 Project Structure

```
SentinelLM/
├── gateway/                    # FastAPI gateway application
│   ├── app/
│   │   ├── main.py             # FastAPI app + pipeline
│   │   ├── config.py           # Environment-based config
│   │   ├── models.py           # Pydantic request/response models
│   │   ├── db.py               # SQLAlchemy async models
│   │   ├── audit.py            # Audit log writer
│   │   ├── policy.py           # Policy engine (YAML rules)
│   │   ├── redact.py           # Redaction with typed tokens
│   │   └── detectors/
│   │       ├── base.py         # EntityType, Finding, BaseDetector
│   │       ├── regex.py        # Pass 1: Regex patterns
│   │       ├── presidio_detector.py  # Pass 2: Presidio
│   │       ├── semantic.py     # Pass 3: DistilBERT NER
│   │       └── orchestrator.py # Merge + deduplicate
│   ├── policies/
│   │   └── default.yaml        # Default policy config
│   ├── requirements.txt
│   └── Dockerfile
├── model/                      # ML training pipeline
│   ├── data/
│   │   ├── synthetic_obfuscated.jsonl
│   │   ├── hard_negatives.jsonl
│   │   └── prepare_dataset.py
│   ├── configs/
│   │   └── training_config.yaml
│   ├── train.py
│   └── evaluate.py
├── admin/                      # Streamlit dashboard
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── tests/                      # Test suite
│   ├── test_regex.py
│   ├── test_orchestrator.py
│   ├── test_policy.py
│   └── test_redact.py
├── docker-compose.yml
├── .env.example
└── .github/workflows/ci.yml
```

---

## 🔧 Configuration

### Policy Configuration

Edit `gateway/policies/default.yaml` to customize detection behavior:

```yaml
rules:
  - entity_type: EMAIL
    action: MASK          # ALLOW, MASK, or BLOCK
    min_confidence: 0.7   # Minimum confidence threshold

  - entity_type: AWS_KEY
    action: BLOCK
    min_confidence: 0.5
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTINELLM_LLM_BACKEND` | `ollama` | `ollama` or `openai` |
| `SENTINELLM_OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API URL |
| `SENTINELLM_OPENAI_API_KEY` | `""` | OpenAI API key |
| `SENTINELLM_SEMANTIC_MODEL_ENABLED` | `true` | Enable/disable semantic detector |

---

## 🤖 Training the Semantic Model

The semantic NER model is optional but recommended for catching obfuscated PII.

### 1. Prepare Dataset

```bash
cd model/data
python prepare_dataset.py
```

### 2. Train

```bash
python model/train.py
```

### 3. Evaluate

```bash
python model/evaluate.py
```

The trained model will be saved to `model/trained/` and automatically mounted into the gateway container.

---

## 🧪 Testing

```bash
# Install test dependencies
pip install -r gateway/requirements.txt

# Run tests
python -m pytest tests/ -v
```

---

## 📊 API Response

Every response includes a `ppg` metadata block:

```json
{
  "choices": [...],
  "ppg": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "input_decision": "MASK",
    "output_decision": "ALLOW",
    "input_redactions": {"EMAIL": 1},
    "output_redactions": {},
    "policy_id": "default-v1",
    "detectors_used": ["regex", "presidio", "semantic"],
    "latency_ms": {
      "detection": 185,
      "llm": 2340,
      "total": 2540
    }
  }
}
```

---

## 📄 License

MIT

---

**Built by [Varshith Peddineni](https://github.com/varshith145)**
