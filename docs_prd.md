# SentinelLM — Product Requirements Document (PRD)

**Project:** SentinelLM  
**Author:** Varshith Peddineni  
**Date:** 2026-02-11  
**Version:** 1.0  
**Repository:** `varshith145/sentinellm`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Definition](#2-project-definition)
3. [Goals and Success Criteria](#3-goals-and-success-criteria)
4. [Competitive Landscape](#4-competitive-landscape)
5. [Tech Stack](#5-tech-stack)
6. [System Architecture](#6-system-architecture)
7. [Request/Response Pipeline](#7-requestresponse-pipeline)
8. [Three-Pass Detection Layer](#8-three-pass-detection-layer)
9. [Semantic Model — Training and Integration](#9-semantic-model--training-and-integration)
10. [Policy Engine](#10-policy-engine)
11. [Redaction and Transformation](#11-redaction-and-transformation)
12. [Audit Logging](#12-audit-logging)
13. [Admin Dashboard (Streamlit)](#13-admin-dashboard-streamlit)
14. [API Design (FastAPI Gateway)](#14-api-design-fastapi-gateway)
15. [Database Schema (PostgreSQL)](#15-database-schema-postgresql)
16. [Docker Compose Setup](#16-docker-compose-setup)
17. [Configuration](#17-configuration)
18. [Testing Strategy](#18-testing-strategy)
19. [CI/CD (GitHub Actions)](#19-cicd-github-actions)
20. [Evaluation and Metrics](#20-evaluation-and-metrics)
21. [Build Order (Week-by-Week)](#21-build-order-week-by-week)
22. [Folder Structure](#22-folder-structure)
23. [README Requirements](#23-readme-requirements)
24. [Known Limitations](#24-known-limitations)
25. [Future Work (V2)](#25-future-work-v2)

---

## 1. Problem Statement

Organizations are adopting LLMs (ChatGPT, Copilot, internal Ollama deployments) across engineering, support, and operations teams. This creates three concrete risks:

**Risk 1 — Accidental data leakage.** Employees paste PII (emails, phone numbers, SSNs, credit card numbers) and secrets (API keys, tokens, passwords) into LLM prompts. Once sent, that data may be logged, cached, or used for training by the provider. There is no "undo."

**Risk 2 — No visibility.** Security teams cannot answer basic questions: Who sent what to the model? Was any sensitive data included? What happened to it? There is no audit trail.

**Risk 3 — Obfuscation bypasses pattern matching.** Even organizations that deploy basic regex-based scanning get bypassed by users who write PII in natural language — "my email is john at gmail dot com" or "the password is hunter two." Regex catches `john@gmail.com` but not the obfuscated version.

SentinelLM solves all three.

---

## 2. Project Definition

**SentinelLM** is a self-hosted AI gateway (reverse proxy) that sits between any application and an LLM backend. It inspects every prompt and every response using a **three-pass detection pipeline** — regex, Microsoft Presidio, and a custom fine-tuned semantic model — to detect sensitive data. It enforces configurable policies (allow, mask, or block), logs every decision to a tamper-evident audit store, and provides an admin dashboard for review.

### What makes it different from existing tools

- **Three-pass detection pipeline.** Regex for speed, Presidio for structured PII, and a fine-tuned DistilBERT NER model for obfuscated/informal PII that pattern matching misses.
- **Output scanning.** Scans model responses before returning them to the client. Most open-source alternatives only scan input.
- **Self-hosted, single command.** `docker compose up` runs the entire stack. No SaaS dependency, no API keys to a third party.
- **Full audit trail.** Every request and decision is logged with redacted content, never raw secrets.

### One-sentence pitch

*SentinelLM is an AI gateway that detects and redacts sensitive data in LLM traffic using a three-pass detection pipeline (regex + Presidio + fine-tuned semantic NER model), with full audit logging and an admin dashboard.*

---

## 3. Goals and Success Criteria

### Primary Goals

| # | Goal | Measurable Outcome |
|---|------|-------------------|
| G1 | Provide an OpenAI-compatible proxy endpoint | `POST /v1/chat/completions` works with any client that speaks OpenAI format |
| G2 | Detect structured PII via regex + Presidio | Email, phone, SSN, credit card detected with >95% precision on structured inputs |
| G3 | Detect obfuscated PII via semantic model | "john at gmail dot com" detected with >80% recall on obfuscated eval set |
| G4 | Detect secrets and block them | AWS keys, GitHub tokens, JWTs, generic passwords blocked — never forwarded to LLM |
| G5 | Enforce configurable policies | YAML-based policy file determines allow/mask/block per entity type |
| G6 | Scan model output | Responses scanned before returning to client, same pipeline as input |
| G7 | Log every decision | Postgres audit table with redacted content, decision, reasons, timestamps |
| G8 | Provide admin dashboard | Streamlit UI to browse, filter, and inspect audit records |
| G9 | Run as one command | `docker compose up` starts gateway + db + admin |
| G10 | Tests and CI | pytest suite + GitHub Actions running lint, type check, tests on every push |

### Success Criteria (Definition of Done)

- [ ] A prompt containing `john@example.com` is masked to `[REDACTED_EMAIL]` before forwarding.
- [ ] A prompt containing `ghp_abc123def456` is blocked entirely — LLM never receives it.
- [ ] A prompt containing `"my email is john at gmail dot com"` is detected by the semantic model and masked.
- [ ] A model response containing PII is redacted before returning to the client.
- [ ] Admin dashboard shows all requests with decision, reasons, redaction counts, and timestamps.
- [ ] All tests pass in CI.
- [ ] `docker compose up` brings up the full stack with no manual steps.
- [ ] README contains architecture diagram, quickstart, example curl commands, and screenshot of dashboard.

---

## 4. Competitive Landscape

You must know these exist. Interviewers will ask.

| Tool | What It Does | How SentinelLM Differs |
|------|-------------|----------------------|
| **LLM Guard** (open source) | Input/output scanning with regex + Presidio + some heuristics | No semantic NER model. No fine-tuned detection for obfuscated PII. SentinelLM adds the ML layer. |
| **Lakera Guard** (commercial SaaS) | Prompt injection + PII detection API | SaaS — you send your data to them for scanning. SentinelLM is fully self-hosted. |
| **Nightfall AI** (commercial SaaS) | DLP for AI, Slack, email | Commercial product, not open source. Different target (enterprise DLP platform). |
| **LiteLLM** (open source) | Multi-provider LLM proxy with auth, logging | Focuses on routing and cost tracking, not security scanning. No PII detection. |
| **Portkey AI Gateway** (commercial) | LLM gateway with observability, caching, fallbacks | No security scanning. Focused on reliability and cost. |
| **Microsoft Presidio** (open source library) | PII detection and anonymization library | A library, not a gateway. SentinelLM uses Presidio as one component in a larger pipeline. |

**SentinelLM's positioning:** Open-source, self-hosted AI gateway with a three-pass detection pipeline (regex + Presidio + semantic ML model) that scans both input and output. No SaaS dependency.

---

## 5. Tech Stack

| Component | Technology | Version | Why This Choice |
|-----------|-----------|---------|----------------|
| Language | Python | 3.11+ | Ecosystem support for ML + web |
| Gateway API | FastAPI + Uvicorn | FastAPI 0.100+ | Async, fast, OpenAPI docs auto-generated |
| LLM Backend | Ollama (local) | Latest | Free, local, OpenAI-compatible API |
| LLM Backend (optional) | OpenAI API | v1 | Pluggable backend for cloud deployments |
| PII Detection (structured) | Microsoft Presidio | 2.2+ | Industry standard, extensible, supports custom recognizers |
| PII Detection (semantic) | Fine-tuned DistilBERT | HuggingFace transformers 4.x | 66M params, fast inference on CPU, good enough accuracy for NER |
| NLP Backend for Presidio | spaCy | 3.x (en_core_web_lg) | Required by Presidio for NER |
| Database | PostgreSQL | 15+ | Reliable, JSONB support for structured metadata |
| ORM / DB access | SQLAlchemy + asyncpg | 2.0+ | Async Postgres access from FastAPI |
| Admin UI | Streamlit | 1.30+ | Fast to build, good enough for dashboards |
| HTTP Client | httpx | 0.27+ | Async HTTP client for gateway → LLM calls |
| Containerization | Docker + Docker Compose | Compose v2 | One-command deployment |
| Testing | pytest + pytest-asyncio | Latest | Async test support for FastAPI |
| Linting | ruff | Latest | Fast, replaces flake8 + isort + black |
| Type Checking | mypy | Latest | Static type analysis |
| CI | GitHub Actions | N/A | Free for public repos |
| ML Training | HuggingFace Transformers + Datasets | 4.x | Standard fine-tuning workflow |
| ML Evaluation | seqeval | Latest | Sequence-level NER metrics (precision, recall, F1) |

---

## 6. System Architecture

### High-Level Flow

```
┌─────────────────┐
│  Client / App   │
│  (any OpenAI-   │
│   compatible    │
│   client)       │
└────────┬────────┘
         │ POST /v1/chat/completions
         ▼
┌─────────────────────────────────────────────┐
│          SentinelLM Gateway (FastAPI)        │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │  INPUT SCANNING                     │    │
│  │                                     │    │
│  │  ┌──────────┐  ┌──────────┐        │    │
│  │  │  Regex   │  │ Presidio │        │    │
│  │  │  Pass    │  │  Pass    │        │    │
│  │  │  (<1ms)  │  │ (~30ms)  │        │    │
│  │  └────┬─────┘  └────┬─────┘        │    │
│  │       │              │              │    │
│  │  ┌────▼──────────────▼─────┐        │    │
│  │  │     Semantic Model      │        │    │
│  │  │  (DistilBERT NER)       │        │    │
│  │  │  (~100-200ms)           │        │    │
│  │  └────────────┬────────────┘        │    │
│  │               │                     │    │
│  │  ┌────────────▼────────────┐        │    │
│  │  │   Orchestrator          │        │    │
│  │  │   (merge + deduplicate) │        │    │
│  │  └────────────┬────────────┘        │    │
│  └───────────────┼─────────────────────┘    │
│                  │                           │
│  ┌───────────────▼─────────────────────┐    │
│  │  Policy Engine (YAML rules)         │    │
│  │  → ALLOW / MASK / BLOCK             │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────▼─────────────────────┐    │
│  │  Redaction (if MASK)                │    │
│  │  Replace matched spans with tokens  │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────▼─────────────────────┐    │
│  │  Audit Logger                       │    │
│  │  Write to Postgres (redacted only)  │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│         ┌────────▼────────┐                  │
│         │  BLOCK?         │                  │
│         │  Yes → return   │                  │
│         │  policy error   │                  │
│         │  No → forward   │                  │
│         └────────┬────────┘                  │
│                  │                           │
│  ┌───────────────▼─────────────────────┐    │
│  │  Forward sanitized request to LLM   │    │
│  │  (Ollama / OpenAI / Anthropic)      │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────▼─────────────────────┐    │
│  │  OUTPUT SCANNING                    │    │
│  │  Same three-pass pipeline on        │    │
│  │  model response content             │    │
│  └───────────────┬─────────────────────┘    │
│                  │                           │
│  ┌───────────────▼─────────────────────┐    │
│  │  Return response + ppg metadata     │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  PostgreSQL     │◄────│  Streamlit       │
│  (audit_log)    │     │  Admin Dashboard │
└─────────────────┘     └─────────────────┘
```

### Deployment Model

- **Ollama** runs natively on the host machine (Mac/Linux).
- **Gateway**, **PostgreSQL**, and **Streamlit** run inside Docker Compose.
- Gateway reaches Ollama via `http://host.docker.internal:11434` (Mac) or `http://172.17.0.1:11434` (Linux).
- The fine-tuned DistilBERT model is baked into the gateway Docker image (or mounted as a volume).

### Port Assignments

| Service | Internal Port | Exposed Port |
|---------|--------------|-------------|
| Gateway (FastAPI) | 8000 | 8000 |
| PostgreSQL | 5432 | 5432 (optional) |
| Streamlit Admin | 8501 | 8501 |
| Ollama (host) | 11434 | N/A (host network) |

---

## 7. Request/Response Pipeline

This section specifies exactly what happens when a request hits the gateway end to end. Every step is numbered. Implement them in this order.

### 7.1 — Receive Request

The gateway exposes `POST /v1/chat/completions`. It accepts the OpenAI chat completions request format:

```json
{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this email from john@acme.com about project Alpha."}
  ],
  "temperature": 0.7,
  "stream": false,
  "user": "varshith"
}
```

**Implementation detail:** Use a Pydantic model to validate the incoming request. Accept all standard OpenAI fields (`model`, `messages`, `temperature`, `top_p`, `max_tokens`, `stream`, `user`, etc.). Pass through any fields you don't explicitly handle — the gateway should be transparent for fields it doesn't care about.

### 7.2 — Extract Scannable Text

Extract the `content` field from every message in the `messages` array. Scan all roles: `user`, `system`, `assistant`, and `tool`. Each message is scanned independently but results are aggregated.

```python
# Pseudocode
texts_to_scan = []
for msg in request.messages:
    if msg.content:
        texts_to_scan.append({
            "role": msg.role,
            "index": i,
            "content": msg.content
        })
```

### 7.3 — Run Three-Pass Detection

For each extracted text, run all three detectors. They run in **parallel** (asyncio.gather) to minimize latency. See [Section 8](#8-three-pass-detection-layer) for full detector specifications.

```python
# Pseudocode
async def scan_text(text: str) -> list[Finding]:
    regex_findings, presidio_findings, semantic_findings = await asyncio.gather(
        regex_detector.detect(text),
        presidio_detector.detect(text),
        semantic_detector.detect(text),
    )
    return orchestrator.merge(regex_findings, presidio_findings, semantic_findings)
```

### 7.4 — Policy Decision

Pass merged findings to the policy engine. The policy engine reads rules from a YAML config file and returns a decision: `ALLOW`, `MASK`, or `BLOCK`. See [Section 10](#10-policy-engine) for full policy specification.

Decision priority: `BLOCK` > `MASK` > `ALLOW`. If any finding triggers `BLOCK`, the entire request is blocked.

### 7.5 — Redact (if MASK)

If the decision is `MASK`, replace matched spans in the original text with typed redaction tokens. See [Section 11](#11-redaction-and-transformation) for token formats.

### 7.6 — Write Audit Record

Write the audit record to Postgres **before** forwarding to the LLM. This ensures that even if the LLM call fails or times out, the audit record exists. The audit record contains only **redacted** content — never raw secrets or PII. See [Section 12](#12-audit-logging) for the full schema.

### 7.7 — Block or Forward

- If decision is `BLOCK`: return a policy violation error immediately. Do **not** call the LLM.
- If decision is `ALLOW` or `MASK`: construct the sanitized request and forward to the configured LLM backend.

### 7.8 — Forward to LLM Backend

Use `httpx.AsyncClient` to forward the sanitized request to the LLM backend. The backend is configured via environment variable (`OLLAMA_BASE_URL` or `OPENAI_API_KEY` + `OPENAI_BASE_URL`).

```python
# Pseudocode
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{backend_url}/v1/chat/completions",
        json=sanitized_request,
        timeout=120.0
    )
```

### 7.9 — Output Scanning

Run the same three-pass detection pipeline on the model's response content. Extract `choices[*].message.content` from the response and scan it.

- If output contains secrets → redact or block the response (configurable).
- If output contains PII → redact the response.

### 7.10 — Return Response

Return the (possibly redacted) model response to the client. Add a `ppg` metadata block:

```json
{
  "id": "chatcmpl-xxx",
  "choices": [...],
  "ppg": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "input_decision": "MASK",
    "output_decision": "ALLOW",
    "input_redactions": {"EMAIL": 1, "PHONE": 1},
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

## 8. Three-Pass Detection Layer

This is the core of SentinelLM. Three detectors run in parallel on every input and output text. Results are merged by an orchestrator.

### 8.1 — Detector Interface

Every detector implements this interface:

```python
from dataclasses import dataclass
from enum import Enum

class EntityType(str, Enum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    AWS_KEY = "AWS_KEY"
    GITHUB_TOKEN = "GITHUB_TOKEN"
    JWT = "JWT"
    PASSWORD = "PASSWORD"
    PERSON_NAME = "PERSON_NAME"
    GENERIC_PII = "GENERIC_PII"
    GENERIC_SECRET = "GENERIC_SECRET"

class EntityCategory(str, Enum):
    PII = "PII"
    SECRET = "SECRET"

@dataclass
class Finding:
    entity_type: EntityType
    category: EntityCategory
    start: int          # character offset in original text
    end: int            # character offset in original text
    matched_text: str   # the actual matched substring
    confidence: float   # 0.0 to 1.0
    detector: str       # "regex", "presidio", or "semantic"

class BaseDetector:
    async def detect(self, text: str) -> list[Finding]:
        raise NotImplementedError
```

### 8.2 — Pass 1: Regex Detector

**File:** `gateway/app/detectors/regex.py`

Fast, deterministic pattern matching. Runs in <1ms. Catches structured/formatted PII and secrets.

#### Patterns to implement:

**PII patterns:**

| Entity | Regex Pattern | Notes |
|--------|--------------|-------|
| EMAIL | `r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b'` | Standard email |
| PHONE | `r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'` | US phone numbers |
| SSN | `r'\b\d{3}-\d{2}-\d{4}\b'` | US Social Security Number format |
| CREDIT_CARD | `r'\b(?:\d{4}[-\s]?){3}\d{4}\b'` | 16-digit card numbers, with Luhn validation |

**Secret patterns:**

| Entity | Regex Pattern | Notes |
|--------|--------------|-------|
| AWS_KEY | `r'\bAKIA[0-9A-Z]{16}\b'` | AWS access key ID |
| GITHUB_TOKEN | `r'\b(ghp_[A-Za-z0-9]{36}\|github_pat_[A-Za-z0-9_]{22,})\b'` | GitHub personal access tokens |
| JWT | `r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'` | Three base64url segments starting with eyJ |

**Credit card Luhn validation:**

```python
def luhn_check(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 16:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0
```

**Implementation requirements:**
- Compile all regex patterns at module load time (not per-request).
- Return `Finding` objects with exact character offsets (`start`, `end`).
- Set `confidence` to `0.95` for regex matches (high confidence since pattern matched exactly).
- For credit cards, only return finding if Luhn check passes. Set confidence to `0.99` if Luhn passes.

### 8.3 — Pass 2: Presidio Detector

**File:** `gateway/app/detectors/presidio_detector.py`

Uses Microsoft Presidio Analyzer for structured PII detection. Presidio uses spaCy NER + regex + context analysis internally.

**Setup:**

```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()  # Initialize once at startup
```

**Entities to request from Presidio:**

```python
PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "PERSON",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
]
```

**Mapping Presidio entities to SentinelLM EntityType:**

```python
PRESIDIO_TO_ENTITY_TYPE = {
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "PHONE_NUMBER": EntityType.PHONE,
    "US_SSN": EntityType.SSN,
    "CREDIT_CARD": EntityType.CREDIT_CARD,
    "PERSON": EntityType.PERSON_NAME,
    "US_BANK_NUMBER": EntityType.GENERIC_PII,
    "US_PASSPORT": EntityType.GENERIC_PII,
    "US_DRIVER_LICENSE": EntityType.GENERIC_PII,
    "IP_ADDRESS": EntityType.GENERIC_PII,
}
```

**Implementation requirements:**
- Initialize `AnalyzerEngine` once at app startup, not per request.
- Run Presidio in a thread pool executor (it's synchronous, don't block the async event loop):

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def detect(self, text: str) -> list[Finding]:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        executor,
        lambda: self.analyzer.analyze(text=text, entities=PRESIDIO_ENTITIES, language="en")
    )
    return [self._to_finding(r, text) for r in results]
```

- Set `confidence` from Presidio's own score.
- Set `detector` to `"presidio"`.
- Presidio requires the `en_core_web_lg` spaCy model. Install it in the Dockerfile.

### 8.4 — Pass 3: Semantic Detector (Fine-Tuned DistilBERT)

**File:** `gateway/app/detectors/semantic.py`

This is the differentiator. A fine-tuned DistilBERT model that performs token-level NER to catch PII that regex and Presidio miss — specifically obfuscated, informal, or context-dependent PII.

**What this catches that the other two don't:**

| Input | Regex | Presidio | Semantic Model |
|-------|-------|----------|---------------|
| `john@gmail.com` | ✅ | ✅ | ✅ |
| `john at gmail dot com` | ❌ | ❌ | ✅ |
| `my social is four five six 78 9012` | ❌ | ❌ | ✅ |
| `reach me on gmail, name's john smith` | ❌ | Maybe (PERSON) | ✅ (email intent) |
| `the password is hunter2` | ❌ | ❌ | ✅ |
| `use token abc123xyz to authenticate` | ❌ | ❌ | ✅ |

**Model details are in [Section 9](#9-semantic-model--training-and-integration).**

**Inference implementation:**

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class SemanticDetector(BaseDetector):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    async def detect(self, text: str) -> list[Finding]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect_sync, text)

    def _detect_sync(self, text: str) -> list[Finding]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        findings = []
        current_entity = None
        current_start = None
        current_end = None

        for i, (token, pred, offset) in enumerate(
            zip(tokens, predictions, offset_mapping)
        ):
            label = self.model.config.id2label[pred.item()]

            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    findings.append(self._make_finding(
                        text, current_entity, current_start, current_end
                    ))
                current_entity = label[2:]  # "PII" or "SECRET"
                current_start = offset[0].item()
                current_end = offset[1].item()

            elif label.startswith("I-") and current_entity:
                current_end = offset[1].item()

            else:
                if current_entity:
                    findings.append(self._make_finding(
                        text, current_entity, current_start, current_end
                    ))
                    current_entity = None

        # Don't forget the last entity
        if current_entity:
            findings.append(self._make_finding(
                text, current_entity, current_start, current_end
            ))

        return findings
```

### 8.5 — Orchestrator (Merge and Deduplicate)

**File:** `gateway/app/detectors/orchestrator.py`

The orchestrator runs all three detectors in parallel and merges their findings. It must handle overlapping detections (e.g., regex and Presidio both find the same email).

**Deduplication logic:**

1. Sort all findings by `(start, end)`.
2. For overlapping findings (where one finding's span overlaps another's), keep the one with the highest `confidence`.
3. If confidence is equal, prefer the detector in this priority order: `semantic` > `presidio` > `regex` (because semantic findings are the ones that matter most for differentiation).

```python
class DetectionOrchestrator:
    def __init__(self, detectors: list[BaseDetector]):
        self.detectors = detectors

    async def scan(self, text: str) -> list[Finding]:
        results = await asyncio.gather(
            *[d.detect(text) for d in self.detectors]
        )
        all_findings = [f for result in results for f in result]
        return self._deduplicate(all_findings)

    def _deduplicate(self, findings: list[Finding]) -> list[Finding]:
        if not findings:
            return []

        # Sort by start position, then by confidence descending
        findings.sort(key=lambda f: (f.start, -f.confidence))

        merged = [findings[0]]
        for f in findings[1:]:
            prev = merged[-1]
            # Check overlap
            if f.start < prev.end:
                # Overlapping — keep higher confidence
                if f.confidence > prev.confidence:
                    merged[-1] = f
            else:
                merged.append(f)

        return merged
```

---

## 9. Semantic Model — Training and Integration

This section specifies everything needed to train, evaluate, and deploy the semantic NER model. This is the ML component of SentinelLM.

### 9.1 — Task Definition

**Task:** Token classification (NER)  
**Model:** `distilbert-base-uncased` (66M parameters)  
**Framework:** HuggingFace Transformers  

**Label scheme (BIO tagging):**

| Label | Meaning |
|-------|---------|
| `O` | Not sensitive |
| `B-PII` | Beginning of a PII entity span |
| `I-PII` | Inside a PII entity span |
| `B-SECRET` | Beginning of a secret/credential span |
| `I-SECRET` | Inside a secret/credential span |

**Example:**

```
Input:    "reach   me    at   john   at   gmail   dot   com"
Labels:    O       O     O    B-PII  I-PII I-PII  I-PII I-PII

Input:    "the  password  is   hunter2"
Labels:    O    O         O    B-SECRET

Input:    "call  me  at  five  five  five  one  two  three  four"
Labels:    O     O   O   B-PII I-PII I-PII I-PII I-PII I-PII I-PII
```

### 9.2 — Training Data

You need three types of training data:

**Type 1: Existing PII datasets**

| Dataset | Source | What It Provides |
|---------|--------|-----------------|
| `ai4privacy/pii-masking-400k` | HuggingFace | 400k examples with PII annotations. Mostly structured PII — good for base coverage. |
| `CoNLL-2003` | HuggingFace (`conll2003`) | Standard NER dataset with PERSON, ORG, LOC. Useful for person name detection. |

**Type 2: Synthetic obfuscated PII (YOU MUST GENERATE THIS)**

This is the critical dataset. Use an LLM (GPT-4, Claude) to generate 500-1000 examples of people writing PII in informal/obfuscated ways. Store as JSONL.

**Prompt to generate synthetic data:**

```
Generate 50 examples of sentences where someone includes personal information
in an obfuscated or informal way. For each example, provide:
1. The sentence
2. The spans that contain sensitive information
3. The type (PII or SECRET)

Examples of what I mean:
- "my email is john at gmail dot com" → "john at gmail dot com" is PII
- "the password is hunter2" → "hunter2" is SECRET
- "reach me at five five five, one two three four" → "five five five one two three four" is PII
- "my social starts with four five six" → "four five six" is PII
- "use token abc123xyz for authentication" → "abc123xyz" is SECRET

Be creative. Include:
- Spelled-out phone numbers
- Emails written with "at" and "dot"
- Passwords mentioned in conversation
- API keys or tokens mentioned casually
- SSNs written in words
- Credit card numbers partially mentioned
```

**Synthetic data format (JSONL):**

```json
{"text": "my email is john at gmail dot com", "entities": [{"start": 12, "end": 33, "label": "PII"}]}
{"text": "the password for prod is xK9#mP2$vL", "entities": [{"start": 24, "end": 35, "label": "SECRET"}]}
{"text": "call me at five five five one two three four", "entities": [{"start": 11, "end": 44, "label": "PII"}]}
```

**Type 3: Hard negatives (ESSENTIAL — prevents false positives)**

Generate 200+ examples of text that looks like it could contain PII but doesn't:

```json
{"text": "the email protocol uses SMTP on port 25", "entities": []}
{"text": "call the function with 10 arguments", "entities": []}
{"text": "the password field in the schema is required", "entities": []}
{"text": "use a token-based authentication flow", "entities": []}
{"text": "social security is an important government program", "entities": []}
```

### 9.3 — Dataset Preparation Script

**File:** `model/data/prepare_dataset.py`

This script must:

1. Load the `ai4privacy/pii-masking-400k` dataset from HuggingFace.
2. Convert its annotations to BIO format with `B-PII`, `I-PII` labels.
3. Load your synthetic JSONL files (`synthetic_obfuscated.jsonl`, `hard_negatives.jsonl`).
4. Convert those to BIO format.
5. Merge all datasets.
6. Split into train (80%), validation (10%), test (10%).
7. Tokenize with DistilBERT tokenizer, aligning BIO labels to subword tokens.
8. Save as HuggingFace Dataset on disk.

**Label alignment for subword tokens:**

When DistilBERT tokenizes "gmail" into `["gm", "##ail"]`, both subword tokens must get the same label. The standard approach:

```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], [PAD])
            new_labels.append(-100)  # Ignored in loss
        elif word_id != current_word:
            # First token of a new word
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            # Continuation of a word
            label = labels[word_id]
            # If the label is B-X, change to I-X for continuation
            if label % 2 == 1:  # B- labels are odd in standard mapping
                new_labels.append(label + 1)
            else:
                new_labels.append(label)
    return new_labels
```

### 9.4 — Training Script

**File:** `model/train.py`

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_from_disk
import numpy as np
from seqeval.metrics import classification_report, f1_score

# --- Config ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./model/trained"
NUM_LABELS = 5  # O, B-PII, I-PII, B-SECRET, I-SECRET

LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

# --- Load ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
dataset = load_from_disk("./model/data/processed")

# --- Data Collator ---
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- Metrics ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [LABEL_LIST[l] for (p_i, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_predictions = [
        [LABEL_LIST[p_i] for (p_i, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions),
    }

# --- Training Args ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=False,  # Set True if GPU available
    seed=42,
)

# --- Train ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### 9.5 — Training Configuration

**File:** `model/configs/training_config.yaml`

```yaml
model:
  base: "distilbert-base-uncased"
  num_labels: 5
  labels: ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]

training:
  epochs: 5
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_length: 512
  seed: 42

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  synthetic_data_path: "model/data/synthetic_obfuscated.jsonl"
  hard_negatives_path: "model/data/hard_negatives.jsonl"

evaluation:
  metric: "f1"
  save_best: true
```

### 9.6 — Evaluation Script

**File:** `model/evaluate.py`

This script runs the trained model against the held-out test set and produces a detailed evaluation report.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import json

MODEL_PATH = "./model/trained"
LABEL_LIST = ["O", "B-PII", "I-PII", "B-SECRET", "I-SECRET"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

dataset = load_from_disk("./model/data/processed")
test_data = dataset["test"]

all_true = []
all_pred = []

for example in test_data:
    inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = inputs.word_ids()

    true_labels = []
    pred_labels = []
    prev_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            true_labels.append(LABEL_LIST[example["labels"][word_id]])
            pred_labels.append(LABEL_LIST[predictions[idx]])
        prev_word_id = word_id

    all_true.append(true_labels)
    all_pred.append(pred_labels)

# --- Report ---
report = classification_report(all_true, all_pred)
print(report)

metrics = {
    "precision": precision_score(all_true, all_pred),
    "recall": recall_score(all_true, all_pred),
    "f1": f1_score(all_true, all_pred),
}

with open("./docs/model_eval.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

### 9.7 — Expected Evaluation Results

Publish these numbers honestly. Here is the target and what to expect realistically:

| Detector Configuration | Precision | Recall | F1 | Latency (p50) |
|----------------------|-----------|--------|-----|---------------|
| Regex only | ~0.97 | ~0.40-0.50 | ~0.55-0.65 | <1ms |
| Presidio only | ~0.90 | ~0.60-0.70 | ~0.72-0.78 | ~30ms |
| Regex + Presidio | ~0.93 | ~0.65-0.72 | ~0.77-0.81 | ~32ms |
| **Regex + Presidio + Semantic** | **~0.88-0.92** | **~0.82-0.90** | **~0.85-0.91** | **~150-250ms** |

**The story:** Precision drops slightly when you add the semantic model (more false positives from ML), but recall jumps significantly (catches obfuscated PII). The latency tradeoff is acceptable for a security proxy.

If your actual numbers are worse than this, that's fine. **Report them honestly.** "My model achieved 0.78 F1 on obfuscated PII" is still impressive if you explain why and what you'd do to improve it (more training data, active learning from audit logs, etc.).

### 9.8 — Model Deployment in Gateway

The trained model is loaded once at gateway startup:

```python
# In gateway/app/main.py
from app.detectors.semantic import SemanticDetector

semantic_detector = SemanticDetector(model_path="./model/trained")
```

**Docker considerations:**
- The model files (~250MB for DistilBERT) are either baked into the Docker image or mounted as a volume.
- For the Docker image approach, add to Dockerfile: `COPY model/trained /app/model/trained`
- For the volume approach, add to docker-compose.yml: `volumes: - ./model/trained:/app/model/trained`
- Volume mount is better for development (no rebuild when retraining). Bake in for production.

---

## 10. Policy Engine

### 10.1 — Policy Configuration File

**File:** `gateway/policies/default.yaml`

```yaml
policy_id: "default-v1"
description: "Default SentinelLM policy — mask PII, block secrets"

rules:
  # PII rules — action: MASK
  - entity_type: EMAIL
    category: PII
    action: MASK
    min_confidence: 0.7

  - entity_type: PHONE
    category: PII
    action: MASK
    min_confidence: 0.7

  - entity_type: SSN
    category: PII
    action: MASK
    min_confidence: 0.5    # Lower threshold — SSNs are high risk

  - entity_type: CREDIT_CARD
    category: PII
    action: MASK
    min_confidence: 0.8

  - entity_type: PERSON_NAME
    category: PII
    action: MASK
    min_confidence: 0.85   # Higher threshold to avoid false positives on names

  - entity_type: GENERIC_PII
    category: PII
    action: MASK
    min_confidence: 0.7

  # Secret rules — action: BLOCK
  - entity_type: AWS_KEY
    category: SECRET
    action: BLOCK
    min_confidence: 0.5    # Any confidence — block

  - entity_type: GITHUB_TOKEN
    category: SECRET
    action: BLOCK
    min_confidence: 0.5

  - entity_type: JWT
    category: SECRET
    action: BLOCK
    min_confidence: 0.7

  - entity_type: PASSWORD
    category: SECRET
    action: BLOCK
    min_confidence: 0.6

  - entity_type: GENERIC_SECRET
    category: SECRET
    action: BLOCK
    min_confidence: 0.6

# Default action for entity types not listed above
default_action: ALLOW

# Output scanning policy
output_scanning:
  enabled: true
  # For output, we mask instead of block (model already generated it)
  secret_action: MASK   # Don't block the whole response, just redact
  pii_action: MASK
```

### 10.2 — Policy Engine Implementation

**File:** `gateway/app/policy.py`

```python
import yaml
from pathlib import Path
from app.detectors.base import Finding, EntityCategory

class PolicyDecision:
    def __init__(self, action: str, reasons: list[str], findings: list[Finding]):
        self.action = action          # "ALLOW", "MASK", "BLOCK"
        self.reasons = reasons        # Human-readable reasons
        self.findings = findings      # Findings that triggered this decision

class PolicyEngine:
    def __init__(self, policy_path: str = "policies/default.yaml"):
        with open(policy_path) as f:
            self.config = yaml.safe_load(f)
        self.policy_id = self.config["policy_id"]
        self.rules = {r["entity_type"]: r for r in self.config["rules"]}
        self.default_action = self.config.get("default_action", "ALLOW")

    def evaluate(self, findings: list[Finding]) -> PolicyDecision:
        if not findings:
            return PolicyDecision("ALLOW", [], [])

        action = "ALLOW"
        reasons = []
        actionable_findings = []

        for finding in findings:
            rule = self.rules.get(finding.entity_type.value)

            if rule is None:
                # No rule for this entity type — use default
                continue

            if finding.confidence < rule.get("min_confidence", 0.5):
                # Below confidence threshold — skip
                continue

            rule_action = rule["action"]
            reasons.append(
                f"{rule_action}: {finding.entity_type.value} detected "
                f"(confidence={finding.confidence:.2f}, detector={finding.detector})"
            )
            actionable_findings.append(finding)

            # Escalate action: ALLOW < MASK < BLOCK
            if rule_action == "BLOCK":
                action = "BLOCK"
            elif rule_action == "MASK" and action != "BLOCK":
                action = "MASK"

        return PolicyDecision(action, reasons, actionable_findings)
```

---

## 11. Redaction and Transformation

### 11.1 — Redaction Token Format

When a finding triggers `MASK`, the matched text is replaced with a typed token:

| Entity Type | Redaction Token |
|-------------|----------------|
| EMAIL | `[REDACTED_EMAIL]` |
| PHONE | `[REDACTED_PHONE]` |
| SSN | `[REDACTED_SSN]` |
| CREDIT_CARD | `[REDACTED_CC]` |
| PERSON_NAME | `[REDACTED_NAME]` |
| AWS_KEY | `[REDACTED_SECRET]` |
| GITHUB_TOKEN | `[REDACTED_SECRET]` |
| JWT | `[REDACTED_SECRET]` |
| PASSWORD | `[REDACTED_SECRET]` |
| GENERIC_PII | `[REDACTED_PII]` |
| GENERIC_SECRET | `[REDACTED_SECRET]` |

### 11.2 — Redaction Implementation

**File:** `gateway/app/redact.py`

```python
from app.detectors.base import Finding, EntityType

REDACTION_TOKENS = {
    EntityType.EMAIL: "[REDACTED_EMAIL]",
    EntityType.PHONE: "[REDACTED_PHONE]",
    EntityType.SSN: "[REDACTED_SSN]",
    EntityType.CREDIT_CARD: "[REDACTED_CC]",
    EntityType.PERSON_NAME: "[REDACTED_NAME]",
    EntityType.AWS_KEY: "[REDACTED_SECRET]",
    EntityType.GITHUB_TOKEN: "[REDACTED_SECRET]",
    EntityType.JWT: "[REDACTED_SECRET]",
    EntityType.PASSWORD: "[REDACTED_SECRET]",
    EntityType.GENERIC_PII: "[REDACTED_PII]",
    EntityType.GENERIC_SECRET: "[REDACTED_SECRET]",
}

def redact_text(text: str, findings: list[Finding]) -> tuple[str, dict[str, int]]:
    """
    Replace matched spans with redaction tokens.
    Returns (redacted_text, redaction_counts).

    Findings MUST be sorted by start position.
    Process in reverse order to preserve character offsets.
    """
    redaction_counts: dict[str, int] = {}
    # Sort by start position descending (process from end to preserve offsets)
    sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)

    redacted = text
    for finding in sorted_findings:
        token = REDACTION_TOKENS.get(finding.entity_type, "[REDACTED]")
        redacted = redacted[:finding.start] + token + redacted[finding.end:]

        entity_name = finding.entity_type.value
        redaction_counts[entity_name] = redaction_counts.get(entity_name, 0) + 1

    return redacted, redaction_counts
```

### 11.3 — Redaction Requirements

- **Never log raw matched text.** The `Finding.matched_text` field is used only during processing and must not be persisted to the database.
- **Preserve readability.** The redacted text should still make sense to the LLM. `"Summarize this email from [REDACTED_EMAIL] about project Alpha"` is a valid prompt the model can respond to.
- **Count statistics.** The `redaction_counts` dict is stored in the audit log and returned in the `ppg` metadata.

---

## 12. Audit Logging

### 12.1 — What Gets Logged

Every request through the gateway produces one audit record. The record contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Primary key, auto-generated |
| `request_id` | UUID | Unique ID for this request, returned in `ppg` metadata |
| `created_at` | TIMESTAMP | When the request was received |
| `user_id` | TEXT | From request's `user` field or `X-User-Id` header. Default: `"anonymous"` |
| `model` | TEXT | The model requested (e.g., `"llama3.2:3b"`) |
| `input_decision` | TEXT | `ALLOW`, `MASK`, or `BLOCK` |
| `output_decision` | TEXT | `ALLOW`, `MASK`, or `null` (if blocked, no output to scan) |
| `policy_id` | TEXT | Which policy was applied (e.g., `"default-v1"`) |
| `reasons` | JSONB | List of human-readable reason strings |
| `input_redactions` | JSONB | `{"EMAIL": 1, "PHONE": 2}` — counts by type |
| `output_redactions` | JSONB | Same format, for output scanning |
| `prompt_redacted` | TEXT | The fully redacted prompt text (safe to store) |
| `response_redacted` | TEXT | The fully redacted response text (safe to store), null if blocked |
| `prompt_hash` | TEXT | SHA-256 hash of the redacted prompt |
| `detection_latency_ms` | INTEGER | Time spent in detection pipeline |
| `llm_latency_ms` | INTEGER | Time spent waiting for LLM response, null if blocked |
| `total_latency_ms` | INTEGER | Total request processing time |

### 12.2 — Audit Writer Implementation

**File:** `gateway/app/audit.py`

```python
import hashlib
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import AuditLog

async def write_audit_record(
    session: AsyncSession,
    request_id: uuid.UUID,
    user_id: str,
    model: str,
    input_decision: str,
    output_decision: str | None,
    policy_id: str,
    reasons: list[str],
    input_redactions: dict[str, int],
    output_redactions: dict[str, int] | None,
    prompt_redacted: str,
    response_redacted: str | None,
    detection_latency_ms: int,
    llm_latency_ms: int | None,
    total_latency_ms: int,
) -> None:
    prompt_hash = hashlib.sha256(prompt_redacted.encode()).hexdigest()

    record = AuditLog(
        id=uuid.uuid4(),
        request_id=request_id,
        created_at=datetime.now(timezone.utc),
        user_id=user_id,
        model=model,
        input_decision=input_decision,
        output_decision=output_decision,
        policy_id=policy_id,
        reasons=reasons,
        input_redactions=input_redactions,
        output_redactions=output_redactions or {},
        prompt_redacted=prompt_redacted,
        response_redacted=response_redacted,
        prompt_hash=prompt_hash,
        detection_latency_ms=detection_latency_ms,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    session.add(record)
    await session.commit()
```

### 12.3 — Critical Rule

**NEVER store raw PII or secrets in the database.** The `prompt_redacted` and `response_redacted` fields must contain only the redacted versions with `[REDACTED_*]` tokens. If the decision is `BLOCK`, store the redacted prompt but set `response_redacted` to `null`.

---

## 13. Admin Dashboard (Streamlit)

### 13.1 — Purpose

The admin dashboard provides a visual interface to browse and inspect audit records. It exists primarily for:
1. Demo/screenshots in the README.
2. Quick review of gateway activity without querying the database directly.

### 13.2 — Pages

**Page 1: Overview Dashboard**

- Total requests (all time)
- Requests by decision (pie chart: ALLOW / MASK / BLOCK)
- Requests over time (line chart, last 24h / 7d / 30d)
- Top redaction types (bar chart)

**Page 2: Request Log**

A filterable table showing all audit records:

| Column | Source |
|--------|--------|
| Time | `created_at` |
| User | `user_id` |
| Model | `model` |
| Input Decision | `input_decision` (color-coded: green/yellow/red) |
| Output Decision | `output_decision` |
| Redactions | Summary of `input_redactions` |
| Latency | `total_latency_ms` |

**Filters:**
- Date range (date picker)
- User ID (text input)
- Decision type (multiselect: ALLOW / MASK / BLOCK)
- Model (dropdown)

**Page 3: Request Detail (click from log)**

When a user clicks a row in the request log, show:

- All fields from the audit record
- Redacted prompt (displayed in a code block)
- Redacted response (if exists)
- Reasons (bullet list)
- Redaction counts (table)
- Latency breakdown (detection / LLM / total)
- Prompt hash (for verification)

### 13.3 — Implementation

**File:** `admin/streamlit_app.py`

Use `sqlalchemy` (sync, since Streamlit doesn't do async) to connect to Postgres and query the `audit_log` table. Use `st.dataframe()` for the table, `st.bar_chart()` / `st.line_chart()` or `plotly` for charts.

```python
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ppg:ppg@db:5432/ppg")
engine = create_engine(DATABASE_URL)

st.set_page_config(page_title="SentinelLM Admin", layout="wide")
st.title("🛡️ SentinelLM — Admin Dashboard")

# --- Filters ---
col1, col2, col3 = st.columns(3)
with col1:
    decision_filter = st.multiselect("Decision", ["ALLOW", "MASK", "BLOCK"], default=["ALLOW", "MASK", "BLOCK"])
with col2:
    user_filter = st.text_input("User ID", "")
with col3:
    limit = st.number_input("Max rows", value=100, min_value=10, max_value=1000)

# --- Query ---
query = text("""
    SELECT request_id, created_at, user_id, model, input_decision,
           output_decision, reasons, input_redactions, total_latency_ms
    FROM audit_log
    WHERE input_decision = ANY(:decisions)
    ORDER BY created_at DESC
    LIMIT :limit
""")

with engine.connect() as conn:
    df = pd.read_sql(query, conn, params={"decisions": decision_filter, "limit": limit})

st.dataframe(df, use_container_width=True)

# --- Detail view ---
if not df.empty:
    selected_id = st.selectbox("Select request to inspect", df["request_id"])
    detail_query = text("SELECT * FROM audit_log WHERE request_id = :rid")
    with engine.connect() as conn:
        detail = pd.read_sql(detail_query, conn, params={"rid": selected_id})

    if not detail.empty:
        row = detail.