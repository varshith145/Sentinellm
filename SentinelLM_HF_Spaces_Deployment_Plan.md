# SentinelLM → Hugging Face Spaces: Deployment Handoff Plan

**For:** Claude Cowork / Claude Code (the executing agent)
**Repo:** https://github.com/varshith145/sentinellm (public)
**Goal:** Take SentinelLM from a local `docker compose` project to a **live, public, clickable demo** on Hugging Face Spaces (free CPU tier), so a non-technical recruiter can paste text into a browser and watch PII/secret detection work — without any local setup.

---

## 0. Read this first — context and the core constraint

SentinelLM is an AI security gateway. Today it runs locally as a **multi-service Docker Compose stack**:
- a **FastAPI gateway** (`gateway/`, port 8000) exposing an OpenAI-compatible `/v1/chat/completions` proxy,
- a **three-pass detection pipeline** (regex → Microsoft Presidio → fine-tuned DistilBERT NER) orchestrated with `asyncio.gather`,
- a **PostgreSQL** audit log,
- a **Streamlit dashboard** (port 8501),
- **Prometheus** metrics,
- an **Ollama** LLM backend reached at `host.docker.internal:11434`.

**The single most important fact about the target platform:** A free Hugging Face Space runs **ONE container**, exposes **ONE HTTP port** (default `7860`), and has **ephemeral disk** (resets on rebuild/sleep) and **16 GB RAM / 2 vCPU**. It will **not** run `docker-compose.yml`, it has **no Ollama**, and it has **no PostgreSQL service**.

Therefore the job is NOT "lift and shift the compose stack." The job is: **carve out the impressive, self-contained core (the detection pipeline) and package it as a single container that needs nothing external.**

**The guiding principle for every decision below:** the *detection pipeline is the product*. The LLM proxy and Postgres were always supporting cast. For a public demo we make the LLM and Postgres **optional**, and we put the detection pipeline front and center.

**Do NOT, at any point:** upgrade the Space to GPU hardware, or add persistent storage. Both cost money. The free CPU Basic tier (16 GB RAM) is sufficient and must remain the target. The model runs on CPU.

---

## 1. Investigate the repo before changing anything

**Goal:** Ground the plan in the actual code. Several later steps branch on what you find here. Do this fully before writing code.

**What to do — read these files and record the answers:**

1. `gateway/app/main.py` — How is the request pipeline structured? Specifically: is the detection step (running the three passes on a string) **separable** from the LLM-proxy step, or are they tangled together in one request handler? Find the function/object that runs detection and note its exact call signature.
2. `gateway/app/detectors/orchestrator.py` — What is the public method that takes a string and returns findings? Record its exact name, parameters, and return type (e.g., a list of `Finding` objects with `entity_type`, `start`, `end`, `confidence`, `detector`). This is the function the new demo endpoint will call.
3. `gateway/app/detectors/base.py` — Note the `EntityType` enum and the `Finding` dataclass shape, so output can be serialized to JSON.
4. `gateway/app/policy.py` — How is a policy decision (ALLOW/MASK/BLOCK) computed from findings? Record the entry function.
5. `gateway/app/redact.py` — How is masked text produced from findings? Record the entry function.
6. `gateway/app/config.py` — List every setting and its env var name and default. Confirm settings are read from environment (pydantic-settings). Note especially: LLM backend selection, DB connection, model path, semantic-model-enabled flag.
7. `gateway/app/db.py` — What database driver/URL format does it use? Is it hardcoded to PostgreSQL, or can the connection string be swapped? Note whether it uses SQLAlchemy async.
8. `gateway/app/detectors/semantic.py` — How does it load the DistilBERT model? From a local path (`SENTINELLM_MODEL_PATH=/app/model/trained`) or from a HF model id? Record the load call.
9. The **gateway Dockerfile** — `docker-compose.yml` must reference a Dockerfile for the gateway service. Find it (likely `gateway/Dockerfile` or similar). Read it. This is the starting point for the Spaces Dockerfile.
10. `requirements.txt` (or equivalent in `gateway/`) — list dependencies. Note heavy ones: `torch`, `transformers`, `presidio-analyzer`, `spacy`, and which spaCy model (`en_core_web_lg`).
11. Check: **is the trained model (`model/trained/`) committed to git, or is it gitignored/generated?** (It is almost certainly NOT in the repo because it's large.) This determines Step 6.

**How to verify this step is done:** You can write one paragraph describing exactly which function to call to run detection on a string and get back findings + a policy decision + redacted text, naming the real functions. If you cannot, re-read until you can.

---

## 2. Confirm the target architecture (the shape we're building)

**Goal:** Lock the design so the rest is mechanical.

**Decision (use this unless investigation reveals a blocker):**
Build a **single Docker container, "Docker SDK" Hugging Face Space**, that runs the **FastAPI gateway only**, listening on **port 7860**, with:
- a **new `POST /scan` endpoint** that runs the detection pipeline on text and returns findings + decision + redacted text, **calling no LLM** (this is the demo's engine — see Step 3);
- the existing `/v1/chat/completions` proxy kept, but made **graceful when no LLM backend is configured** (see Step 4);
- **audit logging made optional with a SQLite fallback**, so no external Postgres is needed (see Step 5);
- the **full three-pass detector running**, including the fine-tuned DistilBERT, because 16 GB RAM is plenty (see Step 6);
- a **minimal HTML demo page served at `/`** so a recruiter has something to click (see Step 8).

**Why this shape:**
- One container = matches what a free Space can run.
- `/scan` with no LLM = removes the Ollama dependency, which is the hardest blocker, while still showcasing the detection pipeline (the actual differentiator).
- SQLite fallback = removes the Postgres dependency; the demo is fully self-contained.
- Keeping `/v1/chat/completions` and the gateway framing = preserves the "this is a real gateway" story for interviews; it simply runs in demo mode.

**Why NOT a Streamlit-SDK Space (the alternative):** A Streamlit Space would be simpler but throws away the FastAPI gateway and the OpenAI-compatible API story, which is a core part of what makes this project credible. We keep the API.

**Note on the Streamlit dashboard:** Do **not** try to run the Streamlit dashboard and FastAPI in the same free Space (two processes, two ports — fights the one-port model). Leave the dashboard out of the deployed demo for now. It stays in the repo for local use. (Optional future enhancement: a second, separate Streamlit Space — not in scope here.)

---

## 3. Add the `POST /scan` detect-only endpoint

**Goal:** Create the endpoint the demo calls. It runs the three-pass pipeline on a string and returns the full decision trace, **without invoking any LLM**. This is the heart of the public demo.

**Why:** The LLM proxy can't run on a free Space (no Ollama, and calling a hosted LLM would require keys/cost). But detection needs no LLM. Exposing detection on its own makes the demo instant, free, and dependency-light — and it shows the exact thing that's impressive.

**What to do:**
- Add a new route to the FastAPI app (in `gateway/app/main.py` or a small new router module).
- It accepts JSON `{ "text": "<user input>" }`.
- It calls the **real** detection orchestrator function identified in Step 1.2, then the policy function (1.4) and redaction function (1.5).
- It returns a JSON payload with: the decision (ALLOW/MASK/BLOCK), the list of findings (entity type, span text or offsets, confidence, which detector caught it), the redacted version of the text, and per-pass latency.

**Template (adapt names to the real functions found in Step 1):**
```python
# gateway/app/routers/scan.py  (or inline in main.py)
from fastapi import APIRouter
from pydantic import BaseModel
import time

# import the REAL orchestrator / policy / redact entrypoints found in Step 1
from app.detectors.orchestrator import Orchestrator          # adapt
from app.policy import evaluate_policy                        # adapt
from app.redact import redact_text                            # adapt

router = APIRouter()

class ScanRequest(BaseModel):
    text: str

@router.post("/scan")
async def scan(req: ScanRequest):
    t0 = time.perf_counter()
    findings = await Orchestrator().detect(req.text)           # adapt to real signature
    decision = evaluate_policy(findings)                       # adapt
    redacted = redact_text(req.text, findings) if decision != "BLOCK" else None  # adapt
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "decision": decision,
        "findings": [
            {
                "entity_type": str(f.entity_type),
                "text": req.text[f.start:f.end],
                "confidence": round(f.confidence, 3),
                "detector": f.detector,
            }
            for f in findings
        ],
        "redacted_text": redacted,
        "latency_ms": elapsed_ms,
    }
```
- Register the router on the app. Keep the existing `/v1/chat/completions` and `/docs` working.

**How to verify:** Locally, `curl -X POST localhost:8000/scan -H 'Content-Type: application/json' -d '{"text":"reach me at john dot smith at company dot com"}'` returns a MASK decision with a finding from the **semantic** detector. Also test `AKIAIOSFODNN7EXAMPLE` → BLOCK, and `we use AWS for cloud hosting` → ALLOW.

---

## 4. Make the LLM backend optional and graceful

**Goal:** Ensure the app **starts and runs without Ollama or any LLM configured**, so the container boots cleanly on Spaces.

**Why:** On a Space there is no Ollama. If the app crashes or hangs at startup because it can't reach `host.docker.internal:11434`, the whole Space fails. The `/scan` demo doesn't need an LLM at all — but the app must not depend on one to boot.

**What to do:**
- In `config.py` / startup: do **not** open a connection to the LLM backend at import/startup time. Connect lazily, only when `/v1/chat/completions` is actually called.
- If `/v1/chat/completions` is called and no backend is reachable/configured, return a clean `503` with a JSON message like `{"error": "LLM backend not configured in demo mode. Use POST /scan to test detection."}` — never an unhandled exception or a hang.
- Add a config flag (reuse or add an env var, e.g. `SENTINELLM_DEMO_MODE=true`) that the demo Space sets, making the LLM path explicitly optional.

**How to verify:** With no `OLLAMA`/`OPENAI` env vars set, the app boots, `/scan` works, `/docs` loads, and hitting `/v1/chat/completions` returns the clean 503 (not a crash).

---

## 5. Make audit logging optional with a SQLite fallback

**Goal:** Remove the hard dependency on PostgreSQL so the container is fully self-contained.

**Why:** A free Space has no Postgres service. Requiring an external managed Postgres adds friction and another thing that can sleep/fail. SQLite (a file on the container's ephemeral disk) is perfect for a demo — audit rows persist during a session and simply reset when the Space rebuilds, which is fine for a demo.

**What to do:**
- In `db.py`, make the database URL come entirely from an env var (e.g. `SENTINELLM_DATABASE_URL`).
- If that env var is **unset**, default to an **async SQLite** URL: `sqlite+aiosqlite:///./sentinellm_audit.db`.
- Add `aiosqlite` to requirements.
- Confirm the SQLAlchemy models/migrations create tables on startup if they don't exist (so a fresh SQLite file works with no manual migration).
- If audit writing fails for any reason, it must **not** break the request — log a warning and continue (audit is supporting cast, not critical path).

**How to verify:** Locally, unset the Postgres env var, start the gateway, run a few `/scan` calls, and confirm a `sentinellm_audit.db` file appears and rows are written, with no Postgres running.

---

## 6. Handle the fine-tuned DistilBERT model

**Goal:** Make the semantic detector's model available inside the Spaces container, loading correctly on CPU.

**Why:** The fine-tuned model (`model/trained/`) is almost certainly **not** committed to GitHub (too large). The container needs it at runtime. We have 16 GB RAM, so we run the **full** semantic pass (this is the differentiator — a recruiter seeing the model catch spelled-out SSNs is the whole point).

**What to do — preferred approach (host the model on the HF Hub):**
1. Confirm whether `model/trained/` exists locally (from a prior `bash train.sh`). If not, run `bash train.sh` once locally to produce it. (Training is ~12 epochs; do this on the local machine, not in the Space build.)
2. Create a **free public Hugging Face Model repo** (e.g. `varshith145/sentinellm-pii-ner`) and upload the contents of `model/trained/` to it using `huggingface_hub` (`upload_folder`). This is free and separate from the Space.
3. Change the semantic detector's model load (per Step 1.8) to load from the HF model id when a local path isn't present — e.g. read `SENTINELLM_MODEL_PATH`, and if it's not a real local dir, fall back to `from_pretrained("varshith145/sentinellm-pii-ner")`.

**Critical Spaces gotcha — writable cache dirs:** HF Spaces containers run as a **non-root user (UID 1000)** with a read-only-ish home. `transformers`/`huggingface_hub`/`spacy` try to write caches and will crash with permission errors unless cache dirs point somewhere writable. In the Dockerfile (Step 7), set:
```
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HUB_CACHE=/tmp/hf_cache
ENV XDG_CACHE_HOME=/tmp/cache
```
and ensure these dirs are created and writable.

**spaCy model:** Presidio needs `en_core_web_lg`. Install it explicitly in the Dockerfile (`python -m spacy download en_core_web_lg`). It's ~600 MB — fine on 16 GB RAM and 50 GB disk, but it makes the image large and the build slow; that's acceptable. (If build times become a real problem, `en_core_web_sm` is a fallback, but prefer `lg` for quality.)

**How to verify:** Locally, delete/rename `model/trained/`, set the model id env, and confirm the semantic detector loads from the Hub and still catches the obfuscated examples.

---

## 7. Create the Hugging Face Spaces configuration (Dockerfile + README metadata)

**Goal:** Produce the two files Spaces needs to build and run a Docker Space, listening on port 7860.

**Why:** Spaces' "Docker SDK" expects a `Dockerfile` at the repo root of the **Space** and a `README.md` with YAML frontmatter declaring the SDK and port. The app must listen on the port Spaces routes to.

**What to do:**

**7a. README frontmatter** — The Space repo's `README.md` must begin with YAML frontmatter:
```yaml
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
```
(Keep the rest of your existing README below this block, but update the Quickstart to point at the live demo — see Step 12.)

**7b. A Spaces Dockerfile** — Base it on the existing gateway Dockerfile, but: single stage that runs only the gateway, on port 7860, as user 1000, with writable caches and the spaCy model installed. Skeleton (adapt to the real entrypoint and paths):
```dockerfile
FROM python:3.11-slim

# System deps Presidio/torch may need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Non-root user (Spaces runs as UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HUB_CACHE=/tmp/hf_cache \
    XDG_CACHE_HOME=/tmp/cache \
    SENTINELLM_DEMO_MODE=true \
    SENTINELLM_SEMANTIC_MODEL_ENABLED=true

WORKDIR /app
COPY --chown=user gateway/requirements.txt .   # adapt path
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir aiosqlite \
    && python -m spacy download en_core_web_lg

COPY --chown=user gateway/ /app/                # adapt to real layout
RUN mkdir -p /tmp/hf_cache /tmp/cache && chown -R user /tmp/hf_cache /tmp/cache

USER user
EXPOSE 7860
# adapt module path to the real FastAPI app object
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**7c. Port** — Ensure nothing hardcodes port 8000 in a way that conflicts; the container must serve on 7860 (matching `app_port`).

**How to verify:** Step 9 (local container test).

---

## 8. Build a minimal demo front page at `/`

**Goal:** Give a recruiter something to click. When they open the Space URL, they should see a text box, a "Scan" button, and pre-loaded example buttons — and see results inline.

**Why:** A bare API at `/docs` is not a demo a non-technical person will engage with. The landing page must let anyone test detection in 10 seconds. The killer example (`reach me at john dot smith at company dot com`) must be one click away — that's the moment that sells the project.

**What to do:**
- Serve a single static HTML page at `GET /` from FastAPI (return `HTMLResponse`, or mount a `static/` dir).
- The page: a `<textarea>`, a "Scan" button that `fetch()`es `POST /scan`, and a results area that renders the decision (color-coded: green ALLOW / amber MASK / red BLOCK), the findings table (entity type, matched text, confidence, detector), the redacted text, and the latency.
- Add 3–4 **example buttons** that populate the textarea: one plain email (regex catch), one obfuscated email/SSN (semantic catch — the differentiator), one secret/AWS key (BLOCK), one benign "we use AWS for cloud hosting" (ALLOW, to show it's not trigger-happy).
- Keep it dependency-free (vanilla HTML/JS, inline CSS). No build step.
- Add a short note: "Demo mode — detection pipeline only; LLM proxy disabled. First load after inactivity may take ~30s to wake."

**How to verify:** Open `/` locally, click each example, confirm correct decisions render clearly.

---

## 9. Test the Spaces container locally before pushing

**Goal:** Catch failures on your machine, where debugging is fast, not in the Space build queue.

**Why:** Spaces builds are slower to iterate on. ~90% of deploy failures (port, permissions, cache dirs, model load, missing dep) are reproducible locally by building the exact Spaces Dockerfile.

**What to do:**
```bash
docker build -f Dockerfile -t sentinellm-space .
docker run -p 7860:7860 sentinellm-space
```
Then: open `http://localhost:7860/` (demo page), `http://localhost:7860/docs` (API), and run the three `/scan` curl tests from Step 3. Confirm: app boots with **no** Ollama and **no** Postgres, the semantic model loads (watch logs), and no permission/cache errors appear.

**How to verify:** All three example categories return correct decisions; logs are clean; memory stays well under 16 GB (`docker stats`).

---

## 10. Create the Space and deploy

**Goal:** Get it live at `https://huggingface.co/spaces/varshith145/sentinellm`.

**What to do:**
1. The user creates a Hugging Face account (if needed) and a new **Space**: SDK = **Docker**, hardware = **CPU basic (free)**. (The agent should pause and ask the user to do the account/Space creation and provide the Space URL + a write token, since account creation and auth are user actions.)
2. Add the Space as a git remote and push the repo (with the new Dockerfile and README frontmatter) to it, **or** use `huggingface_hub`'s upload. The Space repo and the GitHub repo are separate; you are pushing the same code to the Space's git.
3. If any secrets are needed later (e.g., an optional OpenAI key to enable the live proxy), set them in **Space → Settings → Secrets** (never commit them). For pure demo mode, no secrets are required.
4. Watch the **build logs** in the Space UI.

**How to verify:** The Space reaches "Running," and the public URL shows the demo page.

**Auth/account note for the agent:** Creating the HF account, generating a token, and clicking "create Space" are **user actions** — prompt the user to do these and hand back the Space URL and a token rather than attempting them autonomously.

---

## 11. Debug from logs — the likely failure modes

**Goal:** Fix the first deploy, which rarely works on attempt one (this is normal).

**Common failures and fixes:**
- **Permission denied writing cache / model** → cache env vars not set or dirs not writable (Step 6/7). Fix the `ENV` cache paths and `chown`.
- **App not reachable / "no app on port"** → not listening on `7860`, or `app_port` mismatch in README. Align them.
- **OOM during build or boot** → unlikely on 16 GB, but if it happens, confirm you're not loading the model multiple times; load once at startup and reuse.
- **Model download fails** → the HF model repo isn't public, or the id is wrong (Step 6). Make it public; verify the id.
- **spaCy model missing at runtime** → the `spacy download` line didn't run or wrong model name. Confirm it's in the Dockerfile and the name matches what Presidio expects.
- **Startup hang** → something connecting to Ollama/Postgres at boot (Step 4/5 not fully applied). Make those lazy/optional.

**How to verify:** Space status "Running," demo works for a fresh browser (test in an incognito window — that's the recruiter's experience).

---

## 12. Polish for recruiters + update the resume

**Goal:** Make the live project legible and turn it into a resume asset.

**What to do:**
- **README:** put the **live demo link at the very top**, big. Add a screenshot or short GIF of the demo catching the obfuscated SSN. Update the Quickstart to mention both the live demo and the local full-stack run.
- Keep the architecture diagram (it's excellent) but add a one-line note that the public demo runs detection-only in demo mode.
- **Resume bullet** (for the v2-deployed resume version): something like —
  > *"Deployed SentinelLM as a public, self-contained demo on Hugging Face Spaces: added a detect-only `/scan` API, made the LLM backend and Postgres optional (SQLite fallback), hosted the fine-tuned DistilBERT on the HF Hub, and packaged the full three-pass pipeline into a single CPU container — live and usable by anyone in-browser."*
- Tell the user to log the resulting applications as **v2-deployed** in the Notion tracker.

**How to verify:** A stranger opening only the GitHub README can reach a working demo in two clicks and understand what it does in 30 seconds.

---

## 13. Final verification checklist

- [ ] `/` shows a working demo page; example buttons return correct ALLOW/MASK/BLOCK.
- [ ] `/scan` works for plain PII (regex), obfuscated PII (semantic), secrets (BLOCK), benign (ALLOW).
- [ ] `/docs` loads (FastAPI Swagger).
- [ ] App boots with **no** Ollama and **no** Postgres; `/v1/chat/completions` returns a clean 503, not a crash.
- [ ] Semantic DistilBERT model loads from the HF Hub; obfuscated examples are caught.
- [ ] No permission/cache errors in Space logs.
- [ ] Space is on **free CPU basic** hardware; **no** GPU, **no** persistent storage added.
- [ ] Live URL works in an incognito window.
- [ ] README has the live link at top + a screenshot/GIF.
- [ ] The 244 tests still pass locally (`make test`) — the refactor (new `/scan`, optional LLM/DB) must not break them; add a couple of tests for `/scan` and the SQLite fallback.

---

## Summary of code changes the agent will make

1. **New** `/scan` endpoint (no LLM) — the demo engine.
2. **Modify** LLM path to be lazy + optional, with a clean 503 in demo mode.
3. **Modify** `db.py` for an env-driven URL with an async **SQLite fallback**; audit failures non-fatal.
4. **Modify** semantic model loading to fall back to a **HF Hub model id**.
5. **New** root `/` HTML demo page with example buttons.
6. **New** Spaces `Dockerfile` (single container, port 7860, UID 1000, writable caches, `en_core_web_lg`).
7. **Modify** `README.md` to add Spaces YAML frontmatter + live link + screenshot.
8. **New/Upload** a public HF Model repo with the fine-tuned DistilBERT.
9. **Add** `aiosqlite` to requirements; **add** tests for the new paths.

**The container must need nothing external. That is the whole game.**
