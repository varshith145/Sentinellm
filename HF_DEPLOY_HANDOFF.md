# SentinelLM → Hugging Face Spaces: Your Remaining Steps

All the **code** is done and tested (see "What's already implemented" at the
bottom). What's left are the steps that need **your Hugging Face account** —
creating a token, uploading the model, creating the Space, and pushing. Follow
these in order. Total time: ~20 minutes plus one build.

---

## 1. Create an HF account + write token

1. Sign up / log in at https://huggingface.co.
2. Go to **Settings → Access Tokens → Create new token**, type **Write**.
3. Copy the token (starts with `hf_…`). You'll use it twice below.

Install the CLI and log in locally:

```bash
pip install -U huggingface_hub
huggingface-cli login   # paste the hf_ token
```

---

## 2. Upload the fine-tuned model to the Hub

The model (`model/trained/`) is gitignored and **not** pushed to the Space —
the container pulls it from the Hub at runtime (the id is already wired into the
code and Dockerfile as `varshith145/sentinellm-pii-ner`).

`model/trained/` contains dozens of training **checkpoints** you do *not* need.
Upload only the final model files at the root of that folder:

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo_id = "varshith145/sentinellm-pii-ner"
api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    folder_path="model/trained",
    repo_id=repo_id,
    repo_type="model",
    # Only the final model — skip all checkpoint-* dirs and optimizer state.
    allow_patterns=[
        "config.json", "model.safetensors",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt",
    ],
)
print("Uploaded to https://huggingface.co/" + repo_id)
PY
```

Verify the repo page lists `model.safetensors` (~265 MB) and the tokenizer
files, and that it is **public**.

> Using a different id? Override it without code changes via the Space env var
> `SENTINELLM_SEMANTIC_MODEL_ID`, and update the `ENV` line in the `Dockerfile`.

---

## 3. Create the Space

On https://huggingface.co → **New → Space**:

- **Owner / name:** `varshith145 / sentinellm`
- **SDK:** **Docker**
- **Hardware:** **CPU basic (free)** ← do *not* pick GPU or add persistent storage
- Visibility: **Public**

---

## 4. Push the repo to the Space

The Space is its own git repo, separate from GitHub. Push this code to it:

```bash
# from the repo root
git add -A && git commit -m "HF Spaces: /scan demo, optional LLM/DB, Dockerfile"

git remote add space https://huggingface.co/spaces/varshith145/sentinellm
git push space main      # use your hf_ token as the password if prompted
```

Because `model/trained/` is gitignored, only the code, the root `Dockerfile`,
and the README (with its new YAML frontmatter) get pushed — exactly what Spaces
needs. The build starts automatically.

---

## 5. Watch the build, then test

- Open the Space's **Logs** tab. The build is slow the first time (installs
  torch + `en_core_web_lg` + downloads the model). Expect several minutes.
- When status shows **Running**, open the Space URL in an **incognito window**
  (the recruiter's experience) and test:
  - `reach me at john dot smith at company dot com` → **MASK** (semantic catch)
  - `AKIAIOSFODNN7EXAMPLE` → **BLOCK**
  - `we use AWS for cloud hosting` → **ALLOW**

No secrets are required for demo mode. (If you later want the live LLM proxy,
add an `OPENAI`/`OLLAMA` key under **Space → Settings → Secrets** and set
`SENTINELLM_DEMO_MODE=false`.)

---

## 6. If the first build fails (common — don't panic)

| Symptom | Fix |
|---|---|
| Permission denied writing cache/model | cache env vars are set in the Dockerfile; confirm they survived your edits |
| "No app on port" / not reachable | README `app_port` must be `7860` and match `EXPOSE`/`CMD` |
| Model download fails | the HF model repo isn't public, or the id is wrong (Step 2) |
| spaCy model missing at runtime | `python -m spacy download en_core_web_lg` line must be in the Dockerfile |
| Startup hang | something reaching Ollama/Postgres at boot — shouldn't happen, demo mode + lazy LLM are already in place |

---

## 7. Polish + resume

- Put the **live Space URL** at the top of `README.md` (placeholder is already
  there: search for "add your Space URL here"). Add a screenshot/GIF of the
  obfuscated-SSN catch.
- Resume bullet (v2-deployed):
  > *Deployed SentinelLM as a public, self-contained demo on Hugging Face Spaces:
  > added a detect-only `/scan` API, made the LLM backend and Postgres optional
  > (SQLite fallback), hosted the fine-tuned DistilBERT on the HF Hub, and packaged
  > the full three-pass pipeline into a single CPU container — live and usable by
  > anyone in-browser.*
- Log the resulting applications as **v2-deployed** in your Notion tracker.

---

## What's already implemented (code, done & tested)

1. **`POST /scan`** — detect-only endpoint (regex + Presidio + semantic, no LLM):
   returns decision, findings, redacted text, latency. (`gateway/app/main.py`)
2. **Demo mode** — `SENTINELLM_DEMO_MODE=true` makes `/v1/chat/completions`
   return a clean `503`; the LLM is only contacted lazily, never at startup.
3. **SQLite fallback** — DB URL is env-driven and defaults to async SQLite;
   `db.py` now uses portable column types (was Postgres-only `JSONB`/`UUID`);
   audit writes are non-fatal. (`config.py`, `db.py`, `audit.py`)
4. **Semantic model HF-Hub fallback** — loads from `SENTINELLM_SEMANTIC_MODEL_ID`
   when no local `model/trained/` is present. (`detectors/semantic.py`, `config.py`)
5. **Demo landing page at `/`** — vanilla HTML/JS with example buttons and
   color-coded results. (`gateway/app/demo_page.py`)
6. **Spaces `Dockerfile`** (repo root) — single container, port 7860, UID 1000,
   writable caches, `en_core_web_lg`, model pulled from Hub.
7. **README YAML frontmatter** — `sdk: docker`, `app_port: 7860`, etc.
8. **Deps + tests** — added `aiosqlite`, `huggingface_hub`; new
   `tests/test_scan.py` and `tests/test_sqlite_fallback.py`. **253 tests pass.**

Local sanity check (no Postgres, no Ollama):

```bash
docker build -t sentinellm-space .
docker run -p 7860:7860 sentinellm-space
# open http://localhost:7860/  and  http://localhost:7860/docs
```
