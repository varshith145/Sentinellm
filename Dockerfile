# SentinelLM — Hugging Face Spaces (Docker SDK) image.
#
# A single container that runs ONLY the FastAPI gateway in demo mode on port
# 7860. No Ollama, no Postgres: detection runs via /scan, audit logging falls
# back to SQLite, and the fine-tuned DistilBERT is pulled from the HF Hub.
#
# Build/run locally exactly as Spaces does:
#   docker build -t sentinellm-space .
#   docker run -p 7860:7860 sentinellm-space
FROM python:3.11-slim

# System deps Presidio / torch may need at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as a non-root user (UID 1000) with a locked-down home,
# so all library caches must point at writable dirs or transformers/spaCy crash.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HUB_CACHE=/tmp/hf_cache \
    XDG_CACHE_HOME=/tmp/cache \
    SENTINELLM_DEMO_MODE=true \
    SENTINELLM_SEMANTIC_MODEL_ENABLED=true \
    SENTINELLM_SEMANTIC_MODEL_ID=varshith145/sentinellm-pii-ner \
    SENTINELLM_DATABASE_URL=sqlite+aiosqlite:////tmp/sentinellm_audit.db \
    SENTINELLM_POLICY_PATH=/app/policies/default.yaml

WORKDIR /app

# Install Python deps first (better layer caching), then the spaCy model.
COPY --chown=user gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg

# App code (app/, policies/, etc.). The trained model dir is intentionally NOT
# copied — it is gitignored and loaded from the HF Hub at runtime.
COPY --chown=user gateway/ /app/

# Ensure cache dirs exist and are writable by the runtime user.
RUN mkdir -p /tmp/hf_cache /tmp/cache && chown -R user /tmp/hf_cache /tmp/cache /app

USER user

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
