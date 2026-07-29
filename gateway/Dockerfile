FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for Presidio
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY . .

# Copy policies
COPY policies/ /app/policies/

# Expose port
EXPOSE 8000

# Run with uvicorn. Worker count is env-driven, not hardcoded — detection is
# CPU-bound (spaCy + DistilBERT), so throughput scales with worker
# *processes*, not just asyncio concurrency (see README's Design Notes).
# Default 4 is what was actually measured (bench/results.md); override with
# SENTINELLM_WORKERS on constrained hardware. Postgres (this compose stack's
# DB) handles concurrent multi-process startup fine — this is not safe to
# copy onto the root Dockerfile's single-container SQLite Space.
ENV SENTINELLM_WORKERS=4
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${SENTINELLM_WORKERS}"]
