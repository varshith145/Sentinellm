#!/usr/bin/env bash
# =============================================================================
# SentinelLM — One-command model training runner
#
# Runs the full training pipeline inside a Docker container so you don't
# need to install PyTorch or any ML dependencies locally.
#
# Usage:
#   chmod +x train.sh
#   ./train.sh
#
# What it does:
#   1. Builds a training Docker image (Python 3.11 + PyTorch + HuggingFace)
#   2. Runs prepare_dataset.py  → processes JSONL → HuggingFace Dataset
#   3. Runs train.py            → fine-tunes DistilBERT (5 epochs)
#   4. Runs evaluate.py         → saves metrics to docs/model_eval.json
#   5. Trained model lands in   model/trained/  (ready for the gateway)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="sentinellm-trainer"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      SentinelLM — Model Training Pipeline       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# --- Check Docker is available ---
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"
echo ""

# --- Build training image ---
echo "🔨 Building training Docker image..."
docker build \
    -f "${SCRIPT_DIR}/Dockerfile.train" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo ""
echo "📦 Running dataset preparation..."
docker run --rm \
    -v "${SCRIPT_DIR}/model:/app/model" \
    -v "${SCRIPT_DIR}/docs:/app/docs" \
    -w /app \
    "${IMAGE_NAME}" \
    python model/data/prepare_dataset.py

echo ""
echo "🧠 Starting DistilBERT fine-tuning (this takes ~5-10 min on CPU)..."
docker run --rm \
    -v "${SCRIPT_DIR}/model:/app/model" \
    -w /app \
    "${IMAGE_NAME}" \
    python model/train.py

echo ""
echo "📊 Running evaluation..."
mkdir -p "${SCRIPT_DIR}/docs"
docker run --rm \
    -v "${SCRIPT_DIR}/model:/app/model" \
    -v "${SCRIPT_DIR}/docs:/app/docs" \
    -w /app \
    "${IMAGE_NAME}" \
    python model/evaluate.py

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║          ✅ Training Complete!                   ║"
echo "║                                                  ║"
echo "║  Trained model:  model/trained/                  ║"
echo "║  Eval metrics:   docs/model_eval.json            ║"
echo "║                                                  ║"
echo "║  Restart the gateway to load the new model:      ║"
echo "║    docker compose restart gateway                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
