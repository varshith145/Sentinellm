#!/usr/bin/env bash
# Run Triton Inference Server (CPU, onnxruntime backend) for SentinelLM's
# semantic detector model.
#
# Ports are remapped off Triton's defaults (8000/8001/8002) because the
# SentinelLM gateway itself owns localhost:8000 — every script and doc in
# this repo assumes that. Triton's HTTP/gRPC/metrics land on 8100/8101/8102
# here instead.
#
# Usage:
#   python model/export_onnx.py       # if model/onnx/ doesn't exist yet
#   python triton_deploy/build_model_repo.py # copies model.onnx into triton_deploy/models/
#   ./triton_deploy/run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/models/sentinellm/1/model.onnx" ]; then
  echo "triton_deploy/models/sentinellm/1/model.onnx not found." >&2
  echo "Run: python model/export_onnx.py && python triton_deploy/build_model_repo.py" >&2
  exit 1
fi

docker run --rm \
  -p 8100:8000 \
  -p 8101:8001 \
  -p 8102:8002 \
  -v "$SCRIPT_DIR/models:/models" \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models
