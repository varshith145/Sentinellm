#!/usr/bin/env python3
"""
Compare Triton client transport overhead: HTTP vs gRPC.

Sends the same fixed text, single request at a time (no concurrency — that's
deliberate here, unlike verify_client.py's dynamic_batching probe: this
script isolates per-call client/transport overhead, which concurrency would
obscure by letting Triton's own batching hide it), through both
tritonclient.http and tritonclient.grpc, sequentially, and reports the
per-call latency distribution for each.

gRPC is what gateway/app/detectors/semantic.py's triton backend actually
uses in production — this script is what justifies that choice with a
number, not just "gRPC is generally faster than HTTP/REST".

Usage:
    ./triton_deploy/run.sh                     # in another terminal
    python triton_deploy/compare_http_grpc.py [--n 200]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
ONNX_DIR = REPO_ROOT / "model" / "onnx"

TEXT = "My email is jane.doe@example.com, please follow up."
MODEL_NAME = "sentinellm"


def _encode(tokenizer):
    enc = tokenizer(TEXT, truncation=True, max_length=128)
    input_ids = np.array([enc["input_ids"]], dtype=np.int64)
    attention_mask = np.array([enc["attention_mask"]], dtype=np.int64)
    return input_ids, attention_mask


def run_http(url: str, n: int, input_ids, attention_mask) -> list[float]:
    client = httpclient.InferenceServerClient(url=url)
    latencies = []
    for _ in range(n):
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        outputs = [httpclient.InferRequestedOutput("logits")]

        t0 = time.perf_counter()
        client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def run_grpc(url: str, n: int, input_ids, attention_mask) -> list[float]:
    client = grpcclient.InferenceServerClient(url=url)
    latencies = []
    for _ in range(n):
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        outputs = [grpcclient.InferRequestedOutput("logits")]

        t0 = time.perf_counter()
        client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = max(0, min(len(values) - 1, round(p * (len(values) - 1))))
    return values[idx]


def summarize(name: str, latencies: list[float]) -> None:
    print(
        f"{name:>5}: mean {sum(latencies) / len(latencies):.3f}ms | "
        f"p50 {percentile(latencies, 0.50):.3f}ms | "
        f"p95 {percentile(latencies, 0.95):.3f}ms | "
        f"p99 {percentile(latencies, 0.99):.3f}ms | "
        f"min {min(latencies):.3f}ms | max {max(latencies):.3f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-url", default="localhost:8100")
    parser.add_argument("--grpc-url", default="localhost:8101")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
    input_ids, attention_mask = _encode(tokenizer)

    print(f"Warming up ({args.warmup} requests per transport)...")
    run_http(args.http_url, args.warmup, input_ids, attention_mask)
    run_grpc(args.grpc_url, args.warmup, input_ids, attention_mask)

    print(f"\nSequential single-request latency, n={args.n} per transport:\n")
    http_latencies = run_http(args.http_url, args.n, input_ids, attention_mask)
    summarize("HTTP", http_latencies)

    grpc_latencies = run_grpc(args.grpc_url, args.n, input_ids, attention_mask)
    summarize("gRPC", grpc_latencies)

    http_mean = sum(http_latencies) / len(http_latencies)
    grpc_mean = sum(grpc_latencies) / len(grpc_latencies)
    print(
        f"\ngRPC is {http_mean / grpc_mean:.2f}x faster than HTTP on mean "
        f"per-call latency ({http_mean:.3f}ms -> {grpc_mean:.3f}ms), same "
        "process, same machine, interleaved by transport (HTTP block then "
        "gRPC block, both after a warmup) — not concurrent, so this isolates "
        "client/transport overhead specifically, not queuing or batching."
    )


if __name__ == "__main__":
    main()
