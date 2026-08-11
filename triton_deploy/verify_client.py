#!/usr/bin/env python3
"""
Verify the Triton deployment of SentinelLM's semantic detector.

Sends the same fixed set of texts to two places and asserts they agree:
  1. Triton Inference Server (this script, over HTTP) — the deployment
     under test.
  2. SemanticDetector(inference_backend="onnx") running in-process — the
     same ONNX graph, already proven to match the original PyTorch model to
     ~1.3e-05 max logit diff (see tests/test_onnx_parity.py). Agreement with
     it is the strongest available check without hand-asserting expected
     labels for each input.

Usage:
    pip install -r triton_deploy/requirements.txt
    python model/export_onnx.py           # if model/onnx/ doesn't exist yet
    python triton_deploy/build_model_repo.py
    ./triton_deploy/run.sh
    python triton_deploy/verify_client.py [--url localhost:8100]
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).parent.parent
ONNX_DIR = REPO_ROOT / "model" / "onnx"

sys.path.insert(0, str(REPO_ROOT / "gateway"))

TEST_CASES = [
    ("What's the weather like in San Francisco today?", "clean"),
    ("My email is jane.doe@example.com, please follow up.", "PII"),
    ("export AWS_KEY=AKIAIOSFODNN7EXAMPLE", "SECRET"),
    ("the password is hunter2 for the staging box", "SECRET"),
    ("call me at five five five one two three four", "PII"),
]

# A single request at a time never exercises dynamic_batching — Triton can
# only combine requests that share the same non-batch shape, so this phase
# deliberately pads every text to the same fixed length before firing them
# concurrently. This is the "watch out for" from the task: benchmarking
# dynamic_batching one request at a time makes it look like a no-op (or a
# pure latency cost from max_queue_delay_microseconds) instead of measuring
# what it's actually for.
BATCH_PROBE_TEXTS = [
    "My email is jane.doe@example.com, please follow up.",
    "Call me at 555-867-5309 when you get a chance.",
    "My SSN is 456-78-9012 for the background check.",
    "export AWS_KEY=AKIAIOSFODNN7EXAMPLE",
    "here's the github token: ghp_1234567890abcdefghijklmnopqrstuvwxyz",
    "What's the weather like in San Francisco today?",
    "Summarize the quarterly report in three bullet points.",
    "Write a haiku about autumn leaves.",
]
BATCH_PROBE_FIXED_LEN = 32
BATCH_PROBE_CONCURRENCY = 48  # 6x max_batch_size, so full-size batches are likely


def infer_triton(
    client: httpclient.InferenceServerClient, model_name: str, tokenizer, text: str
):
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")

    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)

    infer_inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
    ]
    infer_inputs[0].set_data_from_numpy(input_ids)
    infer_inputs[1].set_data_from_numpy(attention_mask)

    outputs = [httpclient.InferRequestedOutput("logits")]
    result = client.infer(model_name, inputs=infer_inputs, outputs=outputs)
    logits = result.as_numpy("logits")[0]  # drop batch dim -> (seq_len, num_labels)

    return logits, offset_mapping


def fetch_metric(metrics_url: str, metric_name: str, model_name: str) -> float:
    with urllib.request.urlopen(f"http://{metrics_url}/metrics", timeout=10) as resp:
        text = resp.read().decode()
    match = re.search(
        rf'{metric_name}\{{model="{model_name}",version="1"\}} ([\d.]+)', text
    )
    if not match:
        raise SystemExit(f"Metric {metric_name} not found for model {model_name}")
    return float(match[1])


def verify_dynamic_batching(
    url: str, metrics_url: str, model_name: str, tokenizer
) -> None:
    print(
        f"\nSending {BATCH_PROBE_CONCURRENCY} concurrent fixed-shape "
        f"(len={BATCH_PROBE_FIXED_LEN}) requests to probe dynamic_batching..."
    )

    execs_before = fetch_metric(metrics_url, "nv_inference_exec_count", model_name)
    requests_before = fetch_metric(
        metrics_url, "nv_inference_request_success", model_name
    )

    def one_request(text: str) -> None:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=BATCH_PROBE_FIXED_LEN,
            padding="max_length",
        )
        input_ids = np.array([enc["input_ids"]], dtype=np.int64)
        attention_mask = np.array([enc["attention_mask"]], dtype=np.int64)
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        outputs = [httpclient.InferRequestedOutput("logits")]
        # Each thread gets its own client — tritonclient's HTTP client isn't
        # meant to be shared for concurrent in-flight requests.
        httpclient.InferenceServerClient(url=url).infer(
            model_name, inputs=inputs, outputs=outputs
        )

    texts = (
        BATCH_PROBE_TEXTS * (BATCH_PROBE_CONCURRENCY // len(BATCH_PROBE_TEXTS) + 1)
    )[:BATCH_PROBE_CONCURRENCY]
    with ThreadPoolExecutor(max_workers=BATCH_PROBE_CONCURRENCY) as pool:
        list(pool.map(one_request, texts))

    execs_after = fetch_metric(metrics_url, "nv_inference_exec_count", model_name)
    requests_after = fetch_metric(
        metrics_url, "nv_inference_request_success", model_name
    )

    new_execs = execs_after - execs_before
    new_requests = requests_after - requests_before
    avg_batch = new_requests / new_execs if new_execs else 0

    print(
        f"  {new_requests:.0f} requests served in {new_execs:.0f} executions "
        f"-> average batch size {avg_batch:.2f}"
    )
    if avg_batch > 1.5:
        print(
            "  dynamic_batching is active (batches are forming, not 1:1 request:exec)."
        )
    else:
        raise SystemExit(
            "dynamic_batching does not appear to be batching under concurrency "
            f"(avg batch size {avg_batch:.2f}) — check config.pbtxt and that requests "
            "actually share a common shape."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:8100")
    parser.add_argument("--metrics-url", default="localhost:8102")
    parser.add_argument("--model-name", default="sentinellm")
    parser.add_argument(
        "--skip-batch-probe",
        action="store_true",
        help="Skip the concurrency-based dynamic_batching check",
    )
    args = parser.parse_args()

    from app.detectors.semantic import SemanticDetector

    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        raise SystemExit(f"Triton at {args.url} is not ready — is the container up?")
    if not client.is_model_ready(args.model_name):
        raise SystemExit(f"Model '{args.model_name}' is not ready on {args.url}")
    print(f"Triton server ready, model '{args.model_name}' ready.\n")

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
    reference = SemanticDetector(
        inference_backend="onnx", onnx_model_path=str(ONNX_DIR)
    )
    if not reference.is_available:
        raise SystemExit("Local ONNX reference detector failed to load.")

    all_match = True
    for text, expect_category in TEST_CASES:
        triton_logits, offset_mapping = infer_triton(
            client, args.model_name, tokenizer, text
        )
        triton_findings = reference._decode(text, triton_logits, offset_mapping)

        reference_logits, ref_offsets = reference._infer_onnx(text)
        reference_findings = reference._decode(text, reference_logits, ref_offsets)

        def norm(findings):
            return [(f.category.value, f.start, f.end) for f in findings]

        triton_norm = norm(triton_findings)
        reference_norm = norm(reference_findings)
        match = triton_norm == reference_norm
        all_match &= match

        found_categories = {c for c, _, _ in triton_norm} or {"none"}
        status = "OK" if match else "MISMATCH"
        print(f"[{status}] {text!r}")
        print(
            f"  expected category: {expect_category} | triton found: {sorted(found_categories)}"
        )
        if not match:
            print(f"  triton={triton_norm}")
            print(f"  local ={reference_norm}")

    print()
    if all_match:
        print(
            "All Triton inferences match the local ONNX reference. Deployment verified."
        )
    else:
        raise SystemExit(
            "Mismatch between Triton and local ONNX reference — see above."
        )

    if not args.skip_batch_probe:
        verify_dynamic_batching(args.url, args.metrics_url, args.model_name, tokenizer)


if __name__ == "__main__":
    main()
