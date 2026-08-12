"""
Integration checks for the Triton (gRPC) inference backend.

Two things this covers that the other backend tests don't:
  1. SemanticDetector(inference_backend="triton") produces the same findings
     as the onnx backend it's proxying to — Triton just serves that same
     ONNX graph over gRPC, so this proves the gRPC wire-up is actually
     correct, not just configured. Parametrized over both fp32 (sentinellm)
     and INT8 (sentinellm_int8) — the same 50 fixed inputs
     tests/test_onnx_parity.py uses for the PyTorch/ONNX check, pointed at
     the Triton client instead of a local ORT session, so this is the same
     rigor extended one hop further down the serving path rather than a
     weaker ad hoc check. BENCHMARKS.md's INT8 row previously only had
     aggregate eval F1 agreement (0.9462 both ways) as evidence the
     in-process and Triton INT8 paths serve the same graph — F1 is an
     aggregate and can hide per-example disagreements that happen to
     cancel out; this checks per-example agreement directly.
  2. The regex fast-path short-circuit (DetectionOrchestrator._is_conclusive)
     still skips the semantic pass when the backend is Triton — proven by
     reading Triton's own /metrics exec count before and after a
     regex-conclusive request, not by re-reading the orchestrator's code
     (which is backend-agnostic and already covered generically by
     tests/test_orchestrator.py::test_confident_fast_finding_skips_semantic
     — this test exists to catch a regression the generic one can't: one
     specific to how the triton backend is invoked). Runs the full golden
     path — create policy, trip it, check audit, check the Prometheus
     counter — with the semantic model actually enabled and Triton-backed,
     which tests/test_golden_path.py itself does not (it disables the
     semantic model entirely, so running it with
     SENTINELLM_INFERENCE_BACKEND=triton only proves the option doesn't
     crash startup).

Requires a running Triton server (triton_deploy/run.sh) serving the model
at the configured URL, and model/onnx/ to exist (tokenizer files — Triton
itself only serves the graph). Both are external state pytest can't set up,
so this file skips at collection time (not fails) when either is missing:
it's a local verification tool, not part of CI, which has no Triton
container available.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("tritonclient.grpc")

REPO_ROOT = Path(__file__).parent.parent
ONNX_DIR = REPO_ROOT / "model" / "onnx"
INT8_ONNX_DIR = REPO_ROOT / "model" / "onnx-int8"
TRITON_URL = os.environ.get("SENTINELLM_TEST_TRITON_URL", "localhost:8101")
TRITON_METRICS_URL = os.environ.get(
    "SENTINELLM_TEST_TRITON_METRICS_URL", "localhost:8102"
)
TRITON_MODEL_NAME = "sentinellm"

if not (ONNX_DIR / "model.onnx").exists():
    pytest.skip(
        "model/onnx/ not found — run `python model/export_onnx.py` first",
        allow_module_level=True,
    )

try:
    import tritonclient.grpc as grpcclient

    _client = grpcclient.InferenceServerClient(url=TRITON_URL)
    if not (_client.is_server_ready() and _client.is_model_ready(TRITON_MODEL_NAME)):
        raise RuntimeError("server or model not ready")
except Exception as e:  # noqa: BLE001 — intentional skip if Triton isn't reachable
    pytest.skip(
        f"Triton not reachable at {TRITON_URL} ({e}) — start it with "
        "`./triton_deploy/run.sh` first",
        allow_module_level=True,
    )


def _load_test_inputs() -> list[str]:
    """Same 50 fixed inputs as tests/test_onnx_parity.py — same source
    files, same counts, same order — so this is the identical fixture
    extended one hop further down the serving path, not a separately
    chosen (and potentially easier) test set."""
    texts: list[str] = []
    for filename, count in [
        ("synthetic_obfuscated.jsonl", 40),
        ("hard_negatives.jsonl", 10),
    ]:
        with open(REPO_ROOT / "model" / "data" / filename) as f:
            lines = [json.loads(line)["text"] for line in f]
        texts.extend(lines[:count])
    return texts


TEST_INPUTS = _load_test_inputs()
assert len(TEST_INPUTS) == 50
assert len(set(TEST_INPUTS)) == 50


@pytest.mark.parametrize(
    "onnx_dir,triton_model_name",
    [
        pytest.param(ONNX_DIR, "sentinellm", id="fp32"),
        pytest.param(
            INT8_ONNX_DIR,
            "sentinellm_int8",
            id="int8",
            marks=pytest.mark.xfail(
                reason=(
                    "1 of 50 fixed inputs disagrees between the in-process ONNX "
                    "Runtime 1.28.0 session and Triton 24.09's bundled ONNX "
                    "Runtime 1.19.2 (find its .so version with `docker exec "
                    "<container> find / -iname libonnxruntime*`) — deterministic "
                    "and reproducible (checked 3x), not flaky: max abs logit diff "
                    "0.988 on 'I set my password to capital P at sign double-u "
                    "zero r dee', which the in-process path calls SECRET at "
                    "confidence 0.9633 (near its own decision boundary) and "
                    "Triton doesn't call at all. Most likely cause: different "
                    "kernel implementations for the quantized ops (MatMulInteger "
                    "/ DynamicQuantizeLinear) between ORT versions nine minors "
                    "apart — INT8 requantization is more sensitive to small "
                    "fp32 accumulation differences than fp32 inference is, since "
                    "those differences can shift which side of a rounding "
                    "boundary a value lands on. fp32 (this same test, other "
                    "param) has zero mismatches on the same 50 inputs and the "
                    "same two ORT versions, consistent with that explanation. "
                    "strict=True: if this starts passing (e.g. after a Triton "
                    "image upgrade), that's worth knowing, not silently losing "
                    "the marker."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_triton_matches_onnx(onnx_dir: Path, triton_model_name: str):
    """Same graph, two transports (in-process ORT vs Triton/gRPC) — findings
    must be identical, not just close, on every one of the fixed 50 inputs.
    Run at both precisions: fp32 (sentinellm) confirms the gRPC wire-up
    itself is correct; int8 (sentinellm_int8) confirms the in-process and
    Triton INT8 paths serve the same quantized graph — previously only
    backed by aggregate eval F1 agreement (0.9462 both ways), which can't
    rule out per-example disagreements that happen to cancel out in
    aggregate."""
    if not (onnx_dir / "model.onnx").exists():
        pytest.skip(f"{onnx_dir}/model.onnx not found")
    if not _client.is_model_ready(triton_model_name):
        pytest.skip(
            f"Triton model '{triton_model_name}' not ready — run "
            "`python triton_deploy/build_model_repo.py` and restart Triton"
        )

    from app.detectors.semantic import SemanticDetector

    onnx_detector = SemanticDetector(
        inference_backend="onnx", onnx_model_path=str(onnx_dir)
    )
    triton_detector = SemanticDetector(
        inference_backend="triton",
        onnx_model_path=str(onnx_dir),
        triton_url=TRITON_URL,
        triton_model_name=triton_model_name,
    )
    assert onnx_detector.is_available
    assert triton_detector.is_available

    def norm(findings):
        return [(f.category.value, f.start, f.end) for f in findings]

    mismatches = []
    for text in TEST_INPUTS:
        onnx_logits, onnx_offsets = onnx_detector._infer_onnx(text)
        onnx_findings = norm(onnx_detector._decode(text, onnx_logits, onnx_offsets))

        triton_logits, triton_offsets = triton_detector._infer_triton(text)
        triton_findings = norm(
            triton_detector._decode(text, triton_logits, triton_offsets)
        )

        if onnx_findings != triton_findings:
            mismatches.append((text, onnx_findings, triton_findings))

    assert not mismatches, (
        f"Triton/onnx disagreement ({triton_model_name}) on {len(mismatches)} "
        f"of {len(TEST_INPUTS)} inputs: {mismatches}"
    )


def _fetch_exec_count() -> float:
    with urllib.request.urlopen(
        f"http://{TRITON_METRICS_URL}/metrics", timeout=10
    ) as resp:
        text = resp.read().decode()
    match = re.search(
        rf'nv_inference_exec_count\{{model="{TRITON_MODEL_NAME}",version="1"\}} ([\d.]+)',
        text,
    )
    return float(match[1]) if match else 0.0


def _block_count(metrics_text: str) -> float:
    match = re.search(
        r'sentinellm_requests_total\{decision="BLOCK"\} ([\d.]+)', metrics_text
    )
    return float(match.group(1)) if match else 0.0


@pytest.mark.asyncio
async def test_regex_short_circuit_skips_triton_call():
    """The golden path (tests/test_golden_path.py) with the semantic model
    actually enabled and backed by Triton — that file disables the semantic
    model entirely (`settings.semantic_model_enabled = False`), so running
    it with SENTINELLM_INFERENCE_BACKEND=triton only proves the option
    doesn't crash startup, not that the triton call path itself is safe to
    have wired in end-to-end (confirmed separately: it passes).

    Also the specific check this file exists for: a regex-conclusive
    request (plain email, clears orchestrator._CONCLUSIVE_CONFIDENCE) must
    never reach the semantic pass. Checked via Triton's own exec-count
    metric so a regression in how the triton backend is dispatched would
    actually fail this test, not pass silently because nothing exercised
    that path."""
    db_path = Path(tempfile.gettempdir()) / "sentinellm_test_triton.db"
    db_path.unlink(missing_ok=True)  # fixed filename — start from a clean db each run
    os.environ.setdefault("SENTINELLM_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    import httpx
    from app.config import settings
    from app.db import close_db
    from app.main import app, lifespan

    saved = {
        "inference_backend": settings.inference_backend,
        "semantic_model_enabled": settings.semantic_model_enabled,
        "onnx_model_path": settings.onnx_model_path,
        "triton_url": settings.triton_url,
        "triton_model_name": settings.triton_model_name,
    }
    settings.inference_backend = "triton"
    settings.semantic_model_enabled = True
    settings.onnx_model_path = str(ONNX_DIR)
    settings.triton_url = TRITON_URL
    settings.triton_model_name = TRITON_MODEL_NAME

    try:
        exec_before = _fetch_exec_count()

        async with lifespan(app):
            assert "semantic" in app.state.orchestrator.get_active_detectors(), (
                "triton-backed semantic detector failed to initialize — "
                "the short-circuit check below would pass vacuously"
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                # Force EMAIL -> BLOCK (default policy is MASK, which then
                # forwards to the LLM backend — not configured here and
                # irrelevant to what this test checks). BLOCK returns before
                # any LLM call, same as tests/test_golden_path.py.
                create_resp = await client.post(
                    "/api/v1/policies",
                    json={
                        "name": "Triton Test Block Rule",
                        "entity_type": "EMAIL",
                        "action": "block",
                        "min_confidence": 0.5,
                        "enabled": True,
                    },
                )
                assert create_resp.status_code == 201, create_resp.text

                block_count_before = _block_count((await client.get("/metrics")).text)

                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [
                            {
                                "role": "user",
                                "content": "My email is jane.doe@example.com, please follow up.",
                            }
                        ],
                    },
                )
                assert resp.status_code == 403, resp.text

                # Same assertions as tests/test_golden_path.py: audit
                # visibility and the Prometheus counter, so this is a true
                # golden path on the triton backend, not just a short-circuit
                # check with a request tacked on.
                audit_resp = await client.get(
                    "/api/v1/audit", params={"decision": "block", "limit": 10}
                )
                assert audit_resp.status_code == 200
                items = audit_resp.json()["items"]
                assert len(items) == 1, (
                    f"expected exactly one BLOCK record, got {len(items)}"
                )
                assert "EMAIL" in items[0]["entity_types"]
                assert "jane.doe@example.com" not in items[0]["request_preview"]

                block_count_after = _block_count((await client.get("/metrics")).text)
                assert block_count_after == block_count_before + 1

        exec_after = _fetch_exec_count()
        assert exec_after == exec_before, (
            f"Triton exec count moved ({exec_before} -> {exec_after}) on a "
            "regex-conclusive request — the semantic pass should have been "
            "short-circuited entirely, not just its result discarded."
        )
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)
        await close_db()
