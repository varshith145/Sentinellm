"""
SentinelLM Gateway — FastAPI Application.

Main entry point for the AI gateway. Implements the full request/response
pipeline per PRD Section 7:

1. Receive OpenAI-compatible request
2. Extract scannable text from messages
3. Run three-pass detection (regex + Presidio + semantic) in parallel
4. Evaluate policy (ALLOW / MASK / BLOCK)
5. Redact if MASK
6. Write audit record
7. Block or forward to LLM backend
8. Scan output
9. Return response with ppg metadata
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.audit import write_audit_record
from app.config import settings
from app.db import async_session_factory, close_db, init_db
from app.detectors.orchestrator import DetectionOrchestrator
from app.detectors.regex import RegexDetector
from app.models import (
    ChatCompletionRequest,
    PPGMetadata,
    PolicyViolationResponse,
)
from app.policy import PolicyEngine
from app.redact import redact_text

logger = logging.getLogger("sentinellm")
logging.basicConfig(level=logging.INFO)

# --- Global instances (initialized at startup) ---
orchestrator: DetectionOrchestrator | None = None
policy_engine: PolicyEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize and cleanup resources."""
    global orchestrator, policy_engine

    logger.info("Starting SentinelLM Gateway...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize detectors
    detectors = []

    # Pass 1: Regex (always available)
    regex_detector = RegexDetector()
    detectors.append(regex_detector)
    logger.info("Regex detector initialized")

    # Pass 2: Presidio
    try:
        from app.detectors.presidio_detector import PresidioDetector

        presidio_detector = PresidioDetector()
        detectors.append(presidio_detector)
        logger.info("Presidio detector initialized")
    except Exception as e:
        logger.warning(f"Presidio detector unavailable: {e}")

    # Pass 3: Semantic (graceful fallback if model not trained)
    if settings.semantic_model_enabled:
        try:
            from app.detectors.semantic import SemanticDetector

            semantic_detector = SemanticDetector(model_path=settings.model_path)
            if semantic_detector.is_available:
                detectors.append(semantic_detector)
                logger.info("Semantic detector initialized")
            else:
                logger.info("Semantic detector: model not found, skipping")
        except Exception as e:
            logger.warning(f"Semantic detector unavailable: {e}")

    orchestrator = DetectionOrchestrator(detectors)
    logger.info(
        f"Detection orchestrator ready with: {orchestrator.get_active_detectors()}"
    )

    # Initialize policy engine
    policy_engine = PolicyEngine(policy_path=settings.policy_path)
    logger.info(f"Policy engine loaded: {policy_engine.policy_id}")

    logger.info("SentinelLM Gateway ready!")

    yield

    # Cleanup
    await close_db()
    logger.info("SentinelLM Gateway shut down")


# --- FastAPI App ---

app = FastAPI(
    title="SentinelLM Gateway",
    description=(
        "AI gateway that detects and redacts sensitive data in LLM traffic "
        "using a three-pass detection pipeline (regex + Presidio + semantic NER)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Health Check ---


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sentinellm-gateway",
        "detectors": orchestrator.get_active_detectors() if orchestrator else [],
        "policy_id": policy_engine.policy_id if policy_engine else None,
    }


# --- Main Chat Completions Endpoint ---


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    OpenAI-compatible chat completions endpoint with PII/secret scanning.

    Full pipeline: scan → policy → redact → audit → forward/block → output scan → return.
    """
    request_id = uuid.uuid4()
    start_time = time.time()

    # Extract user ID from request body or headers
    user_id = request.user or raw_request.headers.get("X-User-Id") or "anonymous"

    # ─── Step 1: Extract scannable text ───
    texts_to_scan: list[dict] = []
    for i, msg in enumerate(request.messages):
        if msg.content:
            texts_to_scan.append(
                {
                    "role": msg.role,
                    "index": i,
                    "content": msg.content,
                }
            )

    # ─── Step 2: Run three-pass detection on all messages ───
    detection_start = time.time()

    all_findings_by_index: dict[int, list] = {}
    for item in texts_to_scan:
        findings = await orchestrator.scan(item["content"])
        if findings:
            all_findings_by_index[item["index"]] = findings

    # Flatten all findings for policy evaluation
    all_findings = [f for findings in all_findings_by_index.values() for f in findings]

    detection_latency_ms = int((time.time() - detection_start) * 1000)

    # ─── Step 3: Policy decision ───
    input_decision = policy_engine.evaluate(all_findings)

    # ─── Step 4: Redact if MASK ───
    input_redactions: dict[str, int] = {}
    sanitized_messages = []

    for i, msg in enumerate(request.messages):
        if i in all_findings_by_index and input_decision.action in ("MASK", "BLOCK"):
            redacted_content, counts = redact_text(
                msg.content, all_findings_by_index[i]
            )
            input_redactions.update(counts)
            sanitized_messages.append(
                msg.model_copy(update={"content": redacted_content})
            )
        else:
            sanitized_messages.append(msg)

    # Build the redacted prompt for audit logging
    prompt_redacted = "\n".join(
        f"[{m.role}]: {m.content or ''}" for m in sanitized_messages
    )

    # ─── Step 5: Block or forward ───
    if input_decision.action == "BLOCK":
        total_latency_ms = int((time.time() - start_time) * 1000)

        # Write audit record
        async with async_session_factory() as session:
            await write_audit_record(
                session=session,
                request_id=request_id,
                user_id=user_id,
                model=request.model,
                input_decision="BLOCK",
                output_decision=None,
                policy_id=policy_engine.policy_id,
                reasons=input_decision.reasons,
                input_redactions=input_redactions,
                output_redactions=None,
                prompt_redacted=prompt_redacted,
                response_redacted=None,
                detection_latency_ms=detection_latency_ms,
                llm_latency_ms=None,
                total_latency_ms=total_latency_ms,
            )

        ppg = PPGMetadata(
            request_id=str(request_id),
            input_decision="BLOCK",
            output_decision=None,
            input_redactions=input_redactions,
            output_redactions={},
            policy_id=policy_engine.policy_id,
            detectors_used=orchestrator.get_active_detectors(),
            latency_ms={
                "detection": detection_latency_ms,
                "llm": None,
                "total": total_latency_ms,
            },
        )

        error_response = PolicyViolationResponse(
            error={
                "message": (
                    "Request blocked by SentinelLM policy. "
                    "Sensitive content detected that cannot be forwarded to the LLM."
                ),
                "type": "policy_violation",
                "code": "content_blocked",
                "reasons": input_decision.reasons,
            },
            ppg=ppg,
        )

        return JSONResponse(status_code=403, content=error_response.model_dump())

    # ─── Step 6: Forward to LLM backend ───
    llm_start = time.time()

    # Build sanitized request payload
    sanitized_payload = request.model_dump(exclude_none=True)
    sanitized_payload["messages"] = [
        m.model_dump(exclude_none=True) for m in sanitized_messages
    ]

    # Determine backend URL
    if settings.llm_backend == "openai":
        backend_url = settings.openai_base_url
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
    else:
        backend_url = settings.ollama_base_url
        headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient() as client:
            llm_response = await client.post(
                f"{backend_url}/v1/chat/completions",
                json=sanitized_payload,
                headers=headers,
                timeout=settings.llm_timeout,
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "LLM backend timed out", "type": "timeout"}},
        )
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content={
                "error": {
                    "message": f"LLM backend error: {e.response.text}",
                    "type": "backend_error",
                }
            },
        )
    except httpx.ConnectError:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Cannot connect to LLM backend",
                    "type": "connection_error",
                }
            },
        )

    llm_latency_ms = int((time.time() - llm_start) * 1000)

    # ─── Step 7: Output scanning ───
    output_redactions: dict[str, int] = {}
    output_decision_action = "ALLOW"

    if llm_data.get("choices"):
        for choice in llm_data["choices"]:
            response_content = choice.get("message", {}).get("content", "")
            if response_content:
                output_findings = await orchestrator.scan(response_content)
                if output_findings:
                    output_decision = policy_engine.evaluate(
                        output_findings, is_output=True
                    )
                    if output_decision.action in ("MASK", "BLOCK"):
                        output_decision_action = output_decision.action
                        redacted_output, out_counts = redact_text(
                            response_content, output_decision.findings
                        )
                        choice["message"]["content"] = redacted_output
                        for k, v in out_counts.items():
                            output_redactions[k] = output_redactions.get(k, 0) + v

    total_latency_ms = int((time.time() - start_time) * 1000)

    # Build response text for audit
    response_redacted = "\n".join(
        c.get("message", {}).get("content", "") for c in llm_data.get("choices", [])
    )

    # ─── Step 8: Write audit record ───
    async with async_session_factory() as session:
        await write_audit_record(
            session=session,
            request_id=request_id,
            user_id=user_id,
            model=request.model,
            input_decision=input_decision.action,
            output_decision=output_decision_action,
            policy_id=policy_engine.policy_id,
            reasons=input_decision.reasons,
            input_redactions=input_redactions,
            output_redactions=output_redactions,
            prompt_redacted=prompt_redacted,
            response_redacted=response_redacted,
            detection_latency_ms=detection_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
        )

    # ─── Step 9: Return response with ppg metadata ───
    ppg = PPGMetadata(
        request_id=str(request_id),
        input_decision=input_decision.action,
        output_decision=output_decision_action,
        input_redactions=input_redactions,
        output_redactions=output_redactions,
        policy_id=policy_engine.policy_id,
        detectors_used=orchestrator.get_active_detectors(),
        latency_ms={
            "detection": detection_latency_ms,
            "llm": llm_latency_ms,
            "total": total_latency_ms,
        },
    )

    llm_data["ppg"] = ppg.model_dump()

    return JSONResponse(content=llm_data)
