"""
Audit logging for SentinelLM.

Writes audit records to PostgreSQL for every gateway request.
Only stores redacted content — never raw PII or secrets.
"""

import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AuditLog


async def write_audit_record(
    session: AsyncSession,
    request_id: uuid.UUID,
    user_id: str,
    model: str,
    input_decision: str,
    output_decision: str | None,
    policy_id: str,
    reasons: list[str],
    input_redactions: dict[str, int],
    output_redactions: dict[str, int] | None,
    prompt_redacted: str,
    response_redacted: str | None,
    detection_latency_ms: int,
    llm_latency_ms: int | None,
    total_latency_ms: int,
) -> None:
    """
    Write an audit record to the database.

    This is called BEFORE forwarding to the LLM to ensure the record
    exists even if the LLM call fails. The record is updated after
    output scanning completes.

    Args:
        session: Async SQLAlchemy session.
        request_id: Unique identifier for this request.
        user_id: User who made the request.
        model: LLM model requested.
        input_decision: ALLOW, MASK, or BLOCK.
        output_decision: ALLOW, MASK, or None (if blocked).
        policy_id: Which policy was applied.
        reasons: Human-readable decision reasons.
        input_redactions: Redaction counts by entity type for input.
        output_redactions: Redaction counts by entity type for output.
        prompt_redacted: Fully redacted prompt text (safe to store).
        response_redacted: Fully redacted response text, or None if blocked.
        detection_latency_ms: Time spent in detection pipeline.
        llm_latency_ms: Time waiting for LLM, or None if blocked.
        total_latency_ms: Total request processing time.
    """
    prompt_hash = hashlib.sha256(prompt_redacted.encode()).hexdigest()

    record = AuditLog(
        id=uuid.uuid4(),
        request_id=request_id,
        created_at=datetime.now(timezone.utc),
        user_id=user_id,
        model=model,
        input_decision=input_decision,
        output_decision=output_decision,
        policy_id=policy_id,
        reasons=reasons,
        input_redactions=input_redactions,
        output_redactions=output_redactions or {},
        prompt_redacted=prompt_redacted,
        response_redacted=response_redacted,
        prompt_hash=prompt_hash,
        detection_latency_ms=detection_latency_ms,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    session.add(record)
    await session.commit()
