"""
Console API — read/write surface for the policy-authoring / audit-search /
metrics console, kept under /api/v1/ so it never collides with the proxied
OpenAI-compatible traffic on /v1/*.

Auth: a single X-API-Key header, checked against settings.console_api_key.
Empty key (the default) disables auth, so local dev and pytest don't need a
header on every call — set SENTINELLM_CONSOLE_API_KEY in any deployment
reachable outside localhost.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.console_models import (
    AuditDetail,
    AuditRecord,
    Decision,
    DryRunFinding,
    DryRunRequest,
    DryRunResponse,
    PaginatedAudit,
    PaginatedPolicies,
    PolicyCreate,
    PolicyRule,
    PolicyUpdate,
    StatsSummary,
    TimeseriesPoint,
    _to_decision,
)
from app.db import AuditLog, Policy, get_session
from app.detectors.base import ENTITY_CATEGORY_MAP
from app.policy import evaluate_findings
from app.redact import redact_text
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy import Text, and_, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if settings.console_api_key and x_api_key != settings.console_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


async def deny_writes_if_read_only() -> None:
    """
    Blocks mutating policy routes on a read-only deployment (the public HF
    Spaces demo). GET routes never depend on this — only POST/PATCH/DELETE
    do — so a live public console stays fully browsable but not editable by
    strangers. Same shape as settings.demo_mode disabling the LLM proxy.
    """
    if settings.console_read_only:
        raise HTTPException(
            status_code=403,
            detail="This deployment is read-only — policy writes are disabled.",
        )


router = APIRouter(
    prefix="/api/v1", tags=["console"], dependencies=[Depends(require_api_key)]
)


# --- Cursor helpers (keyset pagination on (created_at, id), both descending) ---


def _encode_cursor(ts: datetime, id_: uuid.UUID) -> str:
    payload = json.dumps({"ts": ts.isoformat(), "id": str(id_)})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return datetime.fromisoformat(payload["ts"]), uuid.UUID(payload["id"])
    except Exception:  # noqa: BLE001 — any malformed input maps to 400, not a 500
        raise HTTPException(status_code=400, detail="Invalid cursor") from None


_DURATION_RE = re.compile(r"^(\d+)([smhd])$")


def _parse_duration(value: str, param_name: str) -> timedelta:
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {param_name} '{value}' — expected e.g. '5m', '1h', '24h', '7d'",
        )
    n, unit = int(m.group(1)), m.group(2)
    return {
        "s": timedelta(seconds=n),
        "m": timedelta(minutes=n),
        "h": timedelta(hours=n),
        "d": timedelta(days=n),
    }[unit]


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


# --- Policies ---


def _policy_to_rule(row: Policy) -> PolicyRule:
    return PolicyRule(
        id=str(row.id),
        name=row.name,
        description=row.description,
        entity_type=row.entity_type,
        category=row.category,
        action=_to_decision(row.action),
        min_confidence=row.min_confidence,
        enabled=row.enabled,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def _get_orchestrator(request: Request):
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Detection pipeline not ready")
    return orchestrator


async def _get_policy_engine(request: Request):
    policy_engine = getattr(request.app.state, "policy_engine", None)
    if policy_engine is None:
        raise HTTPException(status_code=503, detail="Policy engine not ready")
    return policy_engine


async def _get_policy_or_404(session: AsyncSession, policy_id: str) -> Policy:
    try:
        pid = uuid.UUID(policy_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Policy not found") from None
    row = await session.get(Policy, pid)
    if row is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return row


@router.get("/policies", response_model=PaginatedPolicies)
async def list_policies(
    enabled: bool | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(Policy)
    if enabled is not None:
        stmt = stmt.where(Policy.enabled.is_(enabled))
    if cursor:
        ts, cid = _decode_cursor(cursor)
        stmt = stmt.where(
            or_(
                Policy.created_at < ts,
                and_(Policy.created_at == ts, Policy.id < cid),
            )
        )
    stmt = stmt.order_by(Policy.created_at.desc(), Policy.id.desc()).limit(limit + 1)

    rows = (await session.execute(stmt)).scalars().all()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = _encode_cursor(last.created_at, last.id)

    return PaginatedPolicies(
        items=[_policy_to_rule(r) for r in rows], next_cursor=next_cursor
    )


@router.post(
    "/policies",
    response_model=PolicyRule,
    status_code=201,
    dependencies=[Depends(deny_writes_if_read_only)],
)
async def create_policy(
    body: PolicyCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    row = Policy(
        name=body.name,
        description=body.description,
        entity_type=body.entity_type.value,
        category=ENTITY_CATEGORY_MAP[body.entity_type].value,
        action=body.action.value.upper(),
        min_confidence=body.min_confidence,
        enabled=body.enabled,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)

    policy_engine = await _get_policy_engine(request)
    await policy_engine.reload(session)

    return _policy_to_rule(row)


@router.get("/policies/{policy_id}", response_model=PolicyRule)
async def get_policy(policy_id: str, session: AsyncSession = Depends(get_session)):
    row = await _get_policy_or_404(session, policy_id)
    return _policy_to_rule(row)


@router.patch(
    "/policies/{policy_id}",
    response_model=PolicyRule,
    dependencies=[Depends(deny_writes_if_read_only)],
)
async def update_policy(
    policy_id: str,
    body: PolicyUpdate,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    row = await _get_policy_or_404(session, policy_id)

    if body.name is not None:
        row.name = body.name
    if body.description is not None:
        row.description = body.description
    if body.entity_type is not None:
        row.entity_type = body.entity_type.value
        row.category = ENTITY_CATEGORY_MAP[body.entity_type].value
    if body.action is not None:
        row.action = body.action.value.upper()
    if body.min_confidence is not None:
        row.min_confidence = body.min_confidence
    if body.enabled is not None:
        row.enabled = body.enabled

    await session.commit()
    await session.refresh(row)

    policy_engine = await _get_policy_engine(request)
    await policy_engine.reload(session)

    return _policy_to_rule(row)


@router.delete(
    "/policies/{policy_id}",
    status_code=204,
    dependencies=[Depends(deny_writes_if_read_only)],
)
async def delete_policy(
    policy_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    row = await _get_policy_or_404(session, policy_id)
    await session.delete(row)
    await session.commit()

    policy_engine = await _get_policy_engine(request)
    await policy_engine.reload(session)


@router.post("/policies/{policy_id}/dry-run", response_model=DryRunResponse)
async def dry_run_policy(
    policy_id: str,
    body: DryRunRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """
    Evaluate `text` against the live policy rules with this rule's current
    field values overlaid — regardless of whether the row is enabled — so
    you can preview a rule you're about to save. Writes no audit record.
    """
    row = await _get_policy_or_404(session, policy_id)
    orchestrator = await _get_orchestrator(request)
    policy_engine = await _get_policy_engine(request)

    t0 = time.perf_counter()
    findings = await orchestrator.scan(body.text)

    temp_rules = dict(policy_engine.rules)
    temp_rules[row.entity_type] = {
        "action": row.action,
        "min_confidence": row.min_confidence,
    }
    decision = evaluate_findings(findings, temp_rules, policy_engine.output_scanning)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if decision.action in ("MASK", "BLOCK") and decision.findings:
        redacted_text, _ = redact_text(body.text, decision.findings)
    else:
        redacted_text = body.text

    return DryRunResponse(
        decision=_to_decision(decision.action),
        reasons=decision.reasons,
        findings=[
            DryRunFinding(
                entity_type=f.entity_type,
                category=f.category,
                text=f.matched_text,
                start=f.start,
                end=f.end,
                confidence=f.confidence,
                detector=f.detector,
            )
            for f in findings
        ],
        redacted_text=redacted_text,
        latency_ms=round(elapsed_ms, 1),
    )


# --- Audit ---


def _effective_decision(input_decision: str, output_decision: str | None) -> Decision:
    priority = {"ALLOW": 0, "MASK": 1, "BLOCK": 2}
    if input_decision == "BLOCK":
        return Decision.block
    effective = input_decision
    if output_decision and priority.get(output_decision, 0) > priority.get(
        effective, 0
    ):
        effective = output_decision
    return _to_decision(effective)


def _audit_to_record(row: AuditLog) -> AuditRecord:
    entity_types = sorted(
        set((row.input_redactions or {}).keys())
        | set((row.output_redactions or {}).keys())
    )
    return AuditRecord(
        id=str(row.id),
        ts=row.created_at,
        decision=_effective_decision(row.input_decision, row.output_decision),
        matched_rule_ids=entity_types,
        latency_ms=float(row.total_latency_ms),
        detector_confidence=None,
        entity_types=entity_types,
        request_preview=(row.prompt_redacted or "")[:200],
        tenant_id=row.user_id,
    )


@router.get("/audit", response_model=PaginatedAudit)
async def list_audit(
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = Query(default=None),
    decision: Decision | None = Query(default=None),
    rule_id: str | None = Query(
        default=None,
        description="Entity type that fired (e.g. AWS_KEY) — see AuditRecord.matched_rule_ids.",
    ),
    q: str | None = Query(
        default=None, description="Free-text search over redacted content"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    stmt = select(AuditLog)

    if from_ is not None:
        stmt = stmt.where(AuditLog.created_at >= from_)
    if to is not None:
        stmt = stmt.where(AuditLog.created_at <= to)
    if decision is not None:
        stmt = stmt.where(AuditLog.input_decision == decision.value.upper())
    if rule_id:
        needle = f'%"{rule_id}"%'
        stmt = stmt.where(
            or_(
                cast(AuditLog.input_redactions, Text).like(needle),
                cast(AuditLog.output_redactions, Text).like(needle),
            )
        )
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                AuditLog.prompt_redacted.like(like),
                AuditLog.response_redacted.like(like),
            )
        )
    if cursor:
        ts, cid = _decode_cursor(cursor)
        stmt = stmt.where(
            or_(
                AuditLog.created_at < ts,
                and_(AuditLog.created_at == ts, AuditLog.id < cid),
            )
        )

    stmt = stmt.order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(
        limit + 1
    )

    rows = (await session.execute(stmt)).scalars().all()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = _encode_cursor(last.created_at, last.id)

    return PaginatedAudit(
        items=[_audit_to_record(r) for r in rows], next_cursor=next_cursor
    )


@router.get("/audit/{record_id}", response_model=AuditDetail)
async def get_audit_record(
    record_id: str, session: AsyncSession = Depends(get_session)
):
    try:
        rid = uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Audit record not found") from None

    row = await session.get(AuditLog, rid)
    if row is None:
        raise HTTPException(status_code=404, detail="Audit record not found")

    base = _audit_to_record(row)
    return AuditDetail(
        **base.model_dump(),
        model=row.model,
        input_decision=_to_decision(row.input_decision),
        output_decision=_to_decision(row.output_decision)
        if row.output_decision
        else None,
        reasons=row.reasons or [],
        input_redactions=row.input_redactions or {},
        output_redactions=row.output_redactions or {},
        prompt_redacted=row.prompt_redacted,
        response_redacted=row.response_redacted,
        detection_latency_ms=row.detection_latency_ms,
        llm_latency_ms=row.llm_latency_ms,
        total_latency_ms=row.total_latency_ms,
        policy_id=row.policy_id,
    )


# --- Stats ---


@router.get("/stats/summary", response_model=StatsSummary)
async def stats_summary(
    window: str = Query(default="24h"),
    session: AsyncSession = Depends(get_session),
):
    delta = _parse_duration(window, "window")
    since = datetime.now(timezone.utc) - delta

    stmt = select(
        AuditLog.input_decision,
        AuditLog.output_decision,
        AuditLog.total_latency_ms,
        AuditLog.input_redactions,
        AuditLog.output_redactions,
    ).where(AuditLog.created_at >= since)
    rows = (await session.execute(stmt)).all()

    by_decision = {Decision.allow: 0, Decision.mask: 0, Decision.block: 0}
    latencies: list[float] = []
    entity_counts: dict[str, int] = {}

    for input_decision, output_decision, total_latency_ms, in_red, out_red in rows:
        by_decision[_effective_decision(input_decision, output_decision)] += 1
        latencies.append(float(total_latency_ms))
        for d in (in_red or {}, out_red or {}):
            for entity_type, count in d.items():
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + count

    latencies.sort()
    top_rules = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return StatsSummary(
        window=window,
        total=len(rows),
        by_decision=by_decision,
        p50_ms=_percentile(latencies, 0.50),
        p95_ms=_percentile(latencies, 0.95),
        p99_ms=_percentile(latencies, 0.99),
        top_rules=top_rules,
    )


@router.get("/stats/timeseries", response_model=list[TimeseriesPoint])
async def stats_timeseries(
    window: str = Query(default="24h"),
    bucket: str = Query(default="5m"),
    session: AsyncSession = Depends(get_session),
):
    window_delta = _parse_duration(window, "window")
    bucket_delta = _parse_duration(bucket, "bucket")
    if bucket_delta <= timedelta(0) or bucket_delta > window_delta:
        raise HTTPException(status_code=400, detail="bucket must be > 0 and <= window")

    now = datetime.now(timezone.utc)
    since = now - window_delta
    num_buckets = max(1, int(window_delta / bucket_delta) + 1)

    stmt = select(
        AuditLog.created_at,
        AuditLog.input_decision,
        AuditLog.output_decision,
        AuditLog.total_latency_ms,
    ).where(AuditLog.created_at >= since)
    rows = (await session.execute(stmt)).all()

    buckets: list[dict] = [
        {"allow": 0, "mask": 0, "block": 0, "latencies": []} for _ in range(num_buckets)
    ]

    for created_at, input_decision, output_decision, total_latency_ms in rows:
        # SQLite returns naive datetimes even for DateTime(timezone=True)
        # columns (Postgres returns aware ones) — normalize before diffing.
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        idx = int((created_at - since) / bucket_delta)
        idx = max(0, min(num_buckets - 1, idx))
        decision = _effective_decision(input_decision, output_decision)
        buckets[idx][decision.value] += 1
        buckets[idx]["latencies"].append(float(total_latency_ms))

    points = []
    for i, b in enumerate(buckets):
        b["latencies"].sort()
        points.append(
            TimeseriesPoint(
                bucket=since + i * bucket_delta,
                allow=b["allow"],
                mask=b["mask"],
                block=b["block"],
                p95_ms=_percentile(b["latencies"], 0.95),
            )
        )
    return points
