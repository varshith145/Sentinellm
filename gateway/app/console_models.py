"""
Pydantic models for the console API (/api/v1/*).

These land in openapi.json, which is what a generated TypeScript client
(openapi-typescript) reads — keep them accurate rather than convenient.

Several fields on AuditRecord are *derived* from columns that exist for a
different reason (see the docstring on each). They are not stored verbatim,
because the audit table predates this API and intentionally stores only
what the request pipeline needs to write cheaply and safely.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from app.detectors.base import EntityCategory, EntityType
from pydantic import BaseModel, Field


class Decision(str, Enum):
    allow = "allow"
    mask = "mask"
    block = "block"


def _to_decision(action: str | None) -> Decision:
    """Map an internal ALLOW/MASK/BLOCK action string to the console Decision enum."""
    return Decision((action or "ALLOW").lower())


# --- Policies ---


class PolicyRule(BaseModel):
    """A single policy rule, as stored in the `policies` table."""

    id: str
    name: str
    description: str | None = None
    entity_type: EntityType
    category: EntityCategory
    action: Decision
    min_confidence: float = Field(ge=0.0, le=1.0)
    enabled: bool
    created_at: datetime
    updated_at: datetime


class PolicyCreate(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    entity_type: EntityType
    action: Decision
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True


class PolicyUpdate(BaseModel):
    """All fields optional — PATCH semantics. Only provided fields change."""

    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    entity_type: EntityType | None = None
    action: Decision | None = None
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool | None = None


class DryRunRequest(BaseModel):
    text: str = Field(min_length=1)


class DryRunFinding(BaseModel):
    entity_type: EntityType
    category: EntityCategory
    text: str
    start: int
    end: int
    confidence: float
    detector: str


class DryRunResponse(BaseModel):
    decision: Decision
    reasons: list[str]
    findings: list[DryRunFinding]
    redacted_text: str
    latency_ms: float


# --- Audit ---


class AuditRecord(BaseModel):
    """
    A single audit log entry, as surfaced to the console.

    Field mapping from the `audit_log` table (see app/db.py `AuditLog`):
      - decision            <- input_decision if BLOCK, else the worse of
                                (input_decision, output_decision) by
                                ALLOW < MASK < BLOCK. The effective outcome
                                a developer cares about, not a raw column.
      - matched_rule_ids     <- keys of input_redactions ∪ output_redactions.
                                The audit table doesn't store rule IDs (only
                                human-readable `reasons` strings), so this is
                                the entity types that fired, which is the
                                closest real signal — not a fabricated ID.
      - entity_types          <- same source as matched_rule_ids.
      - detector_confidence   <- always None. Per-finding confidence isn't
                                persisted in the audit table (only redaction
                                *counts* by entity type are). Not faked.
      - tenant_id             <- user_id column, renamed to match the plan's
                                multi-tenant-shaped field. This project has
                                no real tenant concept yet.
      - request_preview       <- prompt_redacted, truncated to 200 chars.
                                Already redacted before it reached the DB.
    """

    id: str
    ts: datetime
    decision: Decision
    matched_rule_ids: list[str]
    latency_ms: float
    detector_confidence: float | None = None
    entity_types: list[str]
    request_preview: str
    tenant_id: str | None = None


class AuditDetail(AuditRecord):
    """Full record for the audit detail drawer — adds what the list view omits."""

    model: str
    input_decision: Decision
    output_decision: Decision | None = None
    reasons: list[str]
    input_redactions: dict[str, int]
    output_redactions: dict[str, int]
    prompt_redacted: str
    response_redacted: str | None = None
    detection_latency_ms: int
    llm_latency_ms: int | None = None
    total_latency_ms: int
    policy_id: str


class PaginatedAudit(BaseModel):
    items: list[AuditRecord]
    next_cursor: str | None = None


class PaginatedPolicies(BaseModel):
    items: list[PolicyRule]
    next_cursor: str | None = None


# --- Stats ---


class StatsSummary(BaseModel):
    window: str
    total: int
    by_decision: dict[Decision, int]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    top_rules: list[tuple[str, int]] = Field(
        description=(
            "Top entity types by trigger count in the window — the closest "
            "real proxy to 'top rules' (see AuditRecord.matched_rule_ids)."
        )
    )


class TimeseriesPoint(BaseModel):
    bucket: datetime
    allow: int
    mask: int
    block: int
    p95_ms: float
