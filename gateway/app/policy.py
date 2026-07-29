"""
Policy Engine for SentinelLM.

Rules live in the `policies` DB table (console-editable at runtime). The
policy YAML file (gateway/policies/default.yaml) is only read twice:

  1. Once at startup, to seed the `policies` table if it's empty.
  2. To supply policy_id / default_action / output_scanning — global policy
     behavior that isn't yet exposed as a per-rule console setting.

Decision priority: BLOCK > MASK > ALLOW.
If any finding triggers BLOCK, the entire request is blocked.

`evaluate()` stays synchronous over an in-memory rules dict so the hot
request path never hits the DB. Call `reload()` after any write to the
policies table (console CRUD, or seeding) to refresh the cache.
"""

from pathlib import Path

import yaml
from app.db import Policy
from app.detectors.base import Finding
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class PolicyDecision:
    """Result of policy evaluation against a set of findings."""

    def __init__(
        self, action: str, reasons: list[str], findings: list[Finding]
    ) -> None:
        self.action = action  # "ALLOW", "MASK", or "BLOCK"
        self.reasons = reasons  # Human-readable reason strings
        self.findings = findings  # Findings that triggered the decision


def evaluate_findings(
    findings: list[Finding],
    rules: dict[str, dict],
    output_scanning: dict,
    is_output: bool = False,
) -> PolicyDecision:
    """
    Evaluate findings against an arbitrary rules dict.

    Free function (rather than a PolicyEngine method) so a dry-run can build
    a one-off rules dict — e.g. the live rules with a single candidate rule
    swapped in — without mutating the shared PolicyEngine.rules cache that
    the live request path depends on.
    """
    if not findings:
        return PolicyDecision("ALLOW", [], [])

    action = "ALLOW"
    reasons: list[str] = []
    actionable_findings: list[Finding] = []

    for finding in findings:
        rule = rules.get(finding.entity_type.value)

        if rule is None:
            continue

        if finding.confidence < rule.get("min_confidence", 0.5):
            continue

        rule_action = rule["action"]

        if (
            is_output
            and rule_action == "BLOCK"
            and output_scanning.get("enabled", True)
        ):
            rule_action = output_scanning.get("secret_action", "MASK")

        reasons.append(
            f"{rule_action}: {finding.entity_type.value} detected "
            f"(confidence={finding.confidence:.2f}, detector={finding.detector})"
        )
        actionable_findings.append(finding)

        if rule_action == "BLOCK":
            action = "BLOCK"
        elif rule_action == "MASK" and action != "BLOCK":
            action = "MASK"

    return PolicyDecision(action, reasons, actionable_findings)


def _resolve_policy_path(policy_path: str) -> Path:
    policy_file = Path(policy_path)
    if not policy_file.exists():
        # Try relative to gateway directory
        policy_file = Path(__file__).parent.parent / policy_path
    if not policy_file.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    return policy_file


class PolicyEngine:
    """
    Evaluates detection findings against policy rules.

    Each rule specifies an entity type, a minimum confidence threshold,
    and an action (ALLOW, MASK, BLOCK). The engine returns the highest
    priority action triggered by any finding.

    The constructor loads rules from the YAML file directly, so a bare
    `PolicyEngine(policy_path=...)` is immediately usable standalone (tests,
    scripts, dry-run-only tooling) with no DB involved. The live gateway app
    goes one step further: after construction it calls `reload(session)`,
    which overwrites `self.rules` from the `policies` DB table — making the
    DB authoritative at runtime while the YAML remains the one-time seed
    source (see `seed_if_empty`).
    """

    def __init__(self, policy_path: str = "policies/default.yaml") -> None:
        policy_file = _resolve_policy_path(policy_path)
        self.policy_path = policy_file

        with open(policy_file) as f:
            config = yaml.safe_load(f)

        self.policy_id: str = config["policy_id"]
        self.default_action: str = config.get("default_action", "ALLOW")
        self.output_scanning: dict = config.get("output_scanning", {})

        # Seed source for the DB table (see seed_if_empty), and also the
        # engine's own rules until/unless reload(session) is called.
        self.seed_rules: list[dict] = config["rules"]
        self.rules: dict[str, dict] = {
            r["entity_type"]: {
                "action": r["action"],
                "min_confidence": r.get("min_confidence", 0.5),
            }
            for r in self.seed_rules
        }

    async def seed_if_empty(self, session: AsyncSession) -> bool:
        """
        Insert the YAML rules as DB rows if the policies table is empty.

        Returns True if seeding happened, False if rows already existed.
        No-op (and safe to call every startup) once the table is populated —
        the DB is authoritative from that point on.
        """
        existing = (await session.execute(select(Policy.id).limit(1))).first()
        if existing is not None:
            return False

        for rule in self.seed_rules:
            session.add(
                Policy(
                    name=f"{rule['entity_type']} default",
                    description=None,
                    entity_type=rule["entity_type"],
                    category=rule["category"],
                    action=rule["action"],
                    min_confidence=rule.get("min_confidence", 0.5),
                    enabled=True,
                )
            )
        await session.commit()
        return True

    async def reload(self, session: AsyncSession) -> None:
        """
        Rebuild the in-memory rules cache from the `policies` table.

        On duplicate entity_type across enabled rows, the most recently
        updated row wins (rows are read ordered by updated_at ascending,
        so later rows overwrite earlier ones in the dict).
        """
        result = await session.execute(
            select(Policy)
            .where(Policy.enabled.is_(True))
            .order_by(Policy.updated_at.asc())
        )
        rows = result.scalars().all()

        rules: dict[str, dict] = {}
        for row in rows:
            rules[row.entity_type] = {
                "action": row.action,
                "min_confidence": row.min_confidence,
            }
        self.rules = rules

    def evaluate(
        self, findings: list[Finding], is_output: bool = False
    ) -> PolicyDecision:
        """
        Evaluate a list of findings against the cached policy rules.

        Args:
            findings: Detection findings to evaluate.
            is_output: If True, use output scanning rules (MASK instead of BLOCK).

        Returns:
            A PolicyDecision with the appropriate action and reasons.
        """
        return evaluate_findings(findings, self.rules, self.output_scanning, is_output)
