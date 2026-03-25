"""
Policy Engine for SentinelLM.

Reads policy rules from a YAML config file and evaluates detection findings
to produce a decision: ALLOW, MASK, or BLOCK.

Decision priority: BLOCK > MASK > ALLOW.
If any finding triggers BLOCK, the entire request is blocked.
"""

from pathlib import Path

import yaml

from app.detectors.base import Finding


class PolicyDecision:
    """Result of policy evaluation against a set of findings."""

    def __init__(
        self, action: str, reasons: list[str], findings: list[Finding]
    ) -> None:
        self.action = action        # "ALLOW", "MASK", or "BLOCK"
        self.reasons = reasons      # Human-readable reason strings
        self.findings = findings    # Findings that triggered the decision


class PolicyEngine:
    """
    Evaluates detection findings against configurable YAML-based rules.

    Each rule specifies an entity type, a minimum confidence threshold,
    and an action (ALLOW, MASK, BLOCK). The engine returns the highest
    priority action triggered by any finding.
    """

    def __init__(self, policy_path: str = "policies/default.yaml") -> None:
        policy_file = Path(policy_path)
        if not policy_file.exists():
            # Try relative to gateway directory
            policy_file = Path(__file__).parent.parent / policy_path
        if not policy_file.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        with open(policy_file) as f:
            self.config = yaml.safe_load(f)

        self.policy_id: str = self.config["policy_id"]
        self.rules: dict[str, dict] = {
            r["entity_type"]: r for r in self.config["rules"]
        }
        self.default_action: str = self.config.get("default_action", "ALLOW")
        self.output_scanning: dict = self.config.get("output_scanning", {})

    def evaluate(
        self, findings: list[Finding], is_output: bool = False
    ) -> PolicyDecision:
        """
        Evaluate a list of findings against the policy rules.

        Args:
            findings: Detection findings to evaluate.
            is_output: If True, use output scanning rules (MASK instead of BLOCK).

        Returns:
            A PolicyDecision with the appropriate action and reasons.
        """
        if not findings:
            return PolicyDecision("ALLOW", [], [])

        action = "ALLOW"
        reasons: list[str] = []
        actionable_findings: list[Finding] = []

        for finding in findings:
            rule = self.rules.get(finding.entity_type.value)

            if rule is None:
                # No rule for this entity type — use default
                continue

            if finding.confidence < rule.get("min_confidence", 0.5):
                # Below confidence threshold — skip
                continue

            rule_action = rule["action"]

            # For output scanning, override BLOCK → MASK
            if is_output and rule_action == "BLOCK":
                output_cfg = self.output_scanning
                if output_cfg.get("enabled", True):
                    rule_action = output_cfg.get("secret_action", "MASK")

            reasons.append(
                f"{rule_action}: {finding.entity_type.value} detected "
                f"(confidence={finding.confidence:.2f}, detector={finding.detector})"
            )
            actionable_findings.append(finding)

            # Escalate action: ALLOW < MASK < BLOCK
            if rule_action == "BLOCK":
                action = "BLOCK"
            elif rule_action == "MASK" and action != "BLOCK":
                action = "MASK"

        return PolicyDecision(action, reasons, actionable_findings)
