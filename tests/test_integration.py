"""
End-to-end integration tests for the SentinelLM pipeline.

Tests the full three-stage pipeline: detect → evaluate → redact,
using real detectors (RegexDetector) and real engines (PolicyEngine,
redact_text) together. No mocks — everything is wired end-to-end.

These tests validate that the components compose correctly and that
the system-level behaviour (decisions, redacted text, counts) is right.
"""

import pytest

from app.detectors.base import EntityType
from app.detectors.orchestrator import DetectionOrchestrator
from app.detectors.regex import RegexDetector
from app.policy import PolicyEngine
from app.redact import redact_text


# ── Shared Policy Fixture ───────────────────────────────────────

@pytest.fixture
def policy(tmp_path):
    """
    Full-coverage policy matching the real default.yaml semantics.
    Used for all integration tests.
    """
    policy_file = tmp_path / "integration_policy.yaml"
    policy_file.write_text("""\
policy_id: "integration-v1"
description: "Integration test policy"

rules:
  - entity_type: EMAIL
    category: PII
    action: MASK
    min_confidence: 0.7

  - entity_type: PHONE
    category: PII
    action: MASK
    min_confidence: 0.7

  - entity_type: SSN
    category: PII
    action: MASK
    min_confidence: 0.5

  - entity_type: CREDIT_CARD
    category: PII
    action: MASK
    min_confidence: 0.8

  - entity_type: PERSON_NAME
    category: PII
    action: MASK
    min_confidence: 0.85

  - entity_type: GENERIC_PII
    category: PII
    action: MASK
    min_confidence: 0.7

  - entity_type: AWS_KEY
    category: SECRET
    action: BLOCK
    min_confidence: 0.5

  - entity_type: GITHUB_TOKEN
    category: SECRET
    action: BLOCK
    min_confidence: 0.5

  - entity_type: JWT
    category: SECRET
    action: BLOCK
    min_confidence: 0.7

  - entity_type: PASSWORD
    category: SECRET
    action: BLOCK
    min_confidence: 0.6

  - entity_type: GENERIC_SECRET
    category: SECRET
    action: BLOCK
    min_confidence: 0.90

default_action: ALLOW

output_scanning:
  enabled: true
  secret_action: MASK
  pii_action: MASK
""")
    return PolicyEngine(policy_path=str(policy_file))


@pytest.fixture
def regex_detector():
    return RegexDetector()


@pytest.fixture
def orchestrator(regex_detector):
    return DetectionOrchestrator([regex_detector])


# ═══════════════════════════════════════════════════════════════
#  CLEAN TEXT → ALLOW
# ═══════════════════════════════════════════════════════════════

class TestCleanTextAllowed:
    """Text with no sensitive data must pass through untouched."""

    @pytest.mark.asyncio
    async def test_plain_sentence_is_allowed(self, orchestrator, policy):
        text = "What is the weather like today in San Francisco?"
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "ALLOW"
        assert len(decision.findings) == 0

    @pytest.mark.asyncio
    async def test_technical_text_about_email_is_allowed(self, orchestrator, policy):
        """Describing email protocols, not containing an email address."""
        text = "The SMTP protocol transfers email between mail servers."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "ALLOW"

    @pytest.mark.asyncio
    async def test_aws_mention_without_key_is_allowed(self, orchestrator, policy):
        """Mentioning AWS as a service (no AKIA key) must not trigger BLOCK."""
        text = "We deploy our services on AWS using S3 and Lambda."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "ALLOW"

    @pytest.mark.asyncio
    async def test_number_in_context_is_allowed(self, orchestrator, policy):
        text = "The function accepts between 3 and 10 arguments."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "ALLOW"


# ═══════════════════════════════════════════════════════════════
#  PII DETECTION → MASK
# ═══════════════════════════════════════════════════════════════

class TestPIITriggersMAsk:
    """Real PII in text must produce MASK decision and redacted output."""

    @pytest.mark.asyncio
    async def test_email_triggers_mask_and_redacts(self, orchestrator, policy):
        text = "Please contact alice@example.com for onboarding."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "MASK"
        redacted, counts = redact_text(text, decision.findings)
        assert "[REDACTED_EMAIL]" in redacted
        assert "alice@example.com" not in redacted
        assert counts.get("EMAIL") == 1

    @pytest.mark.asyncio
    async def test_phone_triggers_mask_and_redacts(self, orchestrator, policy):
        text = "Call support at 555-867-5309."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "MASK"
        redacted, counts = redact_text(text, decision.findings)
        assert "[REDACTED_PHONE]" in redacted
        assert "555-867-5309" not in redacted

    @pytest.mark.asyncio
    async def test_ssn_triggers_mask_and_redacts(self, orchestrator, policy):
        text = "Her social security number is 987-65-4321."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "MASK"
        redacted, _ = redact_text(text, decision.findings)
        assert "[REDACTED_SSN]" in redacted
        assert "987-65-4321" not in redacted

    @pytest.mark.asyncio
    async def test_credit_card_triggers_mask_and_redacts(self, orchestrator, policy):
        text = "Charge card 4532015112830366 for the order."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "MASK"
        redacted, _ = redact_text(text, decision.findings)
        assert "[REDACTED_CC]" in redacted
        assert "4532015112830366" not in redacted

    @pytest.mark.asyncio
    async def test_multiple_pii_types_all_masked(self, orchestrator, policy):
        text = (
            "User alice@corp.com, phone 555-123-4567, "
            "SSN 111-22-3333 submitted the form."
        )
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "MASK"
        redacted, counts = redact_text(text, decision.findings)
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert "[REDACTED_SSN]" in redacted
        assert "alice@corp.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "111-22-3333" not in redacted


# ═══════════════════════════════════════════════════════════════
#  SECRET DETECTION → BLOCK
# ═══════════════════════════════════════════════════════════════

class TestSecretTriggersBlock:
    """Secrets in text must produce BLOCK decision — redaction not performed."""

    @pytest.mark.asyncio
    async def test_aws_key_triggers_block(self, orchestrator, policy):
        text = "export AWS_KEY=AKIAIOSFODNN7EXAMPLE"
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "BLOCK"

    @pytest.mark.asyncio
    async def test_github_token_triggers_block(self, orchestrator, policy):
        token = "ghp_" + "A" * 36
        text = f"My token is {token}"
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "BLOCK"

    @pytest.mark.asyncio
    async def test_jwt_triggers_block(self, orchestrator, policy):
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9"
            ".eyJzdWIiOiJ1c2VyMSIsImlhdCI6MTYwMH0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        text = f"Token: {jwt}"
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "BLOCK"

    @pytest.mark.asyncio
    async def test_secret_with_pii_gives_block(self, orchestrator, policy):
        """BLOCK takes priority over MASK when both present."""
        text = "Email alice@corp.com and key AKIAIOSFODNN7EXAMPLE"
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "BLOCK"


# ═══════════════════════════════════════════════════════════════
#  OUTPUT SCANNING MODE
# ═══════════════════════════════════════════════════════════════

class TestOutputScanningMode:
    """In output mode, BLOCK → MASK so response is not swallowed."""

    @pytest.mark.asyncio
    async def test_aws_key_in_output_gives_mask(self, orchestrator, policy):
        text = "Here is your key: AKIAIOSFODNN7EXAMPLE"
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings, is_output=True)
        assert decision.action == "MASK"
        redacted, _ = redact_text(text, decision.findings)
        assert "[REDACTED_SECRET]" in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted

    @pytest.mark.asyncio
    async def test_email_in_output_stays_mask(self, orchestrator, policy):
        text = "You can reach the user at dev@company.com."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings, is_output=True)
        assert decision.action == "MASK"

    @pytest.mark.asyncio
    async def test_clean_output_stays_allow(self, orchestrator, policy):
        text = "The task has been completed successfully."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings, is_output=True)
        assert decision.action == "ALLOW"


# ═══════════════════════════════════════════════════════════════
#  REDACTED TEXT INVARIANTS
# ═══════════════════════════════════════════════════════════════

class TestRedactionInvariants:
    """Properties that must hold after redaction regardless of input."""

    @pytest.mark.asyncio
    async def test_redacted_text_does_not_contain_original_pii(self, orchestrator, policy):
        sensitive_values = [
            "alice@example.com",
            "555-123-4567",
            "123-45-6789",
        ]
        text = ", ".join(sensitive_values)
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        if decision.action == "MASK":
            redacted, _ = redact_text(text, decision.findings)
            for value in sensitive_values:
                assert value not in redacted, f"PII not redacted: {value!r}"

    @pytest.mark.asyncio
    async def test_allow_decision_text_unchanged(self, orchestrator, policy):
        text = "No sensitive data in this message at all."
        findings = await orchestrator.scan(text)
        decision = policy.evaluate(findings)
        assert decision.action == "ALLOW"
        # No redaction needed; text stays the same
        redacted, counts = redact_text(text, decision.findings)
        assert redacted == text
        assert counts == {}

    @pytest.mark.asyncio
    async def test_redaction_count_matches_detected_entities(self, orchestrator, policy):
        """Number of EMAIL redactions must equal number of emails detected."""
        text = "Send to alice@corp.com and bob@example.org"
        findings = await orchestrator.scan(text)
        email_findings = [f for f in findings if f.entity_type == EntityType.EMAIL]
        decision = policy.evaluate(findings)
        if decision.action in ("MASK", "BLOCK"):
            _, counts = redact_text(text, decision.findings)
            assert counts.get("EMAIL", 0) == len(email_findings)

    @pytest.mark.asyncio
    async def test_pipeline_is_idempotent_on_clean_text(self, orchestrator, policy):
        """Running detect → evaluate → (no redaction) on clean text yields ALLOW every time."""
        text = "Meeting at 3pm to discuss Q4 roadmap."
        for _ in range(3):
            findings = await orchestrator.scan(text)
            decision = policy.evaluate(findings)
            assert decision.action == "ALLOW"
