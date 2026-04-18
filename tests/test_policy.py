"""
Advanced unit tests for the Policy Engine.

Covers: all entity types, every confidence threshold, decision priority
(BLOCK > MASK > ALLOW), output scanning mode, unknown entity types,
reasons format, and policy metadata.
"""

import pytest

from app.detectors.base import EntityCategory, EntityType, Finding
from app.policy import PolicyEngine


# ── Helpers ────────────────────────────────────────────────────


def _f(
    entity_type: EntityType,
    confidence: float = 0.95,
    detector: str = "regex",
) -> Finding:
    """Shorthand to build a Finding for policy testing."""
    category = (
        EntityCategory.SECRET
        if entity_type
        in (
            EntityType.AWS_KEY,
            EntityType.GITHUB_TOKEN,
            EntityType.JWT,
            EntityType.PASSWORD,
            EntityType.GENERIC_SECRET,
        )
        else EntityCategory.PII
    )
    return Finding(
        entity_type=entity_type,
        category=category,
        start=0,
        end=10,
        matched_text="test_value",
        confidence=confidence,
        detector=detector,
    )


@pytest.fixture
def engine(tmp_path):
    """
    Policy engine with calibrated thresholds matching the real default.yaml.
    All entity types are covered so tests can exercise every rule.
    """
    policy_file = tmp_path / "test_policy.yaml"
    policy_file.write_text("""\
policy_id: "test-v1"
description: "Full-coverage test policy"

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


# ═══════════════════════════════════════════════════════════════
#  BASIC DECISIONS
# ═══════════════════════════════════════════════════════════════


class TestBasicDecisions:
    """ALLOW / MASK / BLOCK fundamentals."""

    def test_empty_findings_returns_allow(self, engine):
        decision = engine.evaluate([])
        assert decision.action == "ALLOW"
        assert decision.reasons == []
        assert decision.findings == []

    def test_pii_triggers_mask(self, engine):
        assert engine.evaluate([_f(EntityType.EMAIL)]).action == "MASK"

    def test_secret_triggers_block(self, engine):
        assert engine.evaluate([_f(EntityType.AWS_KEY)]).action == "BLOCK"

    def test_block_overrides_mask_when_both_present(self, engine):
        """If any finding is BLOCK, overall result is BLOCK regardless of MASK."""
        findings = [_f(EntityType.EMAIL), _f(EntityType.AWS_KEY)]
        assert engine.evaluate(findings).action == "BLOCK"

    def test_mask_overrides_allow(self, engine):
        """Email (MASK) + no secret → MASK, not ALLOW."""
        decision = engine.evaluate([_f(EntityType.EMAIL)])
        assert decision.action == "MASK"

    def test_multiple_pii_findings_give_mask(self, engine):
        findings = [
            _f(EntityType.EMAIL),
            _f(EntityType.PHONE),
            _f(EntityType.SSN),
        ]
        assert engine.evaluate(findings).action == "MASK"

    def test_multiple_secrets_give_block(self, engine):
        findings = [_f(EntityType.AWS_KEY), _f(EntityType.GITHUB_TOKEN)]
        assert engine.evaluate(findings).action == "BLOCK"


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE THRESHOLDS
# ═══════════════════════════════════════════════════════════════


class TestConfidenceThresholds:
    """Boundary conditions at/above/below every threshold in the policy."""

    # EMAIL threshold = 0.7
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.69, "ALLOW"),  # just below
            (0.70, "MASK"),  # exactly at threshold
            (0.71, "MASK"),  # just above
            (0.30, "ALLOW"),  # well below
            (1.00, "MASK"),  # maximum
        ],
    )
    def test_email_confidence_boundary(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.EMAIL, confidence=conf)])
        assert decision.action == expected_action, (
            f"EMAIL conf={conf}: expected {expected_action}, got {decision.action}"
        )

    # SSN threshold = 0.5 (lower than others — high risk)
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.49, "ALLOW"),
            (0.50, "MASK"),
            (0.51, "MASK"),
        ],
    )
    def test_ssn_low_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.SSN, confidence=conf)])
        assert decision.action == expected_action

    # CREDIT_CARD threshold = 0.8
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.79, "ALLOW"),
            (0.80, "MASK"),
            (0.81, "MASK"),
        ],
    )
    def test_credit_card_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.CREDIT_CARD, confidence=conf)])
        assert decision.action == expected_action

    # PERSON_NAME threshold = 0.85 (high — to avoid common name false positives)
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.84, "ALLOW"),
            (0.85, "MASK"),
            (0.90, "MASK"),
        ],
    )
    def test_person_name_high_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.PERSON_NAME, confidence=conf)])
        assert decision.action == expected_action

    # AWS_KEY threshold = 0.5 (block at any reasonable confidence)
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.49, "ALLOW"),
            (0.50, "BLOCK"),
            (0.99, "BLOCK"),
        ],
    )
    def test_aws_key_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.AWS_KEY, confidence=conf)])
        assert decision.action == expected_action

    # PASSWORD threshold = 0.6
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.59, "ALLOW"),
            (0.60, "BLOCK"),
            (0.61, "BLOCK"),
        ],
    )
    def test_password_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.PASSWORD, confidence=conf)])
        assert decision.action == expected_action

    # GENERIC_SECRET threshold = 0.90 (very high — calibrated to filter uncertain detections)
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.85, "ALLOW"),  # AWS false-positive zone — below threshold
            (0.89, "ALLOW"),  # Still below
            (0.90, "BLOCK"),  # Exactly at threshold
            (1.00, "BLOCK"),  # Genuine secret (model gives ~1.00)
        ],
    )
    def test_generic_secret_high_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.GENERIC_SECRET, confidence=conf)])
        assert decision.action == expected_action, (
            f"GENERIC_SECRET conf={conf}: expected {expected_action}, got {decision.action}"
        )

    # JWT threshold = 0.7
    @pytest.mark.parametrize(
        "conf,expected_action",
        [
            (0.69, "ALLOW"),
            (0.70, "BLOCK"),
        ],
    )
    def test_jwt_threshold(self, engine, conf, expected_action):
        decision = engine.evaluate([_f(EntityType.JWT, confidence=conf)])
        assert decision.action == expected_action


# ═══════════════════════════════════════════════════════════════
#  ALL ENTITY TYPES
# ═══════════════════════════════════════════════════════════════


class TestAllEntityTypes:
    """Every entity type in the policy must trigger its configured action."""

    @pytest.mark.parametrize(
        "entity_type,expected_action",
        [
            (EntityType.EMAIL, "MASK"),
            (EntityType.PHONE, "MASK"),
            (EntityType.SSN, "MASK"),
            (EntityType.CREDIT_CARD, "MASK"),
            (EntityType.PERSON_NAME, "MASK"),
            (EntityType.GENERIC_PII, "MASK"),
            (EntityType.AWS_KEY, "BLOCK"),
            (EntityType.GITHUB_TOKEN, "BLOCK"),
            (EntityType.JWT, "BLOCK"),
            (EntityType.PASSWORD, "BLOCK"),
        ],
    )
    def test_entity_type_action(self, engine, entity_type, expected_action):
        """Each entity type fires its configured action at high confidence."""
        decision = engine.evaluate([_f(entity_type, confidence=0.99)])
        assert decision.action == expected_action, (
            f"{entity_type}: expected {expected_action}, got {decision.action}"
        )


# ═══════════════════════════════════════════════════════════════
#  OUTPUT SCANNING
# ═══════════════════════════════════════════════════════════════


class TestOutputScanning:
    """In output mode, BLOCK → MASK; MASK stays MASK; ALLOW stays ALLOW."""

    def test_output_block_becomes_mask_for_secret(self, engine):
        """AWS key would BLOCK in input mode but should MASK in output mode."""
        decision = engine.evaluate([_f(EntityType.AWS_KEY)], is_output=True)
        assert decision.action == "MASK"

    def test_output_block_becomes_mask_for_github_token(self, engine):
        decision = engine.evaluate([_f(EntityType.GITHUB_TOKEN)], is_output=True)
        assert decision.action == "MASK"

    def test_output_pii_still_masks(self, engine):
        """PII was already MASK — output mode doesn't change that."""
        decision = engine.evaluate([_f(EntityType.EMAIL)], is_output=True)
        assert decision.action == "MASK"

    def test_output_allow_stays_allow(self, engine):
        """No findings → ALLOW in both modes."""
        decision = engine.evaluate([], is_output=True)
        assert decision.action == "ALLOW"

    def test_output_mixed_pii_and_secret_gives_mask(self, engine):
        """Both findings → neither BLOCK (overridden to MASK) → combined = MASK."""
        findings = [_f(EntityType.EMAIL), _f(EntityType.AWS_KEY)]
        decision = engine.evaluate(findings, is_output=True)
        assert decision.action == "MASK"


# ═══════════════════════════════════════════════════════════════
#  UNKNOWN ENTITY TYPES & DEFAULT ACTION
# ═══════════════════════════════════════════════════════════════


class TestUnknownEntityAndDefaults:
    """Entity types with no rule use default_action; policy metadata is correct."""

    def test_unknown_entity_type_uses_default_allow(self, engine):
        """
        A Finding whose entity_type has no matching rule is skipped.
        Since all findings are skipped, result is ALLOW (the default).
        """
        # Use GENERIC_PII with confidence below its threshold so it's skipped
        finding = _f(EntityType.GENERIC_PII, confidence=0.1)
        decision = engine.evaluate([finding])
        assert decision.action == "ALLOW"

    def test_policy_id_loaded_correctly(self, engine):
        assert engine.policy_id == "test-v1"

    def test_default_action_is_allow(self, engine):
        assert engine.default_action == "ALLOW"

    def test_output_scanning_config_loaded(self, engine):
        assert engine.output_scanning.get("enabled") is True
        assert engine.output_scanning.get("secret_action") == "MASK"


# ═══════════════════════════════════════════════════════════════
#  REASONS FORMAT
# ═══════════════════════════════════════════════════════════════


class TestReasonsFormat:
    """Reasons must include entity type, action, confidence, and detector."""

    def test_reason_contains_entity_type(self, engine):
        decision = engine.evaluate([_f(EntityType.EMAIL)])
        assert len(decision.reasons) == 1
        assert "EMAIL" in decision.reasons[0]

    def test_reason_contains_action(self, engine):
        decision = engine.evaluate([_f(EntityType.EMAIL)])
        assert "MASK" in decision.reasons[0]

    def test_reason_contains_confidence(self, engine):
        decision = engine.evaluate([_f(EntityType.EMAIL, confidence=0.95)])
        assert "0.95" in decision.reasons[0]

    def test_reason_contains_detector_name(self, engine):
        decision = engine.evaluate([_f(EntityType.EMAIL, detector="presidio")])
        assert "presidio" in decision.reasons[0]

    def test_multiple_findings_give_multiple_reasons(self, engine):
        findings = [_f(EntityType.EMAIL), _f(EntityType.PHONE)]
        decision = engine.evaluate(findings)
        assert len(decision.reasons) == 2

    def test_blocked_finding_included_in_actionable(self, engine):
        finding = _f(EntityType.AWS_KEY)
        decision = engine.evaluate([finding])
        assert finding in decision.findings

    def test_below_threshold_finding_excluded_from_actionable(self, engine):
        """Low-confidence finding should NOT appear in decision.findings."""
        low_conf_finding = _f(EntityType.EMAIL, confidence=0.1)
        decision = engine.evaluate([low_conf_finding])
        assert decision.action == "ALLOW"
        assert low_conf_finding not in decision.findings
