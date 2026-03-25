"""
Unit tests for the Policy Engine.

Tests ALLOW/MASK/BLOCK decision logic, confidence thresholds,
and decision escalation.
"""

import pytest

from app.detectors.base import EntityCategory, EntityType, Finding
from app.policy import PolicyEngine


@pytest.fixture
def engine(tmp_path):
    """Create a policy engine with a test policy file."""
    policy_file = tmp_path / "test_policy.yaml"
    policy_file.write_text("""
policy_id: "test-v1"
description: "Test policy"

rules:
  - entity_type: EMAIL
    category: PII
    action: MASK
    min_confidence: 0.7

  - entity_type: PHONE
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

  - entity_type: PERSON_NAME
    category: PII
    action: MASK
    min_confidence: 0.85

default_action: ALLOW

output_scanning:
  enabled: true
  secret_action: MASK
  pii_action: MASK
""")
    return PolicyEngine(policy_path=str(policy_file))


def _make_finding(
    entity_type: EntityType,
    category: EntityCategory,
    confidence: float = 0.95,
    detector: str = "regex",
) -> Finding:
    return Finding(
        entity_type=entity_type,
        category=category,
        start=0,
        end=10,
        matched_text="test",
        confidence=confidence,
        detector=detector,
    )


def test_no_findings_returns_allow(engine):
    decision = engine.evaluate([])
    assert decision.action == "ALLOW"
    assert len(decision.reasons) == 0


def test_pii_triggers_mask(engine):
    findings = [_make_finding(EntityType.EMAIL, EntityCategory.PII)]
    decision = engine.evaluate(findings)
    assert decision.action == "MASK"


def test_secret_triggers_block(engine):
    findings = [_make_finding(EntityType.AWS_KEY, EntityCategory.SECRET)]
    decision = engine.evaluate(findings)
    assert decision.action == "BLOCK"


def test_block_overrides_mask(engine):
    """If any finding triggers BLOCK, overall decision is BLOCK."""
    findings = [
        _make_finding(EntityType.EMAIL, EntityCategory.PII),
        _make_finding(EntityType.AWS_KEY, EntityCategory.SECRET),
    ]
    decision = engine.evaluate(findings)
    assert decision.action == "BLOCK"


def test_below_confidence_threshold(engine):
    """Findings below confidence threshold are ignored."""
    findings = [
        _make_finding(EntityType.EMAIL, EntityCategory.PII, confidence=0.3),
    ]
    decision = engine.evaluate(findings)
    assert decision.action == "ALLOW"


def test_at_confidence_threshold(engine):
    """Findings at exactly the threshold are included."""
    findings = [
        _make_finding(EntityType.EMAIL, EntityCategory.PII, confidence=0.7),
    ]
    decision = engine.evaluate(findings)
    assert decision.action == "MASK"


def test_person_name_high_threshold(engine):
    """Person names need 0.85 confidence to trigger MASK."""
    low = [_make_finding(EntityType.PERSON_NAME, EntityCategory.PII, confidence=0.80)]
    high = [_make_finding(EntityType.PERSON_NAME, EntityCategory.PII, confidence=0.90)]

    assert engine.evaluate(low).action == "ALLOW"
    assert engine.evaluate(high).action == "MASK"


def test_output_scanning_overrides_block_to_mask(engine):
    """In output scanning mode, BLOCK becomes MASK."""
    findings = [_make_finding(EntityType.AWS_KEY, EntityCategory.SECRET)]
    decision = engine.evaluate(findings, is_output=True)
    assert decision.action == "MASK"


def test_reasons_contain_details(engine):
    findings = [_make_finding(EntityType.EMAIL, EntityCategory.PII)]
    decision = engine.evaluate(findings)
    assert len(decision.reasons) == 1
    assert "EMAIL" in decision.reasons[0]
    assert "MASK" in decision.reasons[0]


def test_policy_id(engine):
    assert engine.policy_id == "test-v1"
