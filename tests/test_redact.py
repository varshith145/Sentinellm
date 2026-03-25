"""
Unit tests for the Redaction module.

Tests typed token replacement, offset preservation, and count tracking.
"""

import pytest

from app.detectors.base import EntityCategory, EntityType, Finding
from app.redact import redact_text


def _make_finding(
    entity_type: EntityType,
    start: int,
    end: int,
    matched_text: str,
) -> Finding:
    return Finding(
        entity_type=entity_type,
        category=EntityCategory.PII,
        start=start,
        end=end,
        matched_text=matched_text,
        confidence=0.95,
        detector="regex",
    )


def test_redact_single_email():
    text = "Contact john@example.com for help."
    findings = [_make_finding(EntityType.EMAIL, 8, 24, "john@example.com")]
    redacted, counts = redact_text(text, findings)
    assert redacted == "Contact [REDACTED_EMAIL] for help."
    assert counts == {"EMAIL": 1}


def test_redact_multiple_entities():
    text = "Email john@test.com, phone 555-123-4567."
    findings = [
        _make_finding(EntityType.EMAIL, 6, 19, "john@test.com"),
        _make_finding(EntityType.PHONE, 27, 39, "555-123-4567"),
    ]
    redacted, counts = redact_text(text, findings)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert counts == {"EMAIL": 1, "PHONE": 1}


def test_redact_preserves_surrounding_text():
    text = "Before john@test.com after."
    findings = [_make_finding(EntityType.EMAIL, 7, 20, "john@test.com")]
    redacted, _ = redact_text(text, findings)
    assert redacted.startswith("Before ")
    assert redacted.endswith(" after.")


def test_redact_empty_findings():
    text = "No sensitive data here."
    redacted, counts = redact_text(text, [])
    assert redacted == text
    assert counts == {}


def test_redact_ssn():
    text = "SSN: 123-45-6789"
    findings = [_make_finding(EntityType.SSN, 5, 16, "123-45-6789")]
    redacted, counts = redact_text(text, findings)
    assert redacted == "SSN: [REDACTED_SSN]"
    assert counts == {"SSN": 1}


def test_redact_credit_card():
    text = "Card: 4532015112830366"
    findings = [_make_finding(EntityType.CREDIT_CARD, 6, 22, "4532015112830366")]
    redacted, counts = redact_text(text, findings)
    assert redacted == "Card: [REDACTED_CC]"
    assert counts == {"CREDIT_CARD": 1}


def test_redact_secret_types():
    text = "Key: AKIAIOSFODNN7EXAMPLE"
    findings = [
        Finding(
            entity_type=EntityType.AWS_KEY,
            category=EntityCategory.SECRET,
            start=5,
            end=25,
            matched_text="AKIAIOSFODNN7EXAMPLE",
            confidence=0.95,
            detector="regex",
        )
    ]
    redacted, counts = redact_text(text, findings)
    assert redacted == "Key: [REDACTED_SECRET]"
    assert counts == {"AWS_KEY": 1}


def test_redact_multiple_same_type():
    text = "Email a@b.com and c@d.com"
    findings = [
        _make_finding(EntityType.EMAIL, 6, 13, "a@b.com"),
        _make_finding(EntityType.EMAIL, 18, 25, "c@d.com"),
    ]
    redacted, counts = redact_text(text, findings)
    assert redacted.count("[REDACTED_EMAIL]") == 2
    assert counts == {"EMAIL": 2}


def test_redact_offset_preservation():
    """Verify that reverse-order processing preserves correct offsets."""
    text = "A john@a.com B jane@b.com C"
    findings = [
        _make_finding(EntityType.EMAIL, 2, 12, "john@a.com"),
        _make_finding(EntityType.EMAIL, 15, 25, "jane@b.com"),
    ]
    redacted, counts = redact_text(text, findings)
    assert redacted == "A [REDACTED_EMAIL] B [REDACTED_EMAIL] C"
    assert counts == {"EMAIL": 2}
