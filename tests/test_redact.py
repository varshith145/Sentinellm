"""
Advanced unit tests for the Redaction module.

Covers: all 11 entity types and their exact tokens, position invariants
(start/middle/end of string), multiple simultaneous redactions, adjacent
entities, reverse-order offset preservation, and count aggregation.
"""

import pytest

from app.detectors.base import EntityCategory, EntityType, Finding
from app.redact import REDACTION_TOKENS, redact_text


# ── Helper ─────────────────────────────────────────────────────


def _f(
    entity_type: EntityType,
    start: int,
    end: int,
    matched_text: str,
    category: EntityCategory = EntityCategory.PII,
    confidence: float = 0.95,
    detector: str = "regex",
) -> Finding:
    return Finding(
        entity_type=entity_type,
        category=category,
        start=start,
        end=end,
        matched_text=matched_text,
        confidence=confidence,
        detector=detector,
    )


# ═══════════════════════════════════════════════════════════════
#  ALL ENTITY TYPES → CORRECT REDACTION TOKEN
# ═══════════════════════════════════════════════════════════════


class TestRedactionTokens:
    """Every entity type must map to its correct replacement token."""

    @pytest.mark.parametrize(
        "entity_type,expected_token",
        [
            (EntityType.EMAIL, "[REDACTED_EMAIL]"),
            (EntityType.PHONE, "[REDACTED_PHONE]"),
            (EntityType.SSN, "[REDACTED_SSN]"),
            (EntityType.CREDIT_CARD, "[REDACTED_CC]"),
            (EntityType.PERSON_NAME, "[REDACTED_NAME]"),
            (EntityType.AWS_KEY, "[REDACTED_SECRET]"),
            (EntityType.GITHUB_TOKEN, "[REDACTED_SECRET]"),
            (EntityType.JWT, "[REDACTED_SECRET]"),
            (EntityType.PASSWORD, "[REDACTED_SECRET]"),
            (EntityType.GENERIC_PII, "[REDACTED_PII]"),
            (EntityType.GENERIC_SECRET, "[REDACTED_SECRET]"),
        ],
    )
    def test_token_for_entity_type(self, entity_type, expected_token):
        text = f"prefix {entity_type.value} suffix"
        start = len("prefix ")
        end = start + len(entity_type.value)
        finding = _f(entity_type, start, end, entity_type.value)
        redacted, _ = redact_text(text, [finding])
        assert expected_token in redacted, (
            f"{entity_type}: expected {expected_token!r} in redacted text"
        )

    def test_redaction_tokens_dict_covers_all_entity_types(self):
        """REDACTION_TOKENS must have an entry for all 11 entity types."""
        for entity_type in EntityType:
            assert entity_type in REDACTION_TOKENS, (
                f"Missing REDACTION_TOKENS entry for {entity_type}"
            )


# ═══════════════════════════════════════════════════════════════
#  POSITION TESTS
# ═══════════════════════════════════════════════════════════════


class TestPositions:
    """Entity at start, middle, and end of string — surrounding text preserved."""

    def test_redact_at_start(self):
        text = "john@example.com is a contact."
        findings = [_f(EntityType.EMAIL, 0, 16, "john@example.com")]
        redacted, counts = redact_text(text, findings)
        assert redacted == "[REDACTED_EMAIL] is a contact."
        assert counts == {"EMAIL": 1}

    def test_redact_in_middle(self):
        text = "Contact john@example.com for help."
        findings = [_f(EntityType.EMAIL, 8, 24, "john@example.com")]
        redacted, counts = redact_text(text, findings)
        assert redacted == "Contact [REDACTED_EMAIL] for help."
        assert counts == {"EMAIL": 1}

    def test_redact_at_end(self):
        text = "Email address: john@example.com"
        findings = [_f(EntityType.EMAIL, 15, 31, "john@example.com")]
        redacted, counts = redact_text(text, findings)
        assert redacted == "Email address: [REDACTED_EMAIL]"
        assert counts == {"EMAIL": 1}

    def test_redact_spanning_entire_string(self):
        text = "john@example.com"
        findings = [_f(EntityType.EMAIL, 0, 16, "john@example.com")]
        redacted, counts = redact_text(text, findings)
        assert redacted == "[REDACTED_EMAIL]"
        assert counts == {"EMAIL": 1}

    def test_empty_findings_returns_original_text(self):
        text = "No sensitive data here."
        redacted, counts = redact_text(text, [])
        assert redacted == text
        assert counts == {}


# ═══════════════════════════════════════════════════════════════
#  MULTIPLE ENTITIES
# ═══════════════════════════════════════════════════════════════


class TestMultipleEntities:
    """Multiple non-overlapping findings must all be redacted correctly."""

    def test_two_non_overlapping_findings(self):
        text = "Email john@test.com, phone 555-123-4567."
        findings = [
            _f(EntityType.EMAIL, 6, 19, "john@test.com"),
            _f(EntityType.PHONE, 27, 39, "555-123-4567"),
        ]
        redacted, counts = redact_text(text, findings)
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert counts == {"EMAIL": 1, "PHONE": 1}
        assert "john@test.com" not in redacted
        assert "555-123-4567" not in redacted

    def test_two_same_type_findings_count_aggregated(self):
        text = "Email a@b.com and c@d.com"
        findings = [
            _f(EntityType.EMAIL, 6, 13, "a@b.com"),
            _f(EntityType.EMAIL, 18, 25, "c@d.com"),
        ]
        redacted, counts = redact_text(text, findings)
        assert redacted.count("[REDACTED_EMAIL]") == 2
        assert counts == {"EMAIL": 2}

    def test_offset_preservation_multiple_entities(self):
        """Reverse-order processing must keep all offsets correct."""
        text = "A john@a.com B jane@b.com C"
        findings = [
            _f(EntityType.EMAIL, 2, 12, "john@a.com"),
            _f(EntityType.EMAIL, 15, 25, "jane@b.com"),
        ]
        redacted, counts = redact_text(text, findings)
        assert redacted == "A [REDACTED_EMAIL] B [REDACTED_EMAIL] C"
        assert counts == {"EMAIL": 2}

    def test_three_different_types_all_redacted(self):
        text = "john@x.com 555-123-4567 123-45-6789"
        findings = [
            _f(EntityType.EMAIL, 0, 10, "john@x.com"),
            _f(EntityType.PHONE, 11, 23, "555-123-4567"),
            _f(EntityType.SSN, 24, 35, "123-45-6789"),
        ]
        redacted, counts = redact_text(text, findings)
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert "[REDACTED_SSN]" in redacted
        assert counts == {"EMAIL": 1, "PHONE": 1, "SSN": 1}
        # Original values must be gone
        assert "john@x.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "123-45-6789" not in redacted

    def test_findings_provided_in_forward_order_still_works(self):
        """redact_text sorts internally — caller order doesn't matter."""
        text = "SSN 123-45-6789 and email john@test.com"
        # Provide findings in forward order (SSN before email)
        findings = [
            _f(EntityType.SSN, 4, 15, "123-45-6789"),
            _f(EntityType.EMAIL, 26, 39, "john@test.com"),
        ]
        redacted, counts = redact_text(text, findings)
        assert "[REDACTED_SSN]" in redacted
        assert "[REDACTED_EMAIL]" in redacted
        assert counts["SSN"] == 1
        assert counts["EMAIL"] == 1

    def test_findings_provided_in_reverse_order_still_works(self):
        """redact_text sorts internally — reverse caller order is fine too."""
        text = "SSN 123-45-6789 and email john@test.com"
        # Provide findings in reverse order (email before SSN)
        findings = [
            _f(EntityType.EMAIL, 26, 39, "john@test.com"),
            _f(EntityType.SSN, 4, 15, "123-45-6789"),
        ]
        redacted, counts = redact_text(text, findings)
        assert "[REDACTED_SSN]" in redacted
        assert "[REDACTED_EMAIL]" in redacted


# ═══════════════════════════════════════════════════════════════
#  ADJACENT ENTITIES
# ═══════════════════════════════════════════════════════════════


class TestAdjacentEntities:
    """Two findings whose spans are adjacent (end == start) — both redacted."""

    def test_adjacent_findings_both_redacted(self):
        # Construct text: "AB" where A=EMAIL and B=PHONE (no gap between them)
        email_part = "x@y.com"
        phone_part = "5551234567"
        text = email_part + phone_part
        findings = [
            _f(EntityType.EMAIL, 0, len(email_part), email_part),
            _f(EntityType.PHONE, len(email_part), len(text), phone_part),
        ]
        redacted, counts = redact_text(text, findings)
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert counts == {"EMAIL": 1, "PHONE": 1}


# ═══════════════════════════════════════════════════════════════
#  SPECIFIC TOKEN CORRECTNESS
# ═══════════════════════════════════════════════════════════════


class TestSpecificTokens:
    """Spot-check correct token for key entity types."""

    def test_ssn_token(self):
        text = "SSN: 123-45-6789"
        findings = [_f(EntityType.SSN, 5, 16, "123-45-6789")]
        redacted, _ = redact_text(text, findings)
        assert redacted == "SSN: [REDACTED_SSN]"

    def test_credit_card_token(self):
        text = "Card: 4532015112830366"
        findings = [_f(EntityType.CREDIT_CARD, 6, 22, "4532015112830366")]
        redacted, _ = redact_text(text, findings)
        assert redacted == "Card: [REDACTED_CC]"

    def test_aws_key_token(self):
        text = "Key: AKIAIOSFODNN7EXAMPLE"
        findings = [
            _f(
                EntityType.AWS_KEY,
                5,
                25,
                "AKIAIOSFODNN7EXAMPLE",
                category=EntityCategory.SECRET,
            )
        ]
        redacted, counts = redact_text(text, findings)
        assert redacted == "Key: [REDACTED_SECRET]"
        assert counts == {"AWS_KEY": 1}

    def test_person_name_token(self):
        text = "User John Smith logged in."
        findings = [_f(EntityType.PERSON_NAME, 5, 15, "John Smith")]
        redacted, _ = redact_text(text, findings)
        assert redacted == "User [REDACTED_NAME] logged in."

    def test_generic_pii_token(self):
        text = "Data: sensitive_value end"
        findings = [_f(EntityType.GENERIC_PII, 6, 21, "sensitive_value")]
        redacted, _ = redact_text(text, findings)
        assert redacted == "Data: [REDACTED_PII] end"

    def test_generic_secret_token(self):
        text = "secret: hunter2"
        findings = [
            _f(
                EntityType.GENERIC_SECRET,
                8,
                15,
                "hunter2",
                category=EntityCategory.SECRET,
            )
        ]
        redacted, _ = redact_text(text, findings)
        assert redacted == "secret: [REDACTED_SECRET]"

    def test_large_number_of_entities_counts_correctly(self):
        """Five emails in text → count should be 5."""
        emails = ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com"]
        text = " ".join(emails)
        findings = []
        pos = 0
        for email in emails:
            findings.append(_f(EntityType.EMAIL, pos, pos + len(email), email))
            pos += len(email) + 1  # +1 for space
        redacted, counts = redact_text(text, findings)
        assert counts == {"EMAIL": 5}
        assert redacted.count("[REDACTED_EMAIL]") == 5
