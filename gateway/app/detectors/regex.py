"""
Pass 1: Regex-based detector for structured PII and secrets.

Fast, deterministic pattern matching. Runs in <1ms.
Catches formatted PII (emails, phones, SSNs, credit cards) and
secrets (AWS keys, GitHub tokens, JWTs).
"""

import re
from app.detectors.base import (
    BaseDetector,
    EntityType,
    Finding,
    ENTITY_CATEGORY_MAP,
)


# --- Luhn Validation (Credit Cards) ---


def luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 16:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# --- Compiled Regex Patterns ---
# Each entry: (EntityType, compiled_pattern, confidence, requires_luhn)

_PATTERNS: list[tuple[EntityType, re.Pattern, float, bool]] = [
    # PII patterns
    (
        EntityType.EMAIL,
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        0.95,
        False,
    ),
    (
        EntityType.PHONE,
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        0.95,
        False,
    ),
    (
        EntityType.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        0.95,
        False,
    ),
    (
        EntityType.CREDIT_CARD,
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        0.99,  # Only returned if Luhn passes
        True,
    ),
    # Secret patterns
    (
        EntityType.AWS_KEY,
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        0.95,
        False,
    ),
    (
        EntityType.GITHUB_TOKEN,
        re.compile(r"\b(ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"),
        0.95,
        False,
    ),
    (
        EntityType.JWT,
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
        0.95,
        False,
    ),
]


class RegexDetector(BaseDetector):
    """
    Fast regex-based detector for structured PII and known secret formats.

    All patterns are compiled at import time. Returns Finding objects with
    exact character offsets and high confidence scores.
    """

    async def detect(self, text: str) -> list[Finding]:
        """Scan text using compiled regex patterns."""
        findings: list[Finding] = []

        for entity_type, pattern, confidence, requires_luhn in _PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group()

                # Credit card: validate with Luhn algorithm
                if requires_luhn and not luhn_check(matched_text):
                    continue

                findings.append(
                    Finding(
                        entity_type=entity_type,
                        category=ENTITY_CATEGORY_MAP[entity_type],
                        start=match.start(),
                        end=match.end(),
                        matched_text=matched_text,
                        confidence=confidence,
                        detector="regex",
                    )
                )

        return findings
