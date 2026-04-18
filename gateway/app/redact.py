"""
Redaction module for SentinelLM.

Replaces matched sensitive spans with typed redaction tokens.
Processes findings in reverse order to preserve character offsets.
"""

from app.detectors.base import EntityType, Finding


# Redaction token for each entity type
REDACTION_TOKENS: dict[EntityType, str] = {
    EntityType.EMAIL: "[REDACTED_EMAIL]",
    EntityType.PHONE: "[REDACTED_PHONE]",
    EntityType.SSN: "[REDACTED_SSN]",
    EntityType.CREDIT_CARD: "[REDACTED_CC]",
    EntityType.PERSON_NAME: "[REDACTED_NAME]",
    EntityType.AWS_KEY: "[REDACTED_SECRET]",
    EntityType.GITHUB_TOKEN: "[REDACTED_SECRET]",
    EntityType.JWT: "[REDACTED_SECRET]",
    EntityType.PASSWORD: "[REDACTED_SECRET]",
    EntityType.GENERIC_PII: "[REDACTED_PII]",
    EntityType.GENERIC_SECRET: "[REDACTED_SECRET]",
}


def redact_text(text: str, findings: list[Finding]) -> tuple[str, dict[str, int]]:
    """
    Replace matched spans with redaction tokens.

    Processes findings in reverse order (end of string first) to preserve
    character offsets during replacement.

    Args:
        text: The original text containing sensitive data.
        findings: List of Finding objects with start/end offsets.

    Returns:
        A tuple of (redacted_text, redaction_counts).
        - redacted_text: The text with sensitive spans replaced by tokens.
        - redaction_counts: Dict mapping entity type names to counts,
          e.g. {"EMAIL": 1, "PHONE": 2}.
    """
    if not findings:
        return text, {}

    redaction_counts: dict[str, int] = {}

    # Sort by start position descending — process from end to preserve offsets
    sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)

    redacted = text
    for finding in sorted_findings:
        token = REDACTION_TOKENS.get(finding.entity_type, "[REDACTED]")
        redacted = redacted[: finding.start] + token + redacted[finding.end :]

        entity_name = finding.entity_type.value
        redaction_counts[entity_name] = redaction_counts.get(entity_name, 0) + 1

    return redacted, redaction_counts
