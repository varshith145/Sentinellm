"""
Base types and interfaces for the SentinelLM detection pipeline.

Defines EntityType, EntityCategory, Finding, and BaseDetector — the core
abstractions used by all three detection passes (regex, Presidio, semantic).
"""

from dataclasses import dataclass
from enum import Enum


class EntityType(str, Enum):
    """Types of sensitive entities that SentinelLM can detect."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    AWS_KEY = "AWS_KEY"
    GITHUB_TOKEN = "GITHUB_TOKEN"
    JWT = "JWT"
    PASSWORD = "PASSWORD"
    PERSON_NAME = "PERSON_NAME"
    GENERIC_PII = "GENERIC_PII"
    GENERIC_SECRET = "GENERIC_SECRET"


class EntityCategory(str, Enum):
    """High-level category of a detected entity."""

    PII = "PII"
    SECRET = "SECRET"


# Map entity types to their category
ENTITY_CATEGORY_MAP: dict[EntityType, EntityCategory] = {
    EntityType.EMAIL: EntityCategory.PII,
    EntityType.PHONE: EntityCategory.PII,
    EntityType.SSN: EntityCategory.PII,
    EntityType.CREDIT_CARD: EntityCategory.PII,
    EntityType.PERSON_NAME: EntityCategory.PII,
    EntityType.GENERIC_PII: EntityCategory.PII,
    EntityType.AWS_KEY: EntityCategory.SECRET,
    EntityType.GITHUB_TOKEN: EntityCategory.SECRET,
    EntityType.JWT: EntityCategory.SECRET,
    EntityType.PASSWORD: EntityCategory.SECRET,
    EntityType.GENERIC_SECRET: EntityCategory.SECRET,
}


@dataclass
class Finding:
    """
    A single detected sensitive entity in text.

    Attributes:
        entity_type: The specific type of entity detected.
        category: High-level category (PII or SECRET).
        start: Character offset of the start of the match in the original text.
        end: Character offset of the end of the match in the original text.
        matched_text: The actual matched substring.
        confidence: Detection confidence score (0.0 to 1.0).
        detector: Name of the detector that produced this finding
                  ("regex", "presidio", or "semantic").
    """

    entity_type: EntityType
    category: EntityCategory
    start: int
    end: int
    matched_text: str
    confidence: float
    detector: str


class BaseDetector:
    """
    Abstract base class for all detectors in the SentinelLM pipeline.

    Each detector must implement the `detect` method, which takes a text string
    and returns a list of Finding objects.
    """

    async def detect(self, text: str) -> list[Finding]:
        """
        Scan text for sensitive entities.

        Args:
            text: The input text to scan.

        Returns:
            A list of Finding objects for each detected entity.
        """
        raise NotImplementedError
