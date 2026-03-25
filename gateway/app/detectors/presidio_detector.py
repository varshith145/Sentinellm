"""
Pass 2: Microsoft Presidio-based detector for structured PII.

Uses Presidio's AnalyzerEngine with spaCy NER for contextual PII detection.
Runs in a thread pool executor to avoid blocking the async event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from presidio_analyzer import AnalyzerEngine

from app.detectors.base import (
    BaseDetector,
    EntityCategory,
    EntityType,
    Finding,
    ENTITY_CATEGORY_MAP,
)


# --- Presidio Entity Mapping ---

PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "PERSON",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
]

PRESIDIO_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "EMAIL_ADDRESS": EntityType.EMAIL,
    "PHONE_NUMBER": EntityType.PHONE,
    "US_SSN": EntityType.SSN,
    "CREDIT_CARD": EntityType.CREDIT_CARD,
    "PERSON": EntityType.PERSON_NAME,
    "US_BANK_NUMBER": EntityType.GENERIC_PII,
    "US_PASSPORT": EntityType.GENERIC_PII,
    "US_DRIVER_LICENSE": EntityType.GENERIC_PII,
    "IP_ADDRESS": EntityType.GENERIC_PII,
}


class PresidioDetector(BaseDetector):
    """
    Presidio-based PII detector using spaCy NER and contextual analysis.

    The AnalyzerEngine is initialized once at startup. Detection runs in a
    thread pool executor since Presidio is synchronous.
    """

    def __init__(self) -> None:
        self.analyzer = AnalyzerEngine()
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def detect(self, text: str) -> list[Finding]:
        """Run Presidio analysis in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            lambda: self.analyzer.analyze(
                text=text,
                entities=PRESIDIO_ENTITIES,
                language="en",
            ),
        )
        return [self._to_finding(result, text) for result in results]

    def _to_finding(self, result, text: str) -> Finding:
        """Convert a Presidio RecognizerResult to a SentinelLM Finding."""
        entity_type = PRESIDIO_TO_ENTITY_TYPE.get(
            result.entity_type, EntityType.GENERIC_PII
        )
        return Finding(
            entity_type=entity_type,
            category=ENTITY_CATEGORY_MAP[entity_type],
            start=result.start,
            end=result.end,
            matched_text=text[result.start : result.end],
            confidence=result.score,
            detector="presidio",
        )
