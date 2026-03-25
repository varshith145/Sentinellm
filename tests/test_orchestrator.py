"""
Unit tests for the Detection Orchestrator.

Tests merging, deduplication, and parallel detector execution.
"""

import pytest

from app.detectors.base import (
    BaseDetector,
    EntityCategory,
    EntityType,
    Finding,
)
from app.detectors.orchestrator import DetectionOrchestrator


class MockDetector(BaseDetector):
    """A mock detector that returns pre-configured findings."""

    def __init__(self, findings: list[Finding]):
        self._findings = findings

    async def detect(self, text: str) -> list[Finding]:
        return self._findings


class FailingDetector(BaseDetector):
    """A mock detector that raises an exception."""

    async def detect(self, text: str) -> list[Finding]:
        raise RuntimeError("Detector failed!")


@pytest.mark.asyncio
async def test_merge_no_overlap():
    """Non-overlapping findings from different detectors should all be kept."""
    detector1 = MockDetector([
        Finding(EntityType.EMAIL, EntityCategory.PII, 0, 10, "test@a.com", 0.95, "regex"),
    ])
    detector2 = MockDetector([
        Finding(EntityType.PHONE, EntityCategory.PII, 20, 32, "555-123-4567", 0.90, "presidio"),
    ])

    orchestrator = DetectionOrchestrator([detector1, detector2])
    findings = await orchestrator.scan("test text")

    assert len(findings) == 2


@pytest.mark.asyncio
async def test_deduplicate_overlapping_same_span():
    """Overlapping findings: keep the one with higher confidence."""
    detector1 = MockDetector([
        Finding(EntityType.EMAIL, EntityCategory.PII, 0, 15, "john@test.com", 0.95, "regex"),
    ])
    detector2 = MockDetector([
        Finding(EntityType.EMAIL, EntityCategory.PII, 0, 15, "john@test.com", 0.85, "presidio"),
    ])

    orchestrator = DetectionOrchestrator([detector1, detector2])
    findings = await orchestrator.scan("test text")

    assert len(findings) == 1
    assert findings[0].detector == "regex"
    assert findings[0].confidence == 0.95


@pytest.mark.asyncio
async def test_deduplicate_prefer_semantic_on_tie():
    """On confidence tie, prefer semantic > presidio > regex."""
    detector1 = MockDetector([
        Finding(EntityType.GENERIC_PII, EntityCategory.PII, 0, 10, "text", 0.80, "presidio"),
    ])
    detector2 = MockDetector([
        Finding(EntityType.GENERIC_PII, EntityCategory.PII, 0, 10, "text", 0.80, "semantic"),
    ])

    orchestrator = DetectionOrchestrator([detector1, detector2])
    findings = await orchestrator.scan("test text")

    assert len(findings) == 1
    assert findings[0].detector == "semantic"


@pytest.mark.asyncio
async def test_empty_findings():
    """No findings should return empty list."""
    detector = MockDetector([])
    orchestrator = DetectionOrchestrator([detector])
    findings = await orchestrator.scan("clean text")
    assert findings == []


@pytest.mark.asyncio
async def test_failing_detector_doesnt_crash():
    """A failing detector shouldn't crash the orchestrator."""
    good_detector = MockDetector([
        Finding(EntityType.EMAIL, EntityCategory.PII, 0, 10, "test@a.com", 0.95, "regex"),
    ])
    bad_detector = FailingDetector()

    orchestrator = DetectionOrchestrator([good_detector, bad_detector])
    findings = await orchestrator.scan("test text")

    assert len(findings) == 1
    assert findings[0].detector == "regex"


@pytest.mark.asyncio
async def test_get_active_detectors():
    """Should return detector names."""
    from app.detectors.regex import RegexDetector

    orchestrator = DetectionOrchestrator([RegexDetector()])
    names = orchestrator.get_active_detectors()
    assert "regex" in names
