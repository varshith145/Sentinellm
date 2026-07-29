"""
Advanced unit tests for the Detection Orchestrator.

Covers: parallel execution, deduplication algorithm (same span / partial overlap /
adjacent spans), confidence tie-breaking, detector priority, failure resilience,
empty inputs, and result ordering.
"""

import pytest
from app.detectors.base import (
    BaseDetector,
    EntityCategory,
    EntityType,
    Finding,
)
from app.detectors.orchestrator import DetectionOrchestrator

# ── Mock Detectors ─────────────────────────────────────────────


class MockDetector(BaseDetector):
    """Returns a fixed list of pre-configured findings."""

    def __init__(self, findings: list[Finding]):
        self._findings = findings

    async def detect(self, text: str) -> list[Finding]:
        return list(self._findings)  # Return copy to avoid mutation issues


class FailingDetector(BaseDetector):
    """Raises RuntimeError on every detect() call."""

    async def detect(self, text: str) -> list[Finding]:
        raise RuntimeError("Detector crashed!")


class SlowFailingDetector(BaseDetector):
    """Simulates a detector that raises an exception asynchronously."""

    async def detect(self, text: str) -> list[Finding]:
        raise ValueError("Async detector failed!")


class MockSemanticDetector(BaseDetector):
    """A mock detector whose class name matches the orchestrator's
    "semantic" name check, with a call counter so tests can assert whether
    the short-circuit actually skipped or invoked it."""

    def __init__(self, findings: list[Finding]):
        self._findings = findings
        self.call_count = 0

    async def detect(self, text: str) -> list[Finding]:
        self.call_count += 1
        return list(self._findings)


# ── Helper ─────────────────────────────────────────────────────


def _f(
    start: int,
    end: int,
    entity_type: EntityType = EntityType.EMAIL,
    confidence: float = 0.95,
    detector: str = "regex",
) -> Finding:
    return Finding(
        entity_type=entity_type,
        category=EntityCategory.PII,
        start=start,
        end=end,
        matched_text="x" * (end - start),
        confidence=confidence,
        detector=detector,
    )


# ═══════════════════════════════════════════════════════════════
#  BASIC SCAN BEHAVIOUR
# ═══════════════════════════════════════════════════════════════


class TestBasicScan:
    """Fundamental scan behaviour: empty, single detector, multi-detector."""

    @pytest.mark.asyncio
    async def test_no_detectors_returns_empty(self):
        orchestrator = DetectionOrchestrator([])
        findings = await orchestrator.scan("some text")
        assert findings == []

    @pytest.mark.asyncio
    async def test_single_detector_no_findings(self):
        orchestrator = DetectionOrchestrator([MockDetector([])])
        findings = await orchestrator.scan("clean text")
        assert findings == []

    @pytest.mark.asyncio
    async def test_single_detector_single_finding(self):
        expected = _f(0, 10)
        orchestrator = DetectionOrchestrator([MockDetector([expected])])
        findings = await orchestrator.scan("text")
        assert len(findings) == 1
        assert findings[0] is expected

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        from app.detectors.regex import RegexDetector

        orchestrator = DetectionOrchestrator([RegexDetector()])
        findings = await orchestrator.scan("")
        assert findings == []

    @pytest.mark.asyncio
    async def test_results_sorted_by_start_position(self):
        """Findings must be returned in ascending start-position order."""
        d = MockDetector(
            [
                _f(30, 40, confidence=0.9),
                _f(0, 10, confidence=0.9),
                _f(15, 25, confidence=0.9),
            ]
        )
        orchestrator = DetectionOrchestrator([d])
        findings = await orchestrator.scan("text")
        starts = [f.start for f in findings]
        assert starts == sorted(starts), "Findings not sorted by start"


# ═══════════════════════════════════════════════════════════════
#  NON-OVERLAPPING FINDINGS
# ═══════════════════════════════════════════════════════════════


class TestNonOverlappingFindings:
    """Non-overlapping spans from one or multiple detectors all kept."""

    @pytest.mark.asyncio
    async def test_two_non_overlapping_same_detector(self):
        findings = [_f(0, 10), _f(15, 25)]
        orchestrator = DetectionOrchestrator([MockDetector(findings)])
        result = await orchestrator.scan("text")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_two_non_overlapping_different_detectors(self):
        d1 = MockDetector([_f(0, 10, detector="regex")])
        d2 = MockDetector([_f(20, 30, detector="presidio")])
        orchestrator = DetectionOrchestrator([d1, d2])
        result = await orchestrator.scan("text")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_adjacent_spans_both_kept(self):
        """end == start of next → no overlap → both findings kept."""
        d = MockDetector([_f(0, 10), _f(10, 20)])
        orchestrator = DetectionOrchestrator([d])
        result = await orchestrator.scan("text")
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════
#  DEDUPLICATION — SAME SPAN
# ═══════════════════════════════════════════════════════════════


class TestSameSpanDeduplication:
    """Same span from multiple detectors → keep the best one."""

    @pytest.mark.asyncio
    async def test_same_span_keep_higher_confidence(self):
        """regex (0.95) vs presidio (0.85) at same span → keep regex."""
        d1 = MockDetector([_f(0, 15, confidence=0.95, detector="regex")])
        d2 = MockDetector([_f(0, 15, confidence=0.85, detector="presidio")])
        orchestrator = DetectionOrchestrator([d1, d2])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].confidence == 0.95
        assert result[0].detector == "regex"

    @pytest.mark.asyncio
    async def test_same_span_keep_higher_confidence_reversed(self):
        """presidio (0.95) vs regex (0.85) → keep presidio."""
        d1 = MockDetector([_f(0, 15, confidence=0.85, detector="regex")])
        d2 = MockDetector([_f(0, 15, confidence=0.95, detector="presidio")])
        orchestrator = DetectionOrchestrator([d1, d2])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].confidence == 0.95
        assert result[0].detector == "presidio"

    @pytest.mark.asyncio
    async def test_confidence_tie_prefers_semantic_over_presidio(self):
        """Same confidence: semantic (priority=3) beats presidio (priority=2)."""
        d1 = MockDetector([_f(0, 10, confidence=0.80, detector="presidio")])
        d2 = MockDetector([_f(0, 10, confidence=0.80, detector="semantic")])
        orchestrator = DetectionOrchestrator([d1, d2])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].detector == "semantic"

    @pytest.mark.asyncio
    async def test_confidence_tie_prefers_presidio_over_regex(self):
        """Same confidence: presidio (priority=2) beats regex (priority=1)."""
        d1 = MockDetector([_f(0, 10, confidence=0.80, detector="regex")])
        d2 = MockDetector([_f(0, 10, confidence=0.80, detector="presidio")])
        orchestrator = DetectionOrchestrator([d1, d2])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].detector == "presidio"

    @pytest.mark.asyncio
    async def test_confidence_tie_prefers_semantic_over_regex(self):
        d1 = MockDetector([_f(0, 10, confidence=0.80, detector="regex")])
        d2 = MockDetector([_f(0, 10, confidence=0.80, detector="semantic")])
        orchestrator = DetectionOrchestrator([d1, d2])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].detector == "semantic"

    @pytest.mark.asyncio
    async def test_three_detectors_same_span_keep_best(self):
        """All three detectors emit same span; semantic has highest priority."""
        d1 = MockDetector([_f(0, 10, confidence=0.80, detector="regex")])
        d2 = MockDetector([_f(0, 10, confidence=0.80, detector="presidio")])
        d3 = MockDetector([_f(0, 10, confidence=0.80, detector="semantic")])
        orchestrator = DetectionOrchestrator([d1, d2, d3])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].detector == "semantic"


# ═══════════════════════════════════════════════════════════════
#  DEDUPLICATION — PARTIAL OVERLAP
# ═══════════════════════════════════════════════════════════════


class TestPartialOverlapDeduplication:
    """Partially overlapping spans: greedy algorithm keeps the first or the better."""

    @pytest.mark.asyncio
    async def test_partial_overlap_keep_first_when_higher_confidence(self):
        """
        A starts at 0, ends at 15 (confidence 0.95).
        B starts at 10, ends at 25 (confidence 0.80).
        B.start (10) < A.end (15) → overlap.
        B.confidence (0.80) < A.confidence (0.95) → keep A.
        Result: only A.
        """
        d = MockDetector(
            [
                _f(0, 15, confidence=0.95, detector="regex"),
                _f(10, 25, confidence=0.80, detector="presidio"),
            ]
        )
        orchestrator = DetectionOrchestrator([d])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 15

    @pytest.mark.asyncio
    async def test_partial_overlap_replace_when_second_has_higher_confidence(self):
        """
        A: 0-15 confidence 0.80.  B: 10-25 confidence 0.95.
        B overlaps A and has higher confidence → replace A with B.
        Result: only B (at 10-25).
        """
        d = MockDetector(
            [
                _f(0, 15, confidence=0.80, detector="regex"),
                _f(10, 25, confidence=0.95, detector="presidio"),
            ]
        )
        orchestrator = DetectionOrchestrator([d])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].start == 10
        assert result[0].end == 25

    @pytest.mark.asyncio
    async def test_non_overlapping_after_overlap_still_added(self):
        """
        A: 0-10 confidence 0.90
        B: 5-15 confidence 0.80  (overlaps A, A wins)
        C: 20-30 confidence 0.95 (no overlap with whoever won)
        Result: A and C.
        """
        d = MockDetector(
            [
                _f(0, 10, confidence=0.90, detector="regex"),
                _f(5, 15, confidence=0.80, detector="presidio"),
                _f(20, 30, confidence=0.95, detector="semantic"),
            ]
        )
        orchestrator = DetectionOrchestrator([d])
        result = await orchestrator.scan("text")
        assert len(result) == 2
        starts = {f.start for f in result}
        assert 0 in starts
        assert 20 in starts


# ═══════════════════════════════════════════════════════════════
#  FAILURE RESILIENCE
# ═══════════════════════════════════════════════════════════════


class TestFailureResilience:
    """Failing detectors must not crash the orchestrator."""

    @pytest.mark.asyncio
    async def test_one_failing_one_good(self):
        good = MockDetector([_f(0, 10)])
        bad = FailingDetector()
        orchestrator = DetectionOrchestrator([good, bad])
        result = await orchestrator.scan("text")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_all_detectors_failing_returns_empty(self):
        orchestrator = DetectionOrchestrator([FailingDetector(), SlowFailingDetector()])
        result = await orchestrator.scan("text")
        assert result == []

    @pytest.mark.asyncio
    async def test_failing_detector_result_excluded(self):
        """Good detector's findings survive even when another detector crashes."""
        good = MockDetector([_f(0, 10, detector="regex")])
        bad = FailingDetector()
        orchestrator = DetectionOrchestrator([bad, good])
        result = await orchestrator.scan("text")
        assert len(result) == 1
        assert result[0].detector == "regex"


# ═══════════════════════════════════════════════════════════════
#  GET ACTIVE DETECTORS
# ═══════════════════════════════════════════════════════════════


class TestGetActiveDetectors:
    """get_active_detectors() must return correct names for real detectors."""

    def test_regex_detector_name(self):
        from app.detectors.regex import RegexDetector

        orchestrator = DetectionOrchestrator([RegexDetector()])
        assert "regex" in orchestrator.get_active_detectors()

    def test_multiple_detectors_all_names_returned(self):
        from app.detectors.regex import RegexDetector

        orchestrator = DetectionOrchestrator([RegexDetector(), RegexDetector()])
        names = orchestrator.get_active_detectors()
        assert len(names) == 2
        assert all(n == "regex" for n in names)

    def test_empty_orchestrator_returns_empty_list(self):
        orchestrator = DetectionOrchestrator([])
        assert orchestrator.get_active_detectors() == []


# ═══════════════════════════════════════════════════════════════
#  SEMANTIC SHORT-CIRCUIT
# ═══════════════════════════════════════════════════════════════


class TestSemanticShortCircuit:
    """The semantic pass is the expensive one (~150ms vs <1ms/~30ms for the
    others) — it should only run when the fast passes didn't already reach
    a confident conclusion."""

    @pytest.mark.asyncio
    async def test_confident_fast_finding_skips_semantic(self):
        fast = MockDetector([_f(0, 10, confidence=0.95, detector="regex")])
        semantic = MockSemanticDetector([_f(20, 30, confidence=0.9)])
        orchestrator = DetectionOrchestrator([fast, semantic])

        result = await orchestrator.scan("text")

        assert semantic.call_count == 0
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_confident_presidio_finding_still_escalates_to_semantic(self):
        """Regression test: a confident Presidio PERSON_NAME match must not
        skip semantic — it's contextual, not an exact structural match, and
        can co-occur with obfuscated content Presidio itself can't see.
        Measured on eval/run_eval.py: letting it trigger the skip dropped
        micro-F1 from 0.9574 to 0.8298."""
        fast = MockDetector([_f(0, 10, confidence=0.99, detector="presidio")])
        semantic = MockSemanticDetector([_f(20, 30, confidence=0.9)])
        orchestrator = DetectionOrchestrator([fast, semantic])

        result = await orchestrator.scan("text")

        assert semantic.call_count == 1
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_no_fast_findings_escalates_to_semantic(self):
        fast = MockDetector([])
        semantic = MockSemanticDetector([_f(0, 10, confidence=0.9)])
        orchestrator = DetectionOrchestrator([fast, semantic])

        result = await orchestrator.scan("obfuscated text")

        assert semantic.call_count == 1
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_low_confidence_fast_finding_escalates_to_semantic(self):
        fast = MockDetector([_f(0, 10, confidence=0.5, detector="presidio")])
        semantic = MockSemanticDetector([_f(20, 30, confidence=0.9)])
        orchestrator = DetectionOrchestrator([fast, semantic])

        result = await orchestrator.scan("text")

        assert semantic.call_count == 1
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_no_semantic_detector_present_is_unaffected(self):
        fast = MockDetector([_f(0, 10, confidence=0.5, detector="regex")])
        orchestrator = DetectionOrchestrator([fast])

        result = await orchestrator.scan("text")

        assert len(result) == 1
