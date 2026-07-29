"""
Detection Orchestrator — merges and deduplicates findings from all detectors.

Runs the fast structural detectors (regex, Presidio) in parallel via
asyncio.gather first. The semantic pass (fine-tuned DistilBERT, ~150ms —
by far the most expensive of the three, see bench/results.md) only runs
when the fast passes didn't already reach a confident conclusion, then
merges overlapping findings keeping the highest confidence result.
"""

import asyncio
import logging

from app.detectors.base import BaseDetector, Finding

logger = logging.getLogger(__name__)

# Detector priority for tie-breaking (higher = preferred)
_DETECTOR_PRIORITY = {
    "semantic": 3,
    "presidio": 2,
    "regex": 1,
}

# A regex finding at or above this confidence is treated as decisive enough
# to skip the semantic pass for this request. Regex matches are exact
# structural patterns (Luhn-validated card numbers, email/SSN/API-key
# shapes) — a hit is essentially never wrong about *what's there*, but it
# says nothing about what else might be in the same text. Presidio's
# findings deliberately do NOT count here: measured on eval/run_eval.py,
# letting a confident PERSON_NAME match (Presidio's contextual, non-exact
# pass) trigger the skip dropped micro-F1 from 0.9574 to 0.8298 — it fires
# on names inside text that also contains obfuscated PII/secrets Presidio
# itself can't see, and skipping semantic then missed them.
_CONCLUSIVE_CONFIDENCE = 0.85
_CONCLUSIVE_DETECTOR = "regex"


class DetectionOrchestrator:
    """
    Orchestrates all detection passes and merges their results.

    Runs detectors in parallel, then deduplicates overlapping findings
    by keeping the one with the highest confidence. On ties, prefers
    semantic > presidio > regex.
    """

    def __init__(self, detectors: list[BaseDetector]) -> None:
        self.detectors = detectors
        self._fast_detectors = [
            d for d in detectors if "semantic" not in type(d).__name__.lower()
        ]
        self._slow_detectors = [
            d for d in detectors if "semantic" in type(d).__name__.lower()
        ]

    async def scan(self, text: str) -> list[Finding]:
        """
        Run the fast detectors, escalate to the semantic pass only if
        needed, and return deduplicated findings.

        Args:
            text: The input text to scan.

        Returns:
            Deduplicated list of Finding objects sorted by start position.
        """
        fast_results = await asyncio.gather(
            *[d.detect(text) for d in self._fast_detectors],
            return_exceptions=True,
        )
        all_findings = self._flatten(fast_results)

        # Regex and Presidio only ever match known structural patterns —
        # an obfuscated secret or PII string ("john dot smith at co dot com")
        # produces zero fast-pass findings by construction, so "found
        # nothing" must still escalate to semantic, not skip it. Skipping is
        # only safe once a fast pass already found something decisive.
        if self._slow_detectors and not self._is_conclusive(all_findings):
            slow_results = await asyncio.gather(
                *[d.detect(text) for d in self._slow_detectors],
                return_exceptions=True,
            )
            all_findings.extend(self._flatten(slow_results))

        return self._deduplicate(all_findings)

    def _flatten(self, results: list) -> list[Finding]:
        """Collect findings from a asyncio.gather(return_exceptions=True) batch."""
        findings: list[Finding] = []
        for result in results:
            if isinstance(result, Exception):
                # Log but don't crash — other detectors' results still valid
                logger.error(f"Detector failed: {result}", exc_info=result)
                continue
            findings.extend(result)
        return findings

    def _is_conclusive(self, findings: list[Finding]) -> bool:
        """Whether the fast passes alone are confident enough to skip semantic."""
        return any(
            f.detector == _CONCLUSIVE_DETECTOR
            and f.confidence >= _CONCLUSIVE_CONFIDENCE
            for f in findings
        )

    def _deduplicate(self, findings: list[Finding]) -> list[Finding]:
        """
        Remove overlapping findings, keeping the highest confidence result.

        Algorithm:
        1. Sort by start position, then by confidence descending, then by
           detector priority descending.
        2. Walk through sorted list; for overlapping spans, keep the one
           with higher confidence (or higher detector priority on ties).
        """
        if not findings:
            return []

        # Sort: by start asc, then confidence desc, then detector priority desc
        findings.sort(
            key=lambda f: (
                f.start,
                -f.confidence,
                -_DETECTOR_PRIORITY.get(f.detector, 0),
            )
        )

        merged: list[Finding] = [findings[0]]
        for f in findings[1:]:
            prev = merged[-1]

            # Check for overlap
            if f.start < prev.end:
                # Overlapping — keep the better one
                if f.confidence > prev.confidence or (
                    f.confidence == prev.confidence
                    and _DETECTOR_PRIORITY.get(f.detector, 0)
                    > _DETECTOR_PRIORITY.get(prev.detector, 0)
                ):
                    merged[-1] = f
            else:
                # No overlap — add to merged list
                merged.append(f)

        return merged

    def get_active_detectors(self) -> list[str]:
        """Return names of active detector types."""
        names = []
        for d in self.detectors:
            cls_name = type(d).__name__.lower()
            if "regex" in cls_name:
                names.append("regex")
            elif "presidio" in cls_name:
                names.append("presidio")
            elif "semantic" in cls_name:
                names.append("semantic")
            else:
                names.append(cls_name)
        return names
