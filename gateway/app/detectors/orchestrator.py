"""
Detection Orchestrator — merges and deduplicates findings from all detectors.

Runs all three detectors (regex, Presidio, semantic) in parallel via
asyncio.gather, then merges overlapping findings keeping the highest
confidence result.
"""

import asyncio

from app.detectors.base import BaseDetector, Finding


# Detector priority for tie-breaking (higher = preferred)
_DETECTOR_PRIORITY = {
    "semantic": 3,
    "presidio": 2,
    "regex": 1,
}


class DetectionOrchestrator:
    """
    Orchestrates all detection passes and merges their results.

    Runs detectors in parallel, then deduplicates overlapping findings
    by keeping the one with the highest confidence. On ties, prefers
    semantic > presidio > regex.
    """

    def __init__(self, detectors: list[BaseDetector]) -> None:
        self.detectors = detectors

    async def scan(self, text: str) -> list[Finding]:
        """
        Run all detectors in parallel and return deduplicated findings.

        Args:
            text: The input text to scan.

        Returns:
            Deduplicated list of Finding objects sorted by start position.
        """
        results = await asyncio.gather(
            *[d.detect(text) for d in self.detectors],
            return_exceptions=True,
        )

        all_findings: list[Finding] = []
        for result in results:
            if isinstance(result, Exception):
                # Log but don't crash — other detectors' results still valid
                import logging

                logging.getLogger(__name__).error(
                    f"Detector failed: {result}", exc_info=result
                )
                continue
            all_findings.extend(result)

        return self._deduplicate(all_findings)

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
