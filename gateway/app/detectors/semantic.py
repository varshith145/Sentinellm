"""
Pass 3: Semantic detector using a fine-tuned DistilBERT NER model.

Catches obfuscated and informal PII that regex and Presidio miss, such as:
  - "john at gmail dot com"
  - "my social is four five six 78 9012"
  - "the password is hunter2"

The model performs token-level classification with BIO tags:
  O, B-PII, I-PII, B-SECRET, I-SECRET

If the trained model is not available, gracefully returns no findings.
"""

import asyncio
import logging
from pathlib import Path

from app.detectors.base import (
    BaseDetector,
    EntityCategory,
    EntityType,
    Finding,
)

logger = logging.getLogger(__name__)


# Mapping from BIO label entity type to SentinelLM types
_SEMANTIC_ENTITY_MAP = {
    "PII": EntityType.GENERIC_PII,
    "SECRET": EntityType.GENERIC_SECRET,
}

_SEMANTIC_CATEGORY_MAP = {
    "PII": EntityCategory.PII,
    "SECRET": EntityCategory.SECRET,
}


class SemanticDetector(BaseDetector):
    """
    Fine-tuned DistilBERT NER detector for obfuscated/informal PII and secrets.

    Loads the model once at initialization. If the model directory doesn't exist,
    operates in stub mode (returns no findings).
    """

    def __init__(self, model_path: str = "./model/trained") -> None:
        self.model = None
        self.tokenizer = None
        self._available = False

        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.warning(
                f"Semantic model not found at {model_path}. "
                "Semantic detector will return no findings. "
                "Train the model first (see model/train.py)."
            )
            return

        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
            import torch  # noqa: F401 — ensure torch is available

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.eval()
            self._available = True
            logger.info(f"Semantic model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the semantic model is loaded and ready."""
        return self._available

    async def detect(self, text: str) -> list[Finding]:
        """Run semantic NER inference. Returns empty list if model unavailable."""
        if not self._available:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._detect_sync, text)

    def _detect_sync(self, text: str) -> list[Finding]:
        """Synchronous inference — runs in thread pool."""
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # matches training max_length
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = F.softmax(logits, dim=-1)  # real probabilities per token
        predictions = torch.argmax(logits, dim=-1)  # (seq_len,)

        findings: list[Finding] = []
        current_entity = None
        current_start = None
        current_end = None
        current_token_probs: list[float] = []

        for i, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.model.config.id2label[pred.item()]
            char_start = offset[0].item()
            char_end = offset[1].item()
            token_prob = probs[i][
                pred.item()
            ].item()  # softmax prob for predicted label

            # Skip special tokens ([CLS], [SEP], [PAD])
            if char_start == 0 and char_end == 0 and i > 0:
                if current_entity:
                    findings.append(
                        self._make_finding(
                            text,
                            current_entity,
                            current_start,
                            current_end,
                            current_token_probs,
                        )
                    )
                    current_entity = None
                    current_token_probs = []
                continue

            if label.startswith("B-"):
                # Flush previous span
                if current_entity:
                    findings.append(
                        self._make_finding(
                            text,
                            current_entity,
                            current_start,
                            current_end,
                            current_token_probs,
                        )
                    )
                current_entity = label[2:]  # "PII" or "SECRET"
                current_start = char_start
                current_end = char_end
                current_token_probs = [token_prob]

            elif label.startswith("I-") and current_entity:
                current_end = char_end
                current_token_probs.append(token_prob)

            else:
                if current_entity:
                    findings.append(
                        self._make_finding(
                            text,
                            current_entity,
                            current_start,
                            current_end,
                            current_token_probs,
                        )
                    )
                    current_entity = None
                    current_token_probs = []

        # Flush final span
        if current_entity:
            findings.append(
                self._make_finding(
                    text,
                    current_entity,
                    current_start,
                    current_end,
                    current_token_probs,
                )
            )

        return findings

    def _make_finding(
        self,
        text: str,
        entity_label: str,
        start: int,
        end: int,
        token_probs: list[float],
    ) -> Finding:
        """
        Create a Finding with confidence derived from actual model probabilities.

        Confidence = mean softmax probability across all tokens in the span.
        This gives genuinely calibrated scores — uncertain detections (e.g. a
        cloud service name that superficially resembles a secret) will score
        lower than clear-cut cases (explicit 'password is X' patterns).
        """
        entity_type = _SEMANTIC_ENTITY_MAP.get(entity_label, EntityType.GENERIC_PII)
        category = _SEMANTIC_CATEGORY_MAP.get(entity_label, EntityCategory.PII)
        confidence = (
            round(sum(token_probs) / len(token_probs), 4) if token_probs else 0.5
        )

        return Finding(
            entity_type=entity_type,
            category=category,
            start=start,
            end=end,
            matched_text=text[start:end],
            confidence=confidence,
            detector="semantic",
        )
