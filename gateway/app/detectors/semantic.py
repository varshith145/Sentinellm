"""
Pass 3: Semantic detector using a fine-tuned DistilBERT NER model.

Catches obfuscated and informal PII that regex and Presidio miss, such as:
  - "john at gmail dot com"
  - "my social is four five six 78 9012"
  - "the password is hunter2"

The model performs token-level classification with BIO tags:
  O, B-PII, I-PII, B-SECRET, I-SECRET

Three inference backends, selected by SENTINELLM_INFERENCE_BACKEND:
  - "torch" (default): loads the model directly via transformers, as before.
  - "onnx": runs the graph exported by model/export_onnx.py through ONNX
    Runtime. Requires that export to have been run first — no ONNX build is
    published to the Hub, only the PyTorch one.
  - "triton": sends the same ONNX graph to a Triton Inference Server over
    gRPC (see triton_deploy/). Tokenization still happens in-process — Triton
    only serves the model graph, not the tokenizer — using the tokenizer
    files at onnx_model_path (Triton has no use for them itself).

All three backends share the same BIO-decoding logic (_decode) so behavior
can't drift between them independently of the model weights themselves —
see tests/test_onnx_parity.py for the torch/onnx numeric parity check this
rests on, and tests/test_triton_backend.py for the onnx/triton one.

If the model is not available, gracefully returns no findings.
"""

import asyncio
import logging
from pathlib import Path

import numpy as np
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

    Loads the model once at initialization. If the model isn't available,
    operates in stub mode (returns no findings).
    """

    def __init__(
        self,
        model_path: str = "./model/trained",
        model_id: str | None = None,
        inference_backend: str = "torch",
        onnx_model_path: str = "./model/onnx",
        triton_url: str = "localhost:8101",
        triton_model_name: str = "sentinellm",
    ) -> None:
        self.backend = inference_backend
        self.tokenizer = None
        self.model = None  # torch backend only
        self.session = None  # onnx backend only
        self.triton_client = None  # triton backend only
        self.triton_model_name = triton_model_name
        self.id2label: dict[int, str] = {}
        self._available = False

        if self.backend == "onnx":
            self._init_onnx(onnx_model_path)
        elif self.backend == "triton":
            self._init_triton(onnx_model_path, triton_url, triton_model_name)
        else:
            self._init_torch(model_path, model_id)

    def _init_torch(self, model_path: str, model_id: str | None) -> None:
        """
        Resolution order:
          1. Local directory at ``model_path`` (used for local/full-stack runs
             where the model was produced by ``model/train.py``).
          2. Hugging Face Hub model id ``model_id`` (used on Spaces, where the
             large model is not committed to git and is pulled from the Hub).
        """
        load_target: str | None = None
        source_desc: str = ""
        model_dir = Path(model_path)
        if model_dir.exists() and any(model_dir.iterdir()):
            load_target = str(model_path)
            source_desc = f"local path {model_path}"
        elif model_id:
            load_target = model_id
            source_desc = f"Hugging Face Hub id '{model_id}'"

        if load_target is None:
            logger.warning(
                f"Semantic model not found at {model_path} and no Hub id configured. "
                "Semantic detector will return no findings. "
                "Train the model first (see model/train.py) or set "
                "SENTINELLM_SEMANTIC_MODEL_ID."
            )
            return

        try:
            import torch
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
            )

            # Each request's inference already runs in its own thread
            # (loop.run_in_executor in detect() below). Left at its default,
            # torch also parallelizes each individual inference across every
            # core (intra-op threads) — so N concurrent requests spawn N x
            # num_cores threads competing for num_cores cores, thrashing
            # instead of scaling. Pinning to 1 makes the thread pool the only
            # source of parallelism.
            torch.set_num_threads(1)

            self.tokenizer = AutoTokenizer.from_pretrained(load_target)
            self.model = AutoModelForTokenClassification.from_pretrained(load_target)
            self.model.eval()
            self.id2label = self.model.config.id2label
            self._available = True
            logger.info(f"Semantic model loaded (torch backend) from {source_desc}")
        except Exception as e:  # noqa: BLE001 — intentional graceful fallback if the model can't load
            logger.warning(f"Failed to load semantic model from {source_desc}: {e}")

    def _init_onnx(self, onnx_model_path: str) -> None:
        onnx_dir = Path(onnx_model_path)
        model_file = onnx_dir / "model.onnx"

        if not model_file.exists():
            logger.warning(
                f"ONNX model not found at {onnx_model_path} (no model.onnx). "
                "Semantic detector will return no findings. Run "
                "`python model/export_onnx.py` first, or set "
                "SENTINELLM_INFERENCE_BACKEND=torch."
            )
            return

        try:
            import onnxruntime as ort
            from transformers import AutoConfig, AutoTokenizer

            session_options = ort.SessionOptions()
            # Same rationale as torch.set_num_threads(1) above: each request
            # already gets its own thread pool worker (run_in_executor), so
            # ONNX Runtime's own default of using every core per inference
            # call would have concurrent requests thrash instead of scale.
            # Pinning to 1 makes the thread pool the only source of
            # parallelism, deliberately — not left at the ORT default.
            session_options.intra_op_num_threads = 1

            self.session = ort.InferenceSession(
                str(model_file),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
            self.id2label = AutoConfig.from_pretrained(str(onnx_dir)).id2label
            self._available = True
            logger.info(f"Semantic model loaded (onnx backend) from {onnx_model_path}")
        except Exception as e:  # noqa: BLE001 — intentional graceful fallback if the model can't load
            logger.warning(
                f"Failed to load ONNX semantic model from {onnx_model_path}: {e}"
            )

    def _init_triton(
        self, onnx_model_path: str, triton_url: str, triton_model_name: str
    ) -> None:
        """
        Tokenizer/id2label still come from the local ONNX export directory —
        Triton serves the model graph only, so it has no tokenizer of its
        own to ask for. This means `model/export_onnx.py` must still be run
        first even though the graph itself is served remotely.
        """
        onnx_dir = Path(onnx_model_path)
        if not (onnx_dir / "model.onnx").exists():
            logger.warning(
                f"ONNX export not found at {onnx_model_path} (needed for tokenizer "
                "and id2label even though inference runs on Triton). Semantic "
                "detector will return no findings. Run `python model/export_onnx.py` "
                "first."
            )
            return

        try:
            # gRPC, not HTTP: Triton's gRPC endpoint uses a persistent HTTP/2
            # connection and protobuf framing, avoiding a new TCP handshake +
            # text-header parse per call that the HTTP client pays each time.
            # See triton_deploy/compare_http_grpc.py for the measured delta.
            import tritonclient.grpc as grpcclient
            from transformers import AutoConfig, AutoTokenizer

            client = grpcclient.InferenceServerClient(url=triton_url)
            if not client.is_server_ready():
                raise RuntimeError(f"Triton server at {triton_url} is not ready")
            if not client.is_model_ready(triton_model_name):
                raise RuntimeError(
                    f"Triton model '{triton_model_name}' is not ready at {triton_url}"
                )

            self.triton_client = client
            self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
            self.id2label = AutoConfig.from_pretrained(str(onnx_dir)).id2label
            self._available = True
            logger.info(
                f"Semantic model loaded (triton backend) — gRPC {triton_url}, "
                f"model '{triton_model_name}', tokenizer from {onnx_model_path}"
            )
        except Exception as e:  # noqa: BLE001 — intentional graceful fallback if Triton is unreachable
            logger.warning(f"Failed to connect to Triton at {triton_url}: {e}")

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
        if self.backend == "onnx":
            logits, offset_mapping = self._infer_onnx(text)
        elif self.backend == "triton":
            logits, offset_mapping = self._infer_triton(text)
        else:
            logits, offset_mapping = self._infer_torch(text)
        return self._decode(text, logits, offset_mapping)

    def _infer_torch(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,  # matches training max_length
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits[0].numpy(), offset_mapping

    def _infer_onnx(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,  # matches training max_length
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()

        input_names = {i.name for i in self.session.get_inputs()}
        feed = {k: v for k, v in inputs.items() if k in input_names}
        outputs = self.session.run(None, feed)

        return outputs[0][0], offset_mapping

    def _infer_triton(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        # self.triton_client (one instance, created in _init_triton) is
        # shared across every thread-pool worker calling this method
        # concurrently. That's intentional and safe: tritonclient's *gRPC*
        # client wraps a single grpc.Channel meant to be reused across
        # concurrent calls (that's the whole point of HTTP/2 multiplexing —
        # see triton_deploy/compare_http_grpc.py for why gRPC was chosen
        # over HTTP here in the first place). This differs from
        # triton_deploy/verify_client.py's HTTP client, which creates one
        # instance per thread — tritonclient's *HTTP* client is documented
        # as not safe to share across concurrent in-flight requests.
        import tritonclient.grpc as grpcclient

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=128,  # matches training max_length
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")

        input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
        attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)

        infer_inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        infer_inputs[0].set_data_from_numpy(input_ids)
        infer_inputs[1].set_data_from_numpy(attention_mask)

        outputs = [grpcclient.InferRequestedOutput("logits")]
        result = self.triton_client.infer(
            self.triton_model_name, inputs=infer_inputs, outputs=outputs
        )
        logits = result.as_numpy("logits")[0]  # drop batch dim -> (seq_len, num_labels)

        return logits, offset_mapping

    def _decode(
        self,
        text: str,
        logits: np.ndarray,
        offset_mapping: list[tuple[int, int]],
    ) -> list[Finding]:
        """BIO-decode raw logits into findings. Shared by both backends."""
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)  # real probabilities per token
        predictions = np.argmax(logits, axis=-1)

        findings: list[Finding] = []
        current_entity = None
        current_start = None
        current_end = None
        current_token_probs: list[float] = []

        for i, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.id2label[int(pred)]
            char_start, char_end = offset
            token_prob = float(probs[i][pred])  # softmax prob for predicted label

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
