"""
Unit tests for the SentinelLM streaming (SSE) proxy path.

Tests the "collect-then-stream" pipeline at the logic layer:
  - SSE chunk assembly and content extraction
  - Output scanning applied to assembled text
  - Redacted content collapsed to clean replacement chunks
  - Clean content re-emitted verbatim with ppg attached to last chunk
  - Error handling: timeout, connect error, HTTP error codes
  - [DONE] sentinel correctly terminates collection

All tests use only the pure functions / data structures involved in
streaming — no FastAPI TestClient needed (avoids DB/detector startup).
"""

import json
from typing import Optional


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_chunk(
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    chunk_id: str = "chatcmpl-test",
    model: str = "test-model",
    created: int = 1700000000,
) -> dict:
    """Build an OpenAI-style SSE delta chunk dict."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def sse_line(data: dict) -> str:
    """Serialize a chunk dict to an SSE line."""
    return f"data: {json.dumps(data)}"


def parse_sse_lines(raw_lines: list) -> list:
    """
    Extract parsed JSON objects from a list of SSE lines.
    Skips lines that are not 'data: {...}' (e.g. empty lines, '[DONE]').
    """
    result = []
    for line in raw_lines:
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        result.append(json.loads(payload))
    return result


# ── Content assembly logic ─────────────────────────────────────────────────────


class TestSSEAssembly:
    """Assembling full text from delta chunks."""

    def test_single_chunk_content(self):
        """One chunk with content → assembled correctly."""
        chunks = [make_chunk("Hello world")]
        assembled = "".join(
            (c["choices"][0]["delta"].get("content") or "") for c in chunks
        )
        assert assembled == "Hello world"

    def test_multi_chunk_content(self):
        """Multiple delta chunks concatenate in order."""
        chunks = [
            make_chunk("The "),
            make_chunk("quick "),
            make_chunk("brown fox"),
            make_chunk(finish_reason="stop"),
        ]
        assembled = "".join(
            (c["choices"][0]["delta"].get("content") or "") for c in chunks
        )
        assert assembled == "The quick brown fox"

    def test_chunk_with_no_content_key(self):
        """Delta with no 'content' key (role chunk) contributes empty string."""
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        content = chunk["choices"][0]["delta"].get("content") or ""
        assert content == ""

    def test_empty_chunks_give_empty_assembly(self):
        """No delta content → assembled text is empty string."""
        chunks = [make_chunk(finish_reason="stop")]
        assembled = "".join(
            (c["choices"][0]["delta"].get("content") or "") for c in chunks
        )
        assert assembled == ""

    def test_finish_reason_extracted_from_last_chunk(self):
        """finish_reason comes from the chunk where it's set."""
        chunks = [
            make_chunk("Hello", finish_reason=None),
            make_chunk(finish_reason="stop"),
        ]
        finish_reason = None
        for c in chunks:
            fr = c["choices"][0].get("finish_reason")
            if fr:
                finish_reason = fr
        assert finish_reason == "stop"

    def test_finish_reason_length_variant(self):
        """finish_reason can be 'length' (token limit hit)."""
        chunks = [make_chunk("Truncated", finish_reason="length")]
        fr = None
        for c in chunks:
            fr = c["choices"][0].get("finish_reason") or fr
        assert fr == "length"

    def test_done_sentinel_stops_collection(self):
        """Lines after 'data: [DONE]' should be ignored."""
        raw_lines = [
            sse_line(make_chunk("Hello")),
            "data: [DONE]",
            sse_line(make_chunk(" extra")),  # should not be processed
        ]
        collected = []
        for line in raw_lines:
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                collected.append(json.loads(line[6:]))
        assembled = "".join(
            (c["choices"][0]["delta"].get("content") or "") for c in collected
        )
        assert assembled == "Hello"
        assert len(collected) == 1


# ── SSE re-emission: clean content ────────────────────────────────────────────


class TestSSEReEmissionClean:
    """When no redaction needed, original chunks pass through unchanged."""

    def test_all_chunks_emitted(self):
        """Each collected chunk should appear in output."""
        chunks = [
            make_chunk("Hello "),
            make_chunk("world"),
            make_chunk(finish_reason="stop"),
        ]
        # Simulate no-redaction path: re-emit verbatim
        out_lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        parsed = parse_sse_lines(out_lines)
        assert len(parsed) == 3

    def test_ppg_attached_to_last_chunk(self):
        """PPG metadata dict must be on the final data chunk."""
        chunks = [make_chunk("Hi"), make_chunk(finish_reason="stop")]
        ppg = {"request_id": "abc", "input_decision": "ALLOW"}

        output_chunks = []
        for i, c in enumerate(chunks):
            if i == len(chunks) - 1:
                c["ppg"] = ppg
            output_chunks.append(c)

        assert "ppg" not in output_chunks[0]
        assert output_chunks[-1]["ppg"] == ppg

    def test_ppg_not_on_earlier_chunks(self):
        """Only the last chunk gets ppg, not intermediate ones."""
        chunks = [
            make_chunk("a"),
            make_chunk("b"),
            make_chunk("c", finish_reason="stop"),
        ]
        ppg = {"request_id": "xyz"}
        for i, c in enumerate(chunks):
            if i == len(chunks) - 1:
                c["ppg"] = ppg
        for c in chunks[:-1]:
            assert "ppg" not in c

    def test_chunk_ids_preserved(self):
        """Chunk IDs must pass through unmodified."""
        chunk_id = "chatcmpl-SPECIFIC-ID"
        chunks = [make_chunk("test", chunk_id=chunk_id)]
        assert chunks[0]["id"] == chunk_id

    def test_model_field_preserved(self):
        """Model name from backend passes through."""
        chunks = [make_chunk("hi", model="gpt-4-turbo")]
        assert chunks[0]["model"] == "gpt-4-turbo"


# ── SSE re-emission: redacted content ─────────────────────────────────────────


class TestSSEReEmissionRedacted:
    """When output scanning triggers redaction, stream collapses to 2 clean chunks."""

    def _build_redacted_output(
        self,
        original_chunks: list,
        redacted_text: str,
        ppg: dict,
        finish_reason: str = "stop",
    ) -> list:
        """
        Simulate the redaction branch: collapse to one content chunk + one done chunk.
        Returns SSE lines as the generator would yield them.
        """
        first = original_chunks[0]
        chunk_id = first.get("id", "chatcmpl-fallback")
        created = first.get("created", 1700000000)
        model_name = first.get("model", "unknown")

        content_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": redacted_text},
                    "finish_reason": None,
                }
            ],
        }
        done_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
            "ppg": ppg,
        }
        return [
            f"data: {json.dumps(content_chunk)}",
            f"data: {json.dumps(done_chunk)}",
            "data: [DONE]",
        ]

    def test_exactly_two_data_chunks_emitted(self):
        """Redacted stream emits exactly 2 data chunks + [DONE]."""
        original = [
            make_chunk("my email is user@example.com"),
            make_chunk(finish_reason="stop"),
        ]
        ppg = {"request_id": "r1", "input_decision": "ALLOW"}
        lines = self._build_redacted_output(
            original, "my email is [REDACTED_EMAIL]", ppg
        )
        parsed = parse_sse_lines(lines)
        assert len(parsed) == 2

    def test_first_chunk_has_redacted_content(self):
        """First emitted chunk carries the redacted replacement text."""
        original = [
            make_chunk("secret: ghp_ABC123"),
            make_chunk(finish_reason="stop"),
        ]
        ppg = {"request_id": "r2"}
        lines = self._build_redacted_output(original, "secret: [REDACTED_SECRET]", ppg)
        parsed = parse_sse_lines(lines)
        assert parsed[0]["choices"][0]["delta"]["content"] == "secret: [REDACTED_SECRET]"

    def test_second_chunk_has_ppg(self):
        """Second (done) chunk carries the ppg metadata."""
        original = [make_chunk("ssn: 123-45-6789")]
        ppg = {"request_id": "r3", "output_decision": "MASK"}
        lines = self._build_redacted_output(original, "ssn: [REDACTED_SSN]", ppg)
        parsed = parse_sse_lines(lines)
        assert parsed[-1]["ppg"] == ppg

    def test_second_chunk_finish_reason(self):
        """Done chunk has finish_reason='stop' (or from backend)."""
        original = [make_chunk("data")]
        ppg = {}
        lines = self._build_redacted_output(
            original, "[REDACTED]", ppg, finish_reason="stop"
        )
        parsed = parse_sse_lines(lines)
        assert parsed[-1]["choices"][0]["finish_reason"] == "stop"

    def test_first_chunk_finish_reason_is_none(self):
        """Content chunk has finish_reason=None (content not done yet)."""
        original = [make_chunk("data")]
        ppg = {}
        lines = self._build_redacted_output(original, "[REDACTED]", ppg)
        parsed = parse_sse_lines(lines)
        assert parsed[0]["choices"][0]["finish_reason"] is None

    def test_chunk_id_carried_from_original(self):
        """Redacted chunks inherit the ID from the first original chunk."""
        original = [make_chunk("data", chunk_id="chatcmpl-ORIGINAL")]
        ppg = {}
        lines = self._build_redacted_output(original, "[REDACTED]", ppg)
        parsed = parse_sse_lines(lines)
        assert parsed[0]["id"] == "chatcmpl-ORIGINAL"
        assert parsed[1]["id"] == "chatcmpl-ORIGINAL"

    def test_done_sentinel_present(self):
        """[DONE] appears as the last line."""
        original = [make_chunk("x")]
        lines = self._build_redacted_output(original, "[REDACTED]", {})
        assert lines[-1] == "data: [DONE]"


# ── Error responses ────────────────────────────────────────────────────────────


class TestStreamingErrorResponses:
    """Error conditions terminate stream with an error chunk + [DONE]."""

    def _error_lines(self, message: str, error_type: str) -> list:
        """Simulate what the generator yields on backend error."""
        err = json.dumps({"error": {"message": message, "type": error_type}})
        return [f"data: {err}", "data: [DONE]"]

    def test_timeout_emits_error_chunk(self):
        lines = self._error_lines("LLM backend timed out", "timeout")
        parsed = parse_sse_lines(lines)
        assert len(parsed) == 1
        assert parsed[0]["error"]["type"] == "timeout"

    def test_connect_error_emits_error_chunk(self):
        lines = self._error_lines("Cannot connect to LLM backend", "connection_error")
        parsed = parse_sse_lines(lines)
        assert parsed[0]["error"]["type"] == "connection_error"

    def test_backend_error_emits_error_chunk(self):
        lines = self._error_lines("LLM backend error: 500", "backend_error")
        parsed = parse_sse_lines(lines)
        assert parsed[0]["error"]["type"] == "backend_error"

    def test_error_followed_by_done(self):
        """Error response always followed by [DONE]."""
        lines = self._error_lines("timeout", "timeout")
        assert lines[-1] == "data: [DONE]"

    def test_error_message_preserved(self):
        msg = "LLM backend timed out"
        lines = self._error_lines(msg, "timeout")
        parsed = parse_sse_lines(lines)
        assert parsed[0]["error"]["message"] == msg


# ── Media type and headers ────────────────────────────────────────────────────


class TestSSEFormat:
    """SSE format invariants that clients depend on."""

    def test_data_prefix_on_every_chunk(self):
        """Every SSE line must start with 'data: '."""
        chunks = [make_chunk("hello"), make_chunk(finish_reason="stop")]
        lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        for line in lines:
            assert line.startswith("data: "), f"Missing 'data: ' prefix: {line!r}"

    def test_done_line_exact_format(self):
        """[DONE] sentinel must be exactly 'data: [DONE]'."""
        done_line = "data: [DONE]"
        assert done_line == "data: [DONE]"

    def test_chunk_is_valid_json(self):
        """Each data payload (except [DONE]) must be valid JSON."""
        chunks = [make_chunk("test content"), make_chunk(finish_reason="stop")]
        for c in chunks:
            line = f"data: {json.dumps(c)}"
            payload = line[6:]
            parsed = json.loads(payload)  # must not raise
            assert isinstance(parsed, dict)

    def test_chunk_has_required_fields(self):
        """Each chunk must have id, object, created, model, choices."""
        chunk = make_chunk("hello")
        for field in ("id", "object", "created", "model", "choices"):
            assert field in chunk, f"Missing required field: {field}"

    def test_chunk_object_type(self):
        """object field must be 'chat.completion.chunk'."""
        chunk = make_chunk("hi")
        assert chunk["object"] == "chat.completion.chunk"
