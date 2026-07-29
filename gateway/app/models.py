"""
Pydantic models for OpenAI-compatible chat completions API.

Accepts all standard OpenAI fields and passes through unknown fields
transparently — the gateway should be transparent for fields it doesn't
explicitly handle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --- Request Models ---


class ChatMessage(BaseModel):
    """A single message in the chat conversation."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Accepts all standard fields. Unknown fields are passed through
    to the LLM backend.
    """

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None

    model_config = {"extra": "allow"}


# --- Response Models ---


class ChatMessageResponse(BaseModel):
    """A message in the chat completion response."""

    role: str
    content: str | None = None
    tool_calls: list[Any] | None = None

    model_config = {"extra": "allow"}


class ChatChoice(BaseModel):
    """A single choice in the chat completion response."""

    index: int
    message: ChatMessageResponse
    finish_reason: str | None = None

    model_config = {"extra": "allow"}


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    model_config = {"extra": "allow"}


class PPGMetadata(BaseModel):
    """
    SentinelLM metadata block appended to responses.

    Contains scanning decisions, redaction counts, and latency info.
    """

    request_id: str
    input_decision: str
    output_decision: str | None = None
    input_redactions: dict[str, int] = Field(default_factory=dict)
    output_redactions: dict[str, int] = Field(default_factory=dict)
    policy_id: str
    detectors_used: list[str] = Field(default_factory=list)
    latency_ms: dict[str, int | None] = Field(default_factory=dict)


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response with SentinelLM metadata.
    """

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: Usage | None = None
    ppg: PPGMetadata | None = None

    model_config = {"extra": "allow"}


class PolicyViolationResponse(BaseModel):
    """Response returned when a request is blocked by policy."""

    error: dict = Field(default_factory=dict)
    ppg: PPGMetadata | None = None
