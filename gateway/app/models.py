"""
Pydantic models for OpenAI-compatible chat completions API.

Accepts all standard OpenAI fields and passes through unknown fields
transparently — the gateway should be transparent for fields it doesn't
explicitly handle.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# --- Request Models ---

class ChatMessage(BaseModel):
    """A single message in the chat conversation."""
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Accepts all standard fields. Unknown fields are passed through
    to the LLM backend.
    """
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[str | list[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

    model_config = {"extra": "allow"}


# --- Response Models ---

class ChatMessageResponse(BaseModel):
    """A message in the chat completion response."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[Any]] = None

    model_config = {"extra": "allow"}


class ChatChoice(BaseModel):
    """A single choice in the chat completion response."""
    index: int
    message: ChatMessageResponse
    finish_reason: Optional[str] = None

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
    output_decision: Optional[str] = None
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
    usage: Optional[Usage] = None
    ppg: Optional[PPGMetadata] = None

    model_config = {"extra": "allow"}


class PolicyViolationResponse(BaseModel):
    """Response returned when a request is blocked by policy."""
    error: dict = Field(default_factory=dict)
    ppg: Optional[PPGMetadata] = None
