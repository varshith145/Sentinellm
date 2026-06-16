"""
Tests for the public demo surface added for the Hugging Face Spaces deployment:

  - GET /            → serves the HTML demo page
  - POST /scan       → detect-only pipeline (no LLM), correct ALLOW/MASK/BLOCK
  - POST /v1/chat/completions in demo mode → clean 503, never a crash

These run the real FastAPI app through its lifespan (regex detector always on;
Presidio/semantic skipped if their heavy deps aren't installed) over an in-process
ASGI transport — no network, no Ollama, no Postgres.
"""

import os
import tempfile

# Point the audit DB at a private temp SQLite file BEFORE app.config is imported,
# so the engine (created at import time) uses a writable location independent of
# the working directory / mounted filesystem.
os.environ.setdefault(
    "SENTINELLM_DATABASE_URL",
    f"sqlite+aiosqlite:///{os.path.join(tempfile.gettempdir(), 'sentinellm_test_audit.db')}",
)

import httpx
import pytest

# Configure the singleton settings for demo mode BEFORE the app reads them at
# request/lifespan time. Semantic is disabled so the suite stays offline and fast.
from app.config import settings

settings.demo_mode = True
settings.semantic_model_enabled = False

from app.main import app, lifespan  # noqa: E402


async def _client_post(path: str, json: dict | None = None, method: str = "post"):
    """Run the app through its lifespan and issue one request in-process."""
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            if method == "get":
                return await client.get(path)
            return await client.post(path, json=json)


class TestScanEndpoint:
    """POST /scan must return the right decision with no LLM involved."""

    @pytest.mark.asyncio
    async def test_benign_text_allows(self):
        text = "There is nothing sensitive in this sentence."
        resp = await _client_post("/scan", {"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "ALLOW"
        assert data["findings"] == []
        assert data["redacted_text"] == text
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_email_masks_and_redacts(self):
        resp = await _client_post(
            "/scan", {"text": "Please contact alice@example.com for onboarding."}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "MASK"
        assert any(f["entity_type"] == "EMAIL" for f in data["findings"])
        assert "alice@example.com" not in data["redacted_text"]
        assert "[REDACTED_EMAIL]" in data["redacted_text"]

    @pytest.mark.asyncio
    async def test_aws_key_blocks(self):
        resp = await _client_post(
            "/scan", {"text": "export AWS_KEY=AKIAIOSFODNN7EXAMPLE"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "BLOCK"
        assert any(f["category"] == "SECRET" for f in data["findings"])

    @pytest.mark.asyncio
    async def test_finding_shape_is_serializable(self):
        resp = await _client_post(
            "/scan", {"text": "Call 555-867-5309 about the order."}
        )
        data = resp.json()
        for f in data["findings"]:
            assert {
                "entity_type",
                "category",
                "text",
                "start",
                "end",
                "confidence",
                "detector",
            } <= set(f.keys())


class TestDemoModeLLM:
    """In demo mode the LLM proxy must degrade gracefully, not crash."""

    @pytest.mark.asyncio
    async def test_chat_completions_returns_clean_503(self):
        resp = await _client_post(
            "/v1/chat/completions",
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["code"] == "llm_disabled"
        assert "/scan" in body["error"]["message"]


class TestDemoPage:
    @pytest.mark.asyncio
    async def test_root_serves_html(self):
        resp = await _client_post("/", method="get")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "SentinelLM" in resp.text
        assert "/scan" in resp.text
