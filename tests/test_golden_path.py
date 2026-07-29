"""
Golden-path end-to-end test: the actual story the console/gateway pair tells.

Create a blocking policy through the console API -> send a request through
the real gateway proxy path (/v1/chat/completions, not the bare /scan demo
endpoint) that trips it -> confirm the decision is visible in the audit view
-> confirm the Prometheus metric incremented. Every other test file checks
one layer in isolation; this one checks that the layers are actually wired
together the way a user would experience them.

Runs the real FastAPI app through its lifespan over an in-process ASGI
transport (same pattern as test_console_api.py) — no network, no Ollama.
The BLOCK path returns before any LLM backend is contacted, so this is safe
to run in CI with no LLM configured.

The before/after metrics comparison assumes nothing else increments the
BLOCK counter between those two reads, which holds under sequential
pytest but would need revisiting under a parallel runner (e.g. xdist).
"""

import os
import re
import tempfile

os.environ.setdefault(
    "SENTINELLM_DATABASE_URL",
    f"sqlite+aiosqlite:///{os.path.join(tempfile.gettempdir(), 'sentinellm_test_golden_path.db')}",
)

import httpx
import pytest
import pytest_asyncio
from app.config import settings
from sqlalchemy import delete

settings.semantic_model_enabled = False

from app.db import AuditLog, Policy, async_session_factory, init_db
from app.main import app, lifespan


@pytest_asyncio.fixture(autouse=True)
async def _reset_db():
    """Each test gets an empty audit_log and policies table, reseeded from
    YAML defaults on the next lifespan start.

    Also cleans up *after* the test, not just before: other test files use
    the same `os.environ.setdefault("SENTINELLM_DATABASE_URL", ...)`
    pattern, which only takes effect for whichever test module the pytest
    collector happens to import first in the process — every other
    module's setdefault becomes a no-op and it silently inherits this
    module's DB file. The policy this test creates (EMAIL -> BLOCK) must
    not leak into another file's assumptions about default policy behavior.
    """
    await init_db()
    async with async_session_factory() as session:
        await session.execute(delete(AuditLog))
        await session.execute(delete(Policy))
        await session.commit()
    yield
    async with async_session_factory() as session:
        await session.execute(delete(AuditLog))
        await session.execute(delete(Policy))
        await session.commit()


class _AppClient:
    """Runs the app lifespan for the duration of one `async with` block."""

    async def __aenter__(self):
        self._lifespan_cm = lifespan(app)
        await self._lifespan_cm.__aenter__()
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        )
        await self.client.__aenter__()
        return self.client

    async def __aexit__(self, *exc):
        await self.client.__aexit__(*exc)
        await self._lifespan_cm.__aexit__(*exc)


def app_client() -> _AppClient:
    return _AppClient()


def _block_count(metrics_text: str) -> float:
    """Parse `sentinellm_requests_total{decision="BLOCK"} N` out of the
    Prometheus text exposition format. 0 if the series hasn't been
    observed yet (prometheus_client omits unobserved label combinations)."""
    match = re.search(
        r'sentinellm_requests_total\{decision="BLOCK"\} ([\d.]+)', metrics_text
    )
    return float(match.group(1)) if match else 0.0


class TestGoldenPath:
    @pytest.mark.asyncio
    async def test_create_policy_trip_it_see_audit_and_metric(self):
        async with app_client() as client:
            # 1. Create a blocking policy through the console API — the
            # same call the console's Policies page makes. EMAIL already
            # has a default MASK rule from default.yaml; per
            # PolicyEngine.reload's documented last-updated-wins rule, this
            # newer row takes over EMAIL's effective action.
            create_resp = await client.post(
                "/api/v1/policies",
                json={
                    "name": "Golden Path Block Rule",
                    "entity_type": "EMAIL",
                    "action": "block",
                    "min_confidence": 0.5,
                    "enabled": True,
                },
            )
            assert create_resp.status_code == 201, create_resp.text
            assert create_resp.json()["action"] == "block"

            # 2. Capture the BLOCK counter before tripping the policy.
            before_metrics = (await client.get("/metrics")).text
            block_count_before = _block_count(before_metrics)

            # 3. Send a request through the real gateway proxy path (not
            # /scan) containing a plain, regex-catchable email — the BLOCK
            # path returns before any LLM backend would be contacted.
            scan_resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": "My email is jane.doe@example.com, please follow up.",
                        }
                    ],
                },
            )
            assert scan_resp.status_code == 403, scan_resp.text
            body = scan_resp.json()
            assert body["error"]["type"] == "policy_violation"
            assert body["error"]["code"] == "content_blocked"
            assert body["ppg"]["input_decision"] == "BLOCK"

            # 4. Confirm the decision is visible in the audit view — the
            # same query the console's Audit page runs. _reset_db gives
            # each test an empty audit_log, so this one tripped request is
            # the only BLOCK row.
            audit_resp = await client.get(
                "/api/v1/audit", params={"decision": "block", "limit": 10}
            )
            assert audit_resp.status_code == 200
            items = audit_resp.json()["items"]
            assert len(items) == 1, (
                f"expected exactly one BLOCK record from the tripped "
                f"request, got {len(items)}"
            )
            record = items[0]
            assert record["decision"] == "block"
            assert "EMAIL" in record["entity_types"]
            assert "jane.doe@example.com" not in record["request_preview"]

            # 5. Confirm the Prometheus metric incremented by exactly one.
            after_metrics = (await client.get("/metrics")).text
            block_count_after = _block_count(after_metrics)
            assert block_count_after == block_count_before + 1
