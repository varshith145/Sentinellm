"""
Tests for the console API (/api/v1/*): policy CRUD + dry-run, audit search +
pagination, and stats.

Runs the real FastAPI app through its lifespan over an in-process ASGI
transport, same pattern as test_scan.py — no network, no Ollama, no Postgres.
Semantic detector is disabled so the suite stays offline and fast; policy CRUD,
audit filtering, and stats logic don't depend on which detectors are active.

Audit rows for the audit/stats tests are inserted directly via the DB session
rather than driven through /v1/chat/completions — that full 9-step pipeline
is already covered by test_integration.py and test_streaming.py. Here we're
testing the console API's query/filter/pagination/aggregation logic in
isolation, with full control over decision mix and timestamps.
"""

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

os.environ.setdefault(
    "SENTINELLM_DATABASE_URL",
    f"sqlite+aiosqlite:///{os.path.join(tempfile.gettempdir(), 'sentinellm_test_console_audit.db')}",
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
    """
    Each test gets an empty audit_log and policies table. Emptying policies
    means every test's app lifespan reseeds the 11 defaults from YAML, so
    policy-count assertions stay meaningful regardless of test order.
    """
    await init_db()
    async with async_session_factory() as session:
        await session.execute(delete(AuditLog))
        await session.execute(delete(Policy))
        await session.commit()
    yield


async def _insert_audit_row(
    *,
    input_decision: str,
    output_decision: str | None,
    total_latency_ms: int,
    input_redactions: dict | None = None,
    output_redactions: dict | None = None,
    created_at: datetime | None = None,
    prompt_redacted: str = "[user]: hello",
    user_id: str = "anonymous",
) -> uuid.UUID:
    row_id = uuid.uuid4()
    async with async_session_factory() as session:
        session.add(
            AuditLog(
                id=row_id,
                request_id=uuid.uuid4(),
                created_at=created_at or datetime.now(timezone.utc),
                user_id=user_id,
                model="test-model",
                input_decision=input_decision,
                output_decision=output_decision,
                policy_id="default-v1",
                reasons=["test reason"],
                input_redactions=input_redactions or {},
                output_redactions=output_redactions or {},
                prompt_redacted=prompt_redacted,
                response_redacted=None,
                prompt_hash="0" * 64,
                detection_latency_ms=total_latency_ms,
                llm_latency_ms=None,
                total_latency_ms=total_latency_ms,
            )
        )
        await session.commit()
    return row_id


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


# --- Policies ---


class TestPolicyList:
    @pytest.mark.asyncio
    async def test_seeded_defaults_are_present(self):
        async with app_client() as client:
            resp = await client.get("/api/v1/policies")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["items"]) == 11  # default.yaml has 11 rules
            entity_types = {p["entity_type"] for p in data["items"]}
            assert "AWS_KEY" in entity_types
            assert "EMAIL" in entity_types


class TestPolicyCreate:
    @pytest.mark.asyncio
    async def test_create_returns_201_and_shape(self):
        async with app_client() as client:
            resp = await client.post(
                "/api/v1/policies",
                json={
                    "name": "Block IBAN-like secrets",
                    "entity_type": "GENERIC_SECRET",
                    "action": "block",
                    "min_confidence": 0.9,
                },
            )
            assert resp.status_code == 201
            body = resp.json()
            assert body["name"] == "Block IBAN-like secrets"
            assert body["category"] == "SECRET"
            assert body["enabled"] is True
            assert "id" in body

    @pytest.mark.asyncio
    async def test_create_rejects_missing_name(self):
        async with app_client() as client:
            resp = await client.post(
                "/api/v1/policies",
                json={"entity_type": "EMAIL", "action": "mask"},
            )
            assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_rejects_invalid_entity_type(self):
        async with app_client() as client:
            resp = await client.post(
                "/api/v1/policies",
                json={"name": "x", "entity_type": "NOT_A_REAL_TYPE", "action": "mask"},
            )
            assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_created_policy_affects_live_evaluation(self):
        """
        The whole point of DB-authoritative policy: a rule change actually
        flips /scan's decision, not just a 201 on the write. Uses EMAIL
        (regex-detected, always active) rather than GENERIC_PII, which only
        the semantic detector emits — disabled in this test suite.
        """
        async with app_client() as client:
            before = await client.post("/scan", json={"text": "contact me at a@b.com"})
            assert before.json()["decision"] == "MASK"

            policies = (await client.get("/api/v1/policies")).json()["items"]
            email_rule = next(p for p in policies if p["entity_type"] == "EMAIL")

            resp = await client.patch(
                f"/api/v1/policies/{email_rule['id']}", json={"action": "block"}
            )
            assert resp.status_code == 200

            after = await client.post("/scan", json={"text": "contact me at a@b.com"})
            assert after.json()["decision"] == "BLOCK"


class TestPolicyGetUpdateDelete:
    @pytest.mark.asyncio
    async def test_get_missing_returns_404(self):
        async with app_client() as client:
            resp = await client.get(f"/api/v1/policies/{uuid.uuid4()}")
            assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_patch_toggles_enabled(self):
        async with app_client() as client:
            created = (
                await client.post(
                    "/api/v1/policies",
                    json={"name": "x", "entity_type": "PASSWORD", "action": "block"},
                )
            ).json()

            resp = await client.patch(
                f"/api/v1/policies/{created['id']}", json={"enabled": False}
            )
            assert resp.status_code == 200
            assert resp.json()["enabled"] is False

            fetched = await client.get(f"/api/v1/policies/{created['id']}")
            assert fetched.json()["enabled"] is False

    @pytest.mark.asyncio
    async def test_delete_then_get_is_404(self):
        async with app_client() as client:
            created = (
                await client.post(
                    "/api/v1/policies",
                    json={"name": "x", "entity_type": "JWT", "action": "block"},
                )
            ).json()

            resp = await client.delete(f"/api/v1/policies/{created['id']}")
            assert resp.status_code == 204

            resp = await client.get(f"/api/v1/policies/{created['id']}")
            assert resp.status_code == 404


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_against_existing_aws_key_rule(self):
        async with app_client() as client:
            policies = (await client.get("/api/v1/policies")).json()["items"]
            aws_rule = next(p for p in policies if p["entity_type"] == "AWS_KEY")

            resp = await client.post(
                f"/api/v1/policies/{aws_rule['id']}/dry-run",
                json={"text": "my key is AKIAIOSFODNN7EXAMPLE"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["decision"] == "block"
            assert any(f["entity_type"] == "AWS_KEY" for f in body["findings"])
            assert "AKIAIOSFODNN7EXAMPLE" not in body["redacted_text"]

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write_audit_record(self):
        async with app_client() as client:
            policies = (await client.get("/api/v1/policies")).json()["items"]
            aws_rule = next(p for p in policies if p["entity_type"] == "AWS_KEY")

            await client.post(
                f"/api/v1/policies/{aws_rule['id']}/dry-run",
                json={"text": "AKIAIOSFODNN7EXAMPLE"},
            )
            audit = (await client.get("/api/v1/audit")).json()
            assert audit["items"] == []

    @pytest.mark.asyncio
    async def test_dry_run_missing_policy_404(self):
        async with app_client() as client:
            resp = await client.post(
                f"/api/v1/policies/{uuid.uuid4()}/dry-run", json={"text": "hi"}
            )
            assert resp.status_code == 404


# --- Auth ---


class TestConsoleAuth:
    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        assert settings.console_api_key == ""
        async with app_client() as client:
            resp = await client.get("/api/v1/policies")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_enforced_when_key_set(self):
        settings.console_api_key = "secret123"
        try:
            async with app_client() as client:
                resp = await client.get("/api/v1/policies")
                assert resp.status_code == 401

                resp = await client.get(
                    "/api/v1/policies", headers={"X-API-Key": "wrong"}
                )
                assert resp.status_code == 401

                resp = await client.get(
                    "/api/v1/policies", headers={"X-API-Key": "secret123"}
                )
                assert resp.status_code == 200
        finally:
            settings.console_api_key = ""


class TestConsoleReadOnly:
    """The public HF Spaces demo guard: writes 403, reads and dry-run stay open."""

    @pytest.mark.asyncio
    async def test_writes_blocked_when_read_only(self):
        settings.console_read_only = True
        try:
            async with app_client() as client:
                create = await client.post(
                    "/api/v1/policies",
                    json={"name": "x", "entity_type": "EMAIL", "action": "mask"},
                )
                assert create.status_code == 403

                policies = (await client.get("/api/v1/policies")).json()["items"]
                email_rule = next(p for p in policies if p["entity_type"] == "EMAIL")

                patch = await client.patch(
                    f"/api/v1/policies/{email_rule['id']}", json={"enabled": False}
                )
                assert patch.status_code == 403

                delete = await client.delete(f"/api/v1/policies/{email_rule['id']}")
                assert delete.status_code == 403
        finally:
            settings.console_read_only = False

    @pytest.mark.asyncio
    async def test_reads_and_dry_run_stay_open_when_read_only(self):
        settings.console_read_only = True
        try:
            async with app_client() as client:
                assert (await client.get("/api/v1/policies")).status_code == 200
                assert (await client.get("/api/v1/audit")).status_code == 200
                assert (await client.get("/api/v1/stats/summary")).status_code == 200

                policies = (await client.get("/api/v1/policies")).json()["items"]
                aws_rule = next(p for p in policies if p["entity_type"] == "AWS_KEY")
                dry_run = await client.post(
                    f"/api/v1/policies/{aws_rule['id']}/dry-run",
                    json={"text": "AKIAIOSFODNN7EXAMPLE"},
                )
                assert dry_run.status_code == 200
        finally:
            settings.console_read_only = False

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        assert settings.console_read_only is False
        async with app_client() as client:
            resp = await client.post(
                "/api/v1/policies",
                json={"name": "x", "entity_type": "PHONE", "action": "mask"},
            )
            assert resp.status_code == 201


# --- Audit ---


class TestAuditList:
    @pytest.mark.asyncio
    async def test_returns_inserted_rows(self):
        await _insert_audit_row(
            input_decision="BLOCK",
            output_decision=None,
            total_latency_ms=42,
            input_redactions={"AWS_KEY": 1},
        )
        async with app_client() as client:
            resp = await client.get("/api/v1/audit")
            assert resp.status_code == 200
            items = resp.json()["items"]
            assert len(items) == 1
            assert items[0]["decision"] == "block"
            assert items[0]["entity_types"] == ["AWS_KEY"]
            assert items[0]["latency_ms"] == 42

    @pytest.mark.asyncio
    async def test_filter_by_decision(self):
        await _insert_audit_row(
            input_decision="ALLOW", output_decision="ALLOW", total_latency_ms=10
        )
        await _insert_audit_row(
            input_decision="BLOCK",
            output_decision=None,
            total_latency_ms=20,
            input_redactions={"GITHUB_TOKEN": 1},
        )
        async with app_client() as client:
            resp = await client.get("/api/v1/audit?decision=block")
            items = resp.json()["items"]
            assert len(items) == 1
            assert items[0]["decision"] == "block"

    @pytest.mark.asyncio
    async def test_filter_by_rule_id_matches_entity_type(self):
        await _insert_audit_row(
            input_decision="MASK",
            output_decision="ALLOW",
            total_latency_ms=15,
            input_redactions={"EMAIL": 2},
        )
        await _insert_audit_row(
            input_decision="MASK",
            output_decision="ALLOW",
            total_latency_ms=15,
            input_redactions={"SSN": 1},
        )
        async with app_client() as client:
            resp = await client.get("/api/v1/audit?rule_id=EMAIL")
            items = resp.json()["items"]
            assert len(items) == 1
            assert items[0]["entity_types"] == ["EMAIL"]

    @pytest.mark.asyncio
    async def test_free_text_search(self):
        await _insert_audit_row(
            input_decision="ALLOW",
            output_decision="ALLOW",
            total_latency_ms=5,
            prompt_redacted="[user]: what is the weather in Boston",
        )
        await _insert_audit_row(
            input_decision="ALLOW",
            output_decision="ALLOW",
            total_latency_ms=5,
            prompt_redacted="[user]: summarize this document",
        )
        async with app_client() as client:
            resp = await client.get("/api/v1/audit?q=Boston")
            items = resp.json()["items"]
            assert len(items) == 1
            assert "Boston" in items[0]["request_preview"]

    @pytest.mark.asyncio
    async def test_cursor_pagination_covers_all_rows_without_duplicates(self):
        now = datetime.now(timezone.utc)
        for i in range(5):
            await _insert_audit_row(
                input_decision="ALLOW",
                output_decision="ALLOW",
                total_latency_ms=i,
                created_at=now - timedelta(seconds=i),
            )
        async with app_client() as client:
            seen_ids = []
            cursor = None
            for _ in range(10):  # generous upper bound on page count
                url = "/api/v1/audit?limit=2"
                if cursor:
                    url += f"&cursor={cursor}"
                resp = await client.get(url)
                body = resp.json()
                seen_ids.extend(item["id"] for item in body["items"])
                cursor = body["next_cursor"]
                if not cursor:
                    break
            assert len(seen_ids) == 5
            assert len(set(seen_ids)) == 5  # no duplicates across pages

    @pytest.mark.asyncio
    async def test_detail_endpoint_returns_full_record(self):
        row_id = await _insert_audit_row(
            input_decision="MASK",
            output_decision="ALLOW",
            total_latency_ms=33,
            input_redactions={"PHONE": 1},
            prompt_redacted="[user]: call me at [REDACTED_PHONE]",
        )
        async with app_client() as client:
            resp = await client.get(f"/api/v1/audit/{row_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["prompt_redacted"] == "[user]: call me at [REDACTED_PHONE]"
            assert body["input_decision"] == "mask"
            assert body["output_decision"] == "allow"
            assert body["total_latency_ms"] == 33

    @pytest.mark.asyncio
    async def test_detail_missing_returns_404(self):
        async with app_client() as client:
            resp = await client.get(f"/api/v1/audit/{uuid.uuid4()}")
            assert resp.status_code == 404


# --- Stats ---


class TestStatsSummary:
    @pytest.mark.asyncio
    async def test_counts_and_percentiles(self):
        for ms, decision in [(10, "ALLOW"), (20, "MASK"), (30, "MASK"), (100, "BLOCK")]:
            await _insert_audit_row(
                input_decision=decision,
                output_decision=None if decision == "BLOCK" else "ALLOW",
                total_latency_ms=ms,
                input_redactions={"EMAIL": 1} if decision == "MASK" else {},
            )
        async with app_client() as client:
            resp = await client.get("/api/v1/stats/summary?window=24h")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 4
            assert body["by_decision"] == {"allow": 1, "mask": 2, "block": 1}
            assert body["p50_ms"] > 0
            assert body["top_rules"][0][0] == "EMAIL"

    @pytest.mark.asyncio
    async def test_invalid_window_is_400(self):
        async with app_client() as client:
            resp = await client.get("/api/v1/stats/summary?window=bogus")
            assert resp.status_code == 400


class TestStatsTimeseries:
    @pytest.mark.asyncio
    async def test_bucket_count_matches_window_and_bucket_size(self):
        async with app_client() as client:
            resp = await client.get("/api/v1/stats/timeseries?window=1h&bucket=10m")
            assert resp.status_code == 200
            points = resp.json()
            assert len(points) == 7  # 60min / 10min + 1

    @pytest.mark.asyncio
    async def test_rows_land_in_correct_bucket(self):
        now = datetime.now(timezone.utc)
        await _insert_audit_row(
            input_decision="BLOCK",
            output_decision=None,
            total_latency_ms=50,
            created_at=now,
        )
        async with app_client() as client:
            resp = await client.get("/api/v1/stats/timeseries?window=10m&bucket=5m")
            points = resp.json()
            assert sum(p["block"] for p in points) == 1

    @pytest.mark.asyncio
    async def test_bucket_larger_than_window_is_400(self):
        async with app_client() as client:
            resp = await client.get("/api/v1/stats/timeseries?window=5m&bucket=1h")
            assert resp.status_code == 400
