"""
Tests for the self-contained SQLite audit fallback used on Hugging Face Spaces.

Verifies that:
  - the default database URL is async SQLite (no Postgres required),
  - tables are created on a fresh SQLite file with no manual migration,
  - an audit record can be written and read back,
  - audit failures are non-fatal (a broken session does not raise).
"""

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.audit import write_audit_record
from app.config import settings
from app.db import AuditLog, Base


def test_default_database_url_is_sqlite():
    assert settings.database_url.startswith("sqlite+aiosqlite")


@pytest.mark.asyncio
async def test_tables_create_and_audit_roundtrips_on_fresh_sqlite(tmp_path):
    db_file = tmp_path / "audit_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}")
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    # Fresh file → tables must be created automatically.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    req_id = uuid.uuid4()
    async with sessionmaker() as session:
        await write_audit_record(
            session=session,
            request_id=req_id,
            user_id="tester",
            model="demo",
            input_decision="MASK",
            output_decision="ALLOW",
            policy_id="default-v1",
            reasons=["MASK: EMAIL detected"],
            input_redactions={"EMAIL": 1},
            output_redactions={},
            prompt_redacted="contact [REDACTED_EMAIL]",
            response_redacted="ok",
            detection_latency_ms=5,
            llm_latency_ms=None,
            total_latency_ms=7,
        )

    async with sessionmaker() as session:
        rows = (await session.execute(select(AuditLog))).scalars().all()

    assert len(rows) == 1
    row = rows[0]
    assert row.request_id == req_id
    assert row.input_decision == "MASK"
    assert row.input_redactions == {"EMAIL": 1}  # JSON column roundtrips
    assert row.prompt_hash  # sha-256 computed

    await engine.dispose()


@pytest.mark.asyncio
async def test_audit_write_is_non_fatal_on_broken_session():
    class BrokenSession:
        def add(self, _):
            raise RuntimeError("db is down")

        async def rollback(self):
            return None

    # Must not raise — audit is best-effort.
    await write_audit_record(
        session=BrokenSession(),
        request_id=uuid.uuid4(),
        user_id="tester",
        model="demo",
        input_decision="ALLOW",
        output_decision=None,
        policy_id="default-v1",
        reasons=[],
        input_redactions={},
        output_redactions={},
        prompt_redacted="hello",
        response_redacted=None,
        detection_latency_ms=1,
        llm_latency_ms=None,
        total_latency_ms=1,
    )
