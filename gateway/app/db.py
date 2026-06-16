"""
Database models and async engine setup for SentinelLM.

Uses SQLAlchemy 2.0 async with asyncpg for PostgreSQL.
Defines the AuditLog ORM model matching PRD Section 12.1.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Text, Integer, DateTime, String, JSON, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger("sentinellm")

# Portable column types: use native Postgres JSONB/UUID when on Postgres,
# but fall back to generic JSON / CHAR-backed UUID on SQLite so the same
# models work in the self-contained demo (no Postgres required).
JSONType = JSON().with_variant(JSONB(), "postgresql")
UUIDType = Uuid(as_uuid=True)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class AuditLog(Base):
    """
    Audit log record for every request through the gateway.

    Stores only redacted content — never raw PII or secrets.
    """

    __tablename__ = "audit_log"

    id = Column(UUIDType, primary_key=True, default=uuid.uuid4)
    request_id = Column(UUIDType, nullable=False, unique=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    user_id = Column(Text, nullable=False, default="anonymous")
    model = Column(Text, nullable=False)
    input_decision = Column(String(10), nullable=False)  # ALLOW, MASK, BLOCK
    output_decision = Column(String(10), nullable=True)  # ALLOW, MASK, or null
    policy_id = Column(Text, nullable=False)
    reasons = Column(JSONType, nullable=False, default=list)
    input_redactions = Column(JSONType, nullable=False, default=dict)
    output_redactions = Column(JSONType, nullable=False, default=dict)
    prompt_redacted = Column(Text, nullable=False)
    response_redacted = Column(Text, nullable=True)
    prompt_hash = Column(String(64), nullable=False)  # SHA-256 hex
    detection_latency_ms = Column(Integer, nullable=False)
    llm_latency_ms = Column(Integer, nullable=True)
    total_latency_ms = Column(Integer, nullable=False)


# --- Async Engine & Session Factory ---

# SQLite does not support the QueuePool sizing args that Postgres uses, so only
# pass them for non-SQLite backends.
_engine_kwargs: dict = {"echo": settings.debug}
if not settings.database_url.startswith("sqlite"):
    _engine_kwargs.update(pool_size=5, max_overflow=10)

engine = create_async_engine(settings.database_url, **_engine_kwargs)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncSession:
    """FastAPI dependency: yield an async database session."""
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the database engine."""
    await engine.dispose()
