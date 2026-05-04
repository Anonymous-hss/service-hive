"""
Test fixtures.
"""
import pytest
import pytest_asyncio
import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base, Message as DBMessage, AgentTrace, Session as DBSession
from app.db.session import get_db
from app.main import app
from app.srop.pipeline import PipelineResult


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.
    """
    async def mock_run(session_id, message, db):
        from sqlalchemy import select
        if "rotate" in message.lower():
            trace_id = "test-trace-001"
            trace = AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to="knowledge",
                tool_calls=[],
                retrieved_chunk_ids=["chunk_test_123"],
                latency_ms=10,
                created_at=datetime.utcnow()
            )
            db.add(trace)
            
            msg = DBMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content="To rotate a deploy key, use the CLI command.",
                trace_id=trace_id,
                created_at=datetime.utcnow()
            )
            db.add(msg)
            
            res = await db.execute(select(DBSession).where(DBSession.session_id == session_id))
            session_row = res.scalars().first()
            if session_row:
                state = session_row.state
                state["turn_count"] = state.get("turn_count", 0) + 1
                state["last_agent"] = "knowledge"
                session_row.state = state
            
            await db.commit()
            
            return PipelineResult(
                content="To rotate a deploy key, use the CLI command.",
                routed_to="knowledge",
                trace_id=trace_id,
            )
        else:
            trace_id = "test-trace-002"
            trace = AgentTrace(
                trace_id=trace_id,
                session_id=session_id,
                routed_to="account",
                tool_calls=[],
                retrieved_chunk_ids=[],
                latency_ms=10,
                created_at=datetime.utcnow()
            )
            db.add(trace)
            
            msg = DBMessage(
                message_id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content="Your plan tier is pro.",
                trace_id=trace_id,
                created_at=datetime.utcnow()
            )
            db.add(msg)
            
            res = await db.execute(select(DBSession).where(DBSession.session_id == session_id))
            session_row = res.scalars().first()
            if session_row:
                state = session_row.state
                state["turn_count"] = state.get("turn_count", 0) + 1
                state["last_agent"] = "account"
                session_row.state = state

            await db.commit()

            return PipelineResult(
                content="Your plan tier is pro.",
                routed_to="account",
                trace_id=trace_id,
            )

    monkeypatch.setattr("app.srop.pipeline.run", mock_run)
