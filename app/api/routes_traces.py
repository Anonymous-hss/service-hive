"""
GET /v1/traces/{trace_id} — return the structured trace for one pipeline turn.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import AgentTrace
from app.api.errors import TraceNotFoundError

router = APIRouter(tags=["traces"])


class ToolCallRecord(BaseModel):
    tool_name: str
    args: dict
    result: list | dict | str | None


class TraceResponse(BaseModel):
    trace_id: str
    session_id: str
    routed_to: str
    tool_calls: list[ToolCallRecord]
    retrieved_chunk_ids: list[str]
    latency_ms: int


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
) -> TraceResponse:
    """Return trace for one turn. 404 if not found."""
    result = await db.execute(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
    trace_row = result.scalars().first()

    if not trace_row:
        raise TraceNotFoundError(f"Trace {trace_id} not found")

    return TraceResponse(
        trace_id=trace_row.trace_id,
        session_id=trace_row.session_id,
        routed_to=trace_row.routed_to,
        tool_calls=trace_row.tool_calls,
        retrieved_chunk_ids=trace_row.retrieved_chunk_ids,
        latency_ms=trace_row.latency_ms
    )
