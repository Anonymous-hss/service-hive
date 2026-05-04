"""
SROP entrypoint — called by the message route.
"""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from google.genai import types

import structlog

from app.agents.orchestrator import root_agent
from app.agents.tools.search_docs import search_docs
from app.agents.tools.account_tools import get_recent_builds, get_account_status
from app.api.errors import SessionNotFoundError, UpstreamTimeoutError
from app.db.models import Session as DBSession, Message as DBMessage, AgentTrace
from app.settings import settings
from app.srop.state import SessionState
from google.adk.runners import InMemoryRunner

log = structlog.get_logger()


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    # 1. Load session from DB
    result = await db.execute(select(DBSession).where(DBSession.session_id == session_id))
    session_row = result.scalars().first()
    if not session_row:
        raise SessionNotFoundError(f"Session {session_id} not found")

    state = SessionState.from_db_dict(session_row.state)

    # Save user message
    user_msg_row = DBMessage(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        created_at=datetime.utcnow()
    )
    db.add(user_msg_row)

    # 2. Extract prior message history
    history_text = ""
    msg_result = await db.execute(
        select(DBMessage).where(DBMessage.session_id == session_id).order_by(DBMessage.created_at)
    )
    past_messages = msg_result.scalars().all()
    if past_messages:
        history_text = "Previous conversation history:\n"
        for m in past_messages:
            role_label = "User" if m.role == "user" else "Assistant"
            history_text += f"{role_label}: {m.content}\n"

    user_turn_content = f"{history_text}\nNew user message: {user_message}"

    # Set keys in env for ADK
    os.environ["GEMINI_API_KEY"] = settings.google_api_key
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    start_time = time.time()
    routed_to = "smalltalk"
    final_content = ""
    retrieved_chunk_ids = []
    tool_calls = []

    try:
        runner = InMemoryRunner(agent=root_agent, app_name="helix_srop")
        await runner.session_service.create_session(
            app_name="helix_srop", user_id=state.user_id, session_id=session_id
        )

        msg = types.Content(
            role="user",
            parts=[types.Part(text=user_turn_content)]
        )

        # Wrapper coroutine to run async iterator to completion
        async def run_adk_runner():
            events = []
            async for event in runner.run_async(
                user_id=state.user_id,
                session_id=session_id,
                new_message=msg,
            ):
                events.append(event)
            return events

        # Run async with a timeout limit
        events = await asyncio.wait_for(
            run_adk_runner(),
            timeout=settings.llm_timeout_seconds
        )

        for event in events:
            if event.is_final_response():
                if event.author in ["knowledge_agent", "account_agent"]:
                    routed_to = event.author.replace("_agent", "")
                final_content = event.content.parts[0].text if event.content and event.content.parts else ""

        # If LLM didn't return any content, fall back to rule-based routing
        if not final_content:
            raise ValueError("Empty final content from LLM.")

    except asyncio.TimeoutError:
        raise UpstreamTimeoutError(f"LLM did not respond within {settings.llm_timeout_seconds}s")
    except Exception as exc:
        # Log the ADK error — graceful fallback to rule-based routing
        log.warning(
            "adk_agent_failed_fallback_routing",
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        query_lower = user_message.lower()
        if any(k in query_lower for k in ["build", "status", "usage", "account", "plan"]):
            routed_to = "account"
            if any(k in query_lower for k in ["build"]):
                builds = await get_recent_builds(state.user_id)
                tool_calls.append({
                    "tool_name": "get_recent_builds",
                    "args": {"user_id": state.user_id},
                    "result": str(builds)
                })
                final_content = f"Here are your most recent builds:\n" + "\n".join(
                    [f"- Build {b.build_id} for {b.branch} branch: {b.status}" for b in builds]
                )
            else:
                acc_status = await get_account_status(state.user_id)
                tool_calls.append({
                    "tool_name": "get_account_status",
                    "args": {"user_id": state.user_id},
                    "result": str(acc_status)
                })
                final_content = (
                    f"Your account is on the {acc_status.plan_tier} tier. "
                    f"You have used {acc_status.concurrent_builds_used}/{acc_status.concurrent_builds_limit} concurrent builds and "
                    f"{acc_status.storage_used_gb}/{acc_status.storage_limit_gb} GB storage."
                )
        elif any(k in query_lower for k in ["how", "what", "where", "search", "docs"]):
            routed_to = "knowledge"
            chunks = await search_docs(user_message, k=3)
            if chunks:
                retrieved_chunk_ids = [c.chunk_id for c in chunks]
                tool_calls.append({
                    "tool_name": "search_docs",
                    "args": {"query": user_message, "k": 3},
                    "result": [c.chunk_id for c in chunks]
                })
                top = chunks[0]
                final_content = f"According to [{top.chunk_id}]: {top.content}"
            else:
                final_content = "I could not find any matching product documentation for your request."
        else:
            routed_to = "smalltalk"
            final_content = "Hello! I am the Helix Support Concierge. How can I help you today with your builds or product questions?"

    latency_ms = int((time.time() - start_time) * 1000)

    # 3. Record trace
    trace_id = str(uuid.uuid4())
    trace_row = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=retrieved_chunk_ids,
        latency_ms=latency_ms,
        created_at=datetime.utcnow()
    )
    db.add(trace_row)

    # 4. Record assistant message
    assistant_msg_row = DBMessage(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=final_content,
        trace_id=trace_id,
        created_at=datetime.utcnow()
    )
    db.add(assistant_msg_row)

    # 5. Persist updated session state to DB
    state.turn_count += 1
    state.last_agent = routed_to
    session_row.state = state.to_db_dict()

    await db.commit()

    return PipelineResult(content=final_content, routed_to=routed_to, trace_id=trace_id)
