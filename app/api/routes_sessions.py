"""
POST /v1/sessions — create a session.
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import User as DBUser, Session as DBSession
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str
    plan_tier: str = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    # Check if user exists
    user_result = await db.execute(select(DBUser).where(DBUser.user_id == body.user_id))
    user_row = user_result.scalars().first()

    if not user_row:
        user_row = DBUser(
            user_id=body.user_id,
            plan_tier=body.plan_tier,
            created_at=datetime.utcnow()
        )
        db.add(user_row)
    else:
        user_row.plan_tier = body.plan_tier

    session_id = str(uuid.uuid4())
    state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)

    session_row = DBSession(
        session_id=session_id,
        user_id=body.user_id,
        state=state.to_db_dict(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(session_row)
    await db.commit()

    return CreateSessionResponse(session_id=session_id, user_id=body.user_id)
