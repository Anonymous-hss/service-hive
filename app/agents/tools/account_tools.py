"""
Account tools — used by AccountAgent.

These tools query the DB for user-specific data.
Mock data is acceptable for the take-home; the integration matters.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: datetime
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


async def get_recent_builds(user_id: str, limit: int = 5) -> list[BuildSummary]:
    """
    Return the most recent builds for a user, newest first.
    """
    return [
        BuildSummary(
            build_id=f"build_{i}",
            pipeline="helix-ci",
            status="passed" if i % 2 == 0 else "failed",
            branch="main",
            started_at=datetime.utcnow(),
            duration_seconds=120 * (i + 1),
        )
        for i in range(limit)
    ]


from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models import User as DBUser

async def get_account_status(user_id: str) -> AccountStatus:
    """
    Return current account status (plan, usage limits).
    """
    plan_tier = "free"
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(DBUser).where(DBUser.user_id == user_id))
        user_row = result.scalars().first()
        if user_row:
            plan_tier = user_row.plan_tier

    return AccountStatus(
        user_id=user_id,
        plan_tier=plan_tier,
        concurrent_builds_used=3,
        concurrent_builds_limit=10 if plan_tier == "free" else (20 if plan_tier == "pro" else 100),
        storage_used_gb=12.5,
        storage_limit_gb=100.0,
    )

