"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from app.agents.tools.search_docs import search_docs
from app.agents.tools.account_tools import get_recent_builds, get_account_status
from app.settings import settings

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Intent → tool:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their account, builds, status, usage → account_agent
- Greetings or off-topic → respond directly, no tool call

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context will be in the system message — use it.
"""


async def search_knowledge(query: str) -> str:
    """Search product documentation to answer the user's question.

    Args:
        query: natural language query to find in the documentation.
    """
    chunks = await search_docs(query, k=5)
    if not chunks:
        return "No relevant documentation found."
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk.chunk_id}] (score: {chunk.score:.2f}, source: {chunk.metadata.get('source')})\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction="""You are a Helix product knowledge agent.
Answer questions using ONLY the provided context chunks.
Always cite the chunk_id (e.g. "According to [chunk_abc123]...").
If the context does not contain the answer, say so — do not guess.""",
    tools=[search_knowledge],
)


async def recent_builds(user_id: str, limit: int = 5) -> str:
    """Return the most recent builds for a user.

    Args:
        user_id: the user identifier.
        limit: the maximum number of recent builds to return.
    """
    builds = await get_recent_builds(user_id, limit)
    return str(builds)


async def account_status(user_id: str) -> str:
    """Return current account status (plan, usage limits).

    Args:
        user_id: the user identifier.
    """
    status = await get_account_status(user_id)
    return str(status)


account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction="""You are an account management agent.
Answer user's question about their account, status, and builds using your tools.""",
    tools=[recent_builds, account_status],
)


root_agent = LlmAgent(
    name="srop_root",
    model=settings.adk_model,
    instruction=ROOT_INSTRUCTION,
    tools=[
        AgentTool(agent=knowledge_agent),
        AgentTool(agent=account_agent),
    ],
)
