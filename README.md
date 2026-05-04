# Helix SROP — AI Support Concierge

A stateful **RAG Orchestration Pipeline** (SROP) built with FastAPI, Google ADK, ChromaDB, and async SQLAlchemy. The agent handles two workflows in a single multi-turn conversation:

1. **Knowledge questions** — answered from product documentation via RAG with citations
2. **Account lookups** — queries internal tools for builds, status, and usage data

State persists across process restarts via SQLite-backed session management.

---

## Setup (< 5 minutes)

```bash
# 1. Clone
git clone <your-repo-url>
cd helix-srop

# 2. Install dependencies (requires uv — https://docs.astral.sh/uv/)
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Ingest documentation into the vector store
uv run python -m app.rag.ingest --path docs/

# 5. Start the server
uv run uvicorn app.main:app --reload
```

## Quick Test

```bash
# Create a session
SESSION=$(curl -s -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_demo", "plan_tier": "pro"}' | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

echo "Session: $SESSION"

# Turn 1 — Knowledge question (routes to KnowledgeAgent)
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "How do I rotate a deploy key?"}' | python -m json.tool

# Turn 2 — Account question (routes to AccountAgent, remembers context)
curl -s -X POST http://localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content": "Show me my last 3 failed builds"}' | python -m json.tool

# View a trace
curl -s http://localhost:8000/v1/traces/<trace_id_from_above> | python -m json.tool
```

## Run Tests

```bash
uv run pytest -q
```

---

## Architecture

```
POST /v1/chat/{session_id}
         │
         ▼
┌─────────────────────────┐
│  SROP Pipeline          │
│  1. Load session state  │  ← SQLite (sessions.state JSON column)
│  2. Build conversation  │  ← messages table for history
│  3. Run ADK orchestrator│  ← google-adk InMemoryRunner
│  4. Save updated state  │  ← turn_count, last_agent, plan_tier
│  5. Write trace to DB   │  ← agent_traces table
└────────────┬────────────┘
             │ routes via ADK AgentTool
       ┌─────┴──────┐
       ▼            ▼
 KnowledgeAgent  AccountAgent
 (RAG + search)  (DB tools)
       │            │
  ChromaDB       Mock data
  (doc chunks)  (builds, status)
```

### Module Structure

```
app/
├── main.py              # FastAPI app, lifespan, error handlers
├── settings.py          # Pydantic Settings (reads .env)
├── agents/
│   ├── orchestrator.py  # Root agent + sub-agents (ADK AgentTool pattern)
│   └── tools/
│       ├── search_docs.py    # Vector store search (ChromaDB + Gemini embeddings)
│       └── account_tools.py  # Mock account/build data
├── api/
│   ├── routes_sessions.py  # POST /v1/sessions
│   ├── routes_chat.py      # POST /v1/chat/{session_id}
│   ├── routes_traces.py    # GET /v1/traces/{trace_id}
│   └── errors.py           # HelixError hierarchy + RFC 7807 handler
├── db/
│   ├── models.py   # SQLAlchemy 2.x models (users, sessions, messages, agent_traces)
│   └── session.py  # Async engine + session factory
├── obs/
│   └── logging.py  # structlog configuration
├── rag/
│   └── ingest.py   # CLI: chunk + embed + upsert to ChromaDB
└── srop/
    ├── pipeline.py  # Core pipeline: state → ADK → trace → persist
    └── state.py     # SessionState Pydantic model
```

---

## Design Decisions

### State persistence pattern

I used **Pattern 2: DB-backed state reload** from the ADK guide. Session state (`user_id`, `plan_tier`, `turn_count`, `last_agent`) is serialized as JSON in the `sessions.state` column and loaded at the start of every turn. This means:

- State survives process restarts (stored in SQLite, not memory)
- The state schema is explicit and typed (Pydantic `SessionState` model)
- Conversation history is also persisted in the `messages` table and injected into the ADK context on each turn

**Why not ADK's built-in session service?** ADK's `InMemoryRunner` stores sessions in-memory, which doesn't survive restarts. By managing state in our own DB, we get durable persistence with a simple `create_all` setup.

### Chunking strategy

I used **heading-aware + sentence-aware** chunking:

1. Split on `##` and `###` markdown headings to preserve section boundaries
2. If a section exceeds `chunk_size` (default 512 chars), sub-chunk by sentence boundaries with 1-sentence overlap

This preserves semantic coherence — a chunk about "deploy keys" won't bleed into "billing plans". Sentence-level sub-chunking avoids cutting mid-thought.

### Vector store choice

I chose **ChromaDB** (with `PersistentClient`) because:

- Zero-config: no external process needed, stores on disk
- Cosine similarity built-in
- `upsert` with stable IDs means re-ingestion is idempotent
- Recommended in the assignment's `rag-guide.md`

### Stable chunk IDs

Chunk IDs are deterministic: `sha256(filename::chunk_index)[:16]` prefixed with `chunk_`. Re-running ingest on the same docs produces the same IDs, preventing duplicates via ChromaDB's `upsert`.

---

## Known Limitations

- **ADK InMemoryRunner**: Each request creates a new `InMemoryRunner` instance. ADK's session state is not reused across turns — instead, we inject conversation history from our DB. This works but adds latency.
- **Trace data from ADK path**: When the ADK orchestrator succeeds, individual tool call details within sub-agents are not fully captured in traces (ADK's event model doesn't expose intermediate tool calls from sub-agents). The fallback path captures full tool call data.
- **Mock account data**: `get_recent_builds` and `get_account_status` return hardcoded mock data. In production, these would query a real database.
- **No streaming**: Responses are returned in a single JSON response, not streamed.

## What I'd Do With More Time

- **E2: Escalation agent** — add a third sub-agent for ticket creation with a `tickets` DB table
- **E6: Docker** — `Dockerfile` + `docker-compose.yml` for one-command deployment
- **E5: Guardrails** — refusal on out-of-scope queries + PII redaction in logs
- **Better trace capture** — hook into ADK's event stream to capture intermediate tool calls from sub-agents
- **Connection pooling** — reuse ChromaDB client across requests instead of creating per-request

## Time Spent

| Phase | Time |
|-------|------|
| Setup + DB + FastAPI boilerplate | 30 min |
| RAG ingest + search_docs | 40 min |
| ADK agents + AgentTool wiring | 45 min |
| pipeline.py + state persistence | 50 min |
| Tests | 25 min |
| README + cleanup | 30 min |
| **Total** | **~3.5 hours** |

## Extensions Completed

- [x] `/healthz` endpoint (bonus)
- [ ] E1: Idempotency
- [ ] E2: Escalation agent
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness
