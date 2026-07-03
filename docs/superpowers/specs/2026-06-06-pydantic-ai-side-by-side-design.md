# PydanticAI Side-by-Side Design

**Date:** 2026-06-06  
**Status:** Draft

---

## Context

The current chatbot uses LangGraph + LangChain for its agent/chat layer. While functional, this stack introduces significant abstraction weight: a 3-node StateGraph, a LangGraph checkpointer backed by a complex internal PostgreSQL schema, LangChain tool decorators, and `ag-ui-langgraph` as a streaming bridge.

The goal is to add a PydanticAI backend alongside the existing one, switchable via a config flag (`AGENT_BACKEND`). This lets us evaluate PydanticAI for simplicity and type safety without breaking the existing LangGraph implementation. The ingestion pipeline (chunking, langchain-postgres, PGVector) is unaffected — PydanticAI has no equivalent, and those components work well.

---

## Architecture

```
FastAPI lifespan
 ├── shared: vectorstore, retriever, embeddings, observability  (unchanged)
 └── AGENT_BACKEND=pydantic_ai  ──► new PydanticAI components
     AGENT_BACKEND=langgraph     ──► existing LangGraph components (default, unchanged)
```

Both backends expose the same HTTP surface:
- `POST /chat` — synchronous JSON response
- `POST /chat/stream` — AG-UI SSE stream
- `GET /sessions/{session_id}/history` — conversation history (see note below)

All other endpoints (`/health`, `/documents/*`) are unchanged.

---

## New Files

### `app/pydantic_ai_llm.py`
PydanticAI model factory, parallel to `app/llm.py`. Switches on `settings.llm_provider`:
- `openrouter` / `lmstudio` → `OpenAIModel` with custom `base_url`
- `ollama` → `OllamaModel`

### `app/pydantic_ai_agent.py`
Core agent definition. Replaces `app/rag_graph.py`.

```python
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

class AnswerResult(BaseModel):
    answer: str

@dataclass
class Citation:
    source: str
    chunk: int
    doc_id: str

@dataclass
class AgentDeps:
    retriever: Any
    citations: list[Citation] = field(default_factory=list)

agent: Agent[AgentDeps, AnswerResult] = Agent(
    build_pydantic_model(),
    deps_type=AgentDeps,
    result_type=AnswerResult,
    system_prompt=SYSTEM_PROMPT,
)

@agent.tool
async def retrieve_context(ctx: RunContext[AgentDeps], query: str) -> str:
    docs = await ctx.deps.retriever.ainvoke(query)
    ctx.deps.citations.extend(
        Citation(source=d.metadata["source"], chunk=d.metadata["chunk_index"],
                 doc_id=d.metadata["doc_id"]) for d in docs
    )
    return format_docs(docs)
```

Citations accumulate in `deps.citations` during tool calls — no "finalize" node needed.

### `app/pydantic_ai_sessions.py`
Lightweight PostgreSQL session store for the sync `/chat` endpoint. Replaces `app/checkpointer.py`.

Schema (one table):
```sql
CREATE TABLE IF NOT EXISTS pydantic_sessions (
    session_id TEXT PRIMARY KEY,
    messages   JSONB NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

API:
- `lifespan_session_store()` — async context manager yielding `PydanticSessionStore`
- `PydanticSessionStore.load(session_id)` → `list[ModelMessage]`
- `PydanticSessionStore.save(session_id, messages)`

Uses `pydantic_ai.messages.ModelMessagesTypeAdapter` for JSON serialization.

---

## Modified Files

### `app/config.py`
Add one field:
```python
agent_backend: Literal["langgraph", "pydantic_ai"] = "langgraph"
```

### `app/api.py`
The lifespan and chat endpoint routing become conditional on `settings.agent_backend`.

**Lifespan:**
```python
if settings.agent_backend == "pydantic_ai":
    async with lifespan_session_store() as session_store:
        pydantic_agent = build_pydantic_agent(retriever)
        resources.pydantic_agent = pydantic_agent
        resources.session_store = session_store
        register_pydantic_streaming(app, pydantic_agent, retriever)
        yield
else:
    async with lifespan_checkpointer() as checkpointer:
        graph = build_rag_graph(retriever, checkpointer)
        register_streaming_endpoint(app, graph)
        resources.graph = graph
        yield
```

**`POST /chat` routing:**
```python
if settings.agent_backend == "pydantic_ai":
    deps = AgentDeps(retriever=resources.retriever)
    history = await resources.session_store.load(req.session_id)
    result = await resources.pydantic_agent.run(
        req.message, deps=deps, message_history=history
    )
    await resources.session_store.save(req.session_id, result.all_messages())
    return ChatResponse(answer=result.data.answer, citations=deps.citations, ...)
else:
    # existing LangGraph path
```

**`POST /chat/stream`** — registered per-backend:
- LangGraph: `register_streaming_endpoint(app, graph)` as today
- PydanticAI: `AGUIAdapter.dispatch_request(request, agent=agent, deps=...)` via `register_pydantic_streaming()`

**`GET /sessions/{session_id}/history`:**
- LangGraph: existing `graph.aget_state()` path
- PydanticAI: read from `pydantic_sessions` table (only populated by sync `/chat` calls; streaming history is client-managed via AG-UI protocol)

### `pyproject.toml`
Add: `pydantic-ai[ag-ui]`

---

## Unchanged Files

`app/ingestion.py`, `app/vectorstore.py`, `app/embeddings.py`, `app/observability.py`,  
`app/rag_graph.py`, `app/checkpointer.py`, `app/streaming.py`, `app/llm.py`

---

## AG-UI Streaming (PydanticAI backend)

PydanticAI has native AG-UI support via `pydantic_ai.ui.ag_ui.AGUIAdapter`. This replaces `ag-ui-langgraph` entirely:

```python
from pydantic_ai.ui.ag_ui import AGUIAdapter

# In register_pydantic_streaming():
@app.post('/chat/stream')
async def stream(request: Request) -> Response:
    deps = AgentDeps(retriever=retriever)
    return await AGUIAdapter.dispatch_request(request, agent=agent, deps=deps)
```

`RunAgentInput.messages` from the AG-UI protocol carries the full client-side conversation history, so the streaming endpoint needs no server-side session storage.

Install: `pydantic-ai[ag-ui]` (adds the `ag_ui` extra).

---

## Session Persistence: Comparison

| Aspect | LangGraph | PydanticAI |
|---|---|---|
| Storage | LangGraph internal schema (multiple tables) | Single `pydantic_sessions` table with JSONB |
| Format | Internal checkpoint format | `list[ModelMessage]` — readable JSON |
| Streaming sessions | Stored server-side (checkpointer) | Client-managed via AG-UI protocol |
| Sync `/chat` sessions | Stored via checkpointer | Stored in `pydantic_sessions` table |
| Session history endpoint | `graph.aget_state()` | `SELECT messages FROM pydantic_sessions` |

---

## Dependencies: Before / After

| Package | LangGraph backend | PydanticAI backend |
|---|---|---|
| `langgraph` | Required | Not needed |
| `langgraph-checkpoint-postgres` | Required | Not needed |
| `ag-ui-langgraph` | Required | Not needed |
| `pydantic-ai[ag-ui]` | Not needed | Required |
| `langchain-postgres` | Required (ingestion) | Required (ingestion, unchanged) |
| `langchain-text-splitters` | Required (ingestion) | Required (ingestion, unchanged) |
| `ag-ui-protocol` | Required | Required |

---

## Testing Strategy

### Unit tests for PydanticAI agent (`tests/test_pydantic_ai_agent.py`)
Use `pydantic_ai.models.test.TestModel` (or `FunctionModel`) to avoid hitting real LLMs:
- Agent returns `AnswerResult` with typed `answer: str`
- `retrieve_context` tool accumulates citations in `deps.citations`
- Multi-turn: pass `message_history` and verify history is preserved

### Integration tests for session store (`tests/test_pydantic_ai_sessions.py`)
- Follow existing pattern: mock the DB pool, test load/save round-trip
- Verify `ModelMessagesTypeAdapter` serialization is idempotent

### API tests with PydanticAI backend
- Extend existing `tests/test_api.py` with `settings.agent_backend = "pydantic_ai"`
- Override deps to use `TestModel` and mock session store
- Test `/chat`, `/chat/stream` (basic smoke), `/sessions/{id}/history`

---

## Verification

1. `AGENT_BACKEND=langgraph uv run uvicorn app.api:app --reload` — existing behaviour unchanged
2. `AGENT_BACKEND=pydantic_ai uv run uvicorn app.api:app --reload` — PydanticAI backend starts
3. `POST /chat` with a question → returns `AnswerResult`-structured JSON with citations
4. `/chat/stream` → AG-UI SSE stream works with the existing frontend
5. Second `/chat` request with same `session_id` → history preserved
6. `uv run pytest` — all existing tests pass; new PydanticAI tests pass
7. `uv run pyrefly check` — no type errors on new files
