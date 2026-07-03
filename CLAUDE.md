# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (including dev)
uv sync

# Run all tests
uv run pytest

# Run a single test file or test
uv run pytest tests/test_ingestion.py
uv run pytest tests/test_api.py::test_chat_json

# Lint + format (auto-fix)
uv run ruff check --fix && uv run ruff format

# Type check
uv run pyrefly check

# Run pre-commit on all files
uv run pre-commit run --all-files

# Start the dev server (requires docker-compose services running)
uv run uvicorn app.api:app --reload

# Start with PydanticAI backend
AGENT_BACKEND=pydantic_ai uv run uvicorn app.api:app --reload

# Start all services (Ollama must be running on the host; pull nomic-embed-text first)
ollama pull nomic-embed-text
docker compose up --build
```

Makefile aliases: `make install`, `make test`, `make lint`, `make format`, `make typecheck`, `make check`, `make dev`, `make up`, `make down`, `make logs`.

## Architecture

The app is a FastAPI RAG chatbot with two interchangeable agent backends, selected via `AGENT_BACKEND` (default: `langgraph`). Both backends share the ingestion pipeline and vectorstore; only the chat/session layer differs.

### Startup sequence (`app/api.py` lifespan)

**Shared (both backends):**
1. `app/observability.py` — enables LangSmith tracing and/or OTel if env vars are set (both are no-ops when unconfigured)
2. `app/vectorstore.py` — creates a `PGVector` store in async mode and initialises the vector extension + tables
3. Retriever is built from the vectorstore and stored on `resources.retriever`

**LangGraph backend (`AGENT_BACKEND=langgraph`, default):**
4. `app/checkpointer.py` — opens an `AsyncConnectionPool` (psycopg) and creates an `AsyncPostgresSaver` for LangGraph session persistence
5. `app/rag_graph.py` — compiles a three-node LangGraph with the checkpointer attached; conversation history is stored per `thread_id` (= `session_id`)
6. `app/streaming.py` — wraps the compiled graph in a `LangGraphAgent` and registers `POST /chat/stream`

**PydanticAI backend (`AGENT_BACKEND=pydantic_ai`):**
4. `app/pydantic_ai_sessions.py` — opens an `AsyncConnectionPool` and creates the `pydantic_sessions` table (single JSONB column per `session_id`)
5. `app/pydantic_ai_agent.py` — builds `Agent[AgentDeps, str]` with a `retrieve_context` tool
6. `app/pydantic_ai_streaming.py` — registers `POST /chat/stream` via `pydantic_ai.ui.ag_ui.AGUIAdapter`

### LangGraph graph

Three nodes with a tool-calling loop:

```
START → agent ──(has tool calls)──→ tools → agent (loop)
              └──(no tool calls)──→ finalize → END
```

- **`agent`** — LLM with `retrieve_context` tool bound; decides whether to retrieve or answer
- **`tools`** — executes `retrieve_context`, accumulates `Document` objects into `state["context"]`
- **`finalize`** — attaches citations from `state["context"]` to `AIMessage.response_metadata["citations"]`

### LangGraph state

`RAGState` holds `messages` (full conversation, accumulated via `add_messages`) and `context` (retrieved docs, replaced each turn). Citations are stored as `AIMessage.response_metadata["citations"]`.

### PydanticAI agent

`Agent[AgentDeps, str]` defined in `app/pydantic_ai_agent.py`:
- `AgentDeps` dataclass carries the retriever and a mutable `citations: list[Citation]` list (fresh per request)
- `@agent.tool retrieve_context` calls `retriever.ainvoke()` and accumulates citations in `deps.citations`
- `output_type=str` keeps AG-UI streaming as plain text tokens (not JSON)
- Citations are accurate (from actual retrieved docs) rather than LLM-generated

Session persistence uses `PydanticSessionStore` (one `pydantic_sessions` table). Load/save uses `ModelMessagesTypeAdapter` for serialisation. The streaming endpoint (`AGUIAdapter`) is stateless server-side — the AG-UI client sends full message history on each request. **Note:** citations are not emitted in the stream; use `POST /chat` (synchronous) if citation metadata is required.

### LLM providers

Both `app/llm.py` (LangGraph) and `app/pydantic_ai_llm.py` (PydanticAI) switch on `LLM_PROVIDER`:
- `openrouter` / `lmstudio` — OpenAI-compatible API with custom `base_url`
- `ollama` — native Ollama client (`ChatOllama` / `OllamaModel`)

Embeddings always use Ollama (`nomic-embed-text`, 768-dim) regardless of the chat provider. Ollama is **not** included in `docker-compose.yml` — the user must run it on the host and set `OLLAMA_BASE_URL` accordingly.

### Shared prompt

`app/prompts.py` exports `RAG_SYSTEM_PROMPT`, used by both `app/rag_graph.py` and `app/pydantic_ai_agent.py`.

### Document ingestion

`app/ingestion.py` accepts `(filename, bytes)` pairs. Chunk IDs are `sha256(filename:content):chunk_index`, making re-uploads idempotent. Markdown files are split header-aware first; plain text uses `RecursiveCharacterTextSplitter` directly.

### Retrieval

`settings.retrieval_search_type` controls the pgvector search strategy: `similarity` (default) or `mmr` (max-marginal relevance; `fetch_k` is auto-set to `retrieval_k * 3`).

### API endpoints

- `GET /health` — liveness
- `POST /documents/ingest` — multipart upload of `.txt`/`.md` files
- `GET /documents` — list distinct ingested documents (by `doc_id`)
- `DELETE /documents/{doc_id}` — delete all chunks for a document
- `POST /chat` — synchronous JSON (`answer` + `citations`); supports both backends
- `POST /chat/stream` — AG-UI SSE stream; LangGraph backend uses `ag-ui-langgraph`, PydanticAI uses `AGUIAdapter`
- `GET /sessions/{session_id}/history` — message history (LangGraph: from checkpointer; PydanticAI: from `pydantic_sessions` table, only populated by sync `/chat` calls)

### Testing pattern

Tests use `httpx.AsyncClient` + `ASGITransport`. **ASGI lifespan events are not triggered by this transport**, so tests set `resources.*` and dependency overrides directly. Shared fixtures (`sample_docs`, `mock_vectorstore`, `FakeEmbeddings`) live in `tests/conftest.py`. LangGraph tests use `MemorySaver` as the checkpointer. PydanticAI agent tests use `TestModel` via `agent.override(model=TestModel())`.

## Key configuration

All settings are in `app/config.py` (`pydantic-settings`), read from `.env`. See `.env.example` for all variables.

Key settings:
- `AGENT_BACKEND` — `langgraph` (default) or `pydantic_ai`
- `retrieval_search_type` — `similarity` (default) or `mmr`
- `cors_origins` — JSON list, defaults to `["*"]`
- `embedding_dim` — must match the Ollama model; `nomic-embed-text` = 768
- `settings.sqlalchemy_url` — converts `postgresql://` to `postgresql+psycopg://` for SQLAlchemy's async engine
