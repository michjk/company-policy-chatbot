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

# Start all services (Ollama must be running on the host; pull nomic-embed-text first)
ollama pull nomic-embed-text
docker compose up --build
```

Makefile aliases: `make install`, `make test`, `make lint`, `make format`, `make typecheck`, `make check`, `make dev`, `make up`, `make down`, `make logs`.

## Architecture

The app is a FastAPI RAG chatbot. The **startup sequence** wires everything together via the FastAPI `lifespan` in `app/api.py`:

1. `app/observability.py` — enables LangSmith tracing and/or OTel if env vars are set (both are no-ops when unconfigured)
2. `app/checkpointer.py` — opens an `AsyncConnectionPool` (psycopg) and creates an `AsyncPostgresSaver` for LangGraph session persistence
3. `app/vectorstore.py` — creates a `PGVector` store in async mode (`postgresql+psycopg://` URL) and initialises the vector extension + tables
4. `app/rag_graph.py` — compiles a three-node LangGraph with the checkpointer attached; conversation history is stored per `thread_id` (= `session_id`)
5. `app/streaming.py` — wraps the compiled graph in a `LangGraphAgent` and registers a POST `/chat/stream` endpoint that speaks the AG-UI SSE protocol
6. `app/deps.py` — FastAPI dependency injection (`GraphDep`, `VectorstoreDep`) reading from `resources` (a simple namespace set during lifespan)

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

### LLM providers

`app/llm.py` `build_chat_model()` switches on `LLM_PROVIDER`:
- `openrouter` / `lmstudio` — `ChatOpenAI` with a custom `base_url` (both are OpenAI-compatible)
- `ollama` — `ChatOllama`

Embeddings always use Ollama (`nomic-embed-text`, 768-dim) regardless of the chat provider. Ollama is **not** included in `docker-compose.yml` — the user must run it on the host and set `OLLAMA_BASE_URL` accordingly (`http://localhost:11434` for local dev, `http://host.docker.internal:11434` inside Docker).

### Document ingestion

`app/ingestion.py` accepts `(filename, bytes)` pairs. Chunk IDs are `sha256(filename:content):chunk_index`, making re-uploads idempotent. Markdown files are split header-aware first; plain text uses `RecursiveCharacterTextSplitter` directly.

### Retrieval

`settings.retrieval_search_type` controls the pgvector search strategy: `similarity` (default) or `mmr` (max-marginal relevance; `fetch_k` is auto-set to `retrieval_k * 3`).

### API endpoints

- `GET /health` — liveness
- `POST /documents/ingest` — multipart upload of `.txt`/`.md` files
- `GET /documents` — list distinct ingested documents (by `doc_id`)
- `DELETE /documents/{doc_id}` — delete all chunks for a document
- `POST /chat` — synchronous JSON (`answer` + `citations`)
- `POST /chat/stream` — AG-UI SSE stream (managed by `ag-ui-langgraph`); clients send `RunAgentInput` with `thread_id`
- `GET /sessions/{session_id}/history` — fetch full message history from the LangGraph checkpointer

### Testing pattern

Tests use `httpx.AsyncClient` + `ASGITransport`. **ASGI lifespan events are not triggered by this transport**, so test apps set `app.state.*` directly instead of relying on `lifespan`. Shared fixtures (`sample_docs`, `mock_vectorstore`, `FakeEmbeddings`) live in `tests/conftest.py`. LangGraph tests use `MemorySaver` as the checkpointer.

## Key configuration

All settings are in `app/config.py` (`pydantic-settings`), read from `.env`. See `.env.example` for all variables. `settings.sqlalchemy_url` converts the plain `postgresql://` URL to `postgresql+psycopg://` for SQLAlchemy's async engine.

Key settings beyond the obvious: `retrieval_search_type` (`similarity`|`mmr`), `cors_origins` (JSON list, defaults to `["*"]`), `embedding_dim` (must match the Ollama model; `nomic-embed-text` = 768).
