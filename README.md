# Company Policy Chatbot

A RAG-powered chatbot for answering questions about company policies. Upload plain-text or Markdown documents, then ask questions via a REST API. Answers include source citations and conversation history is persisted across turns.

**Frontend:** [michjk/company-policy-chatbot-frontend](https://github.com/michjk/company-policy-chatbot-frontend)

## Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI (async) |
| RAG pipeline | LangChain + LangGraph (tool-calling agent with Postgres checkpointing) |
| Vector store | pgvector (Postgres 17) via `langchain-postgres` |
| Embeddings | Ollama `nomic-embed-text` (768-dim, always local) |
| LLM | OpenRouter (default); Ollama and LMStudio also supported |
| Streaming | [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) over SSE |
| Observability | LangSmith and/or OpenTelemetry (both env-toggled, no-op when unconfigured) |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [Ollama](https://ollama.com/) running on the host (used for embeddings regardless of chat LLM provider)
- An [OpenRouter](https://openrouter.ai/) API key (or switch `LLM_PROVIDER` to `ollama` / `lmstudio`)

## Quick start

```bash
cp .env.example .env
# Edit .env — at minimum set OPENROUTER_API_KEY

# Pull the embedding model (required before starting)
ollama pull nomic-embed-text

# Start Postgres + app
docker compose up --build
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/documents/ingest` | Upload `.txt` / `.md` files (multipart) |
| `GET` | `/documents` | List ingested documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document and all its chunks |
| `POST` | `/chat` | Multi-turn Q&A — synchronous JSON response |
| `POST` | `/chat/stream` | Multi-turn Q&A — AG-UI SSE stream |
| `GET` | `/sessions/{session_id}/history` | Fetch conversation history |

### Ingest documents

```bash
curl -F "files=@policies/handbook.md" \
     -F "files=@policies/code-of-conduct.txt" \
     http://localhost:8000/documents/ingest
```

Re-uploading the same file is idempotent — chunk IDs are derived from `sha256(filename:content)`.

### Chat (JSON)

```bash
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id": "s1", "message": "What is the PTO policy?"}'
```

Response:

```json
{
  "session_id": "s1",
  "answer": "Employees receive 15 days of PTO per year...",
  "citations": [
    {"source": "handbook.md", "chunk": 3, "doc_id": "abc123..."}
  ]
}
```

Follow-up calls with the same `session_id` retain conversation history.

### Chat (AG-UI stream)

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{
    "thread_id": "s1",
    "run_id": "r1",
    "messages": [{"role": "user", "content": "What is the PTO policy?"}],
    "tools": [],
    "context": [],
    "state": {},
    "forwarded_props": {}
  }'
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openrouter` | `openrouter` \| `ollama` \| `lmstudio` |
| `OPENROUTER_API_KEY` | — | Required when using OpenRouter |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-6` | Any model on OpenRouter |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama host (chat + embeddings); use `http://localhost:11434` outside Docker |
| `OLLAMA_CHAT_MODEL` | `llama3.1` | Chat model when `LLM_PROVIDER=ollama` |
| `LMSTUDIO_BASE_URL` | `http://host.docker.internal:1234/v1` | LMStudio base URL |
| `LMSTUDIO_MODEL` | `local-model` | Model name for LMStudio |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_DIM` | `768` | Embedding vector dimension |
| `POSTGRES_URL` | `postgresql://app:app@postgres:5432/policies` | Standard PostgreSQL URL |
| `PGVECTOR_COLLECTION` | `company_policies` | pgvector collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `RETRIEVAL_SEARCH_TYPE` | `similarity` | `similarity` \| `mmr` |
| `CORS_ORIGINS` | `["*"]` | JSON array of allowed origins |
| `LANGSMITH_API_KEY` | — | Set to enable LangSmith tracing |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Set to enable OpenTelemetry export |

## Local development

```bash
uv sync                                              # install all deps including dev
uv run uvicorn app.api:app --reload                  # dev server (requires Postgres running)
uv run pytest                                        # run all tests
uv run pytest tests/test_api.py::test_chat_json      # single test
uv run ruff check --fix && uv run ruff format        # lint + format
uv run pyrefly check                                 # type check
uv run pre-commit run --all-files                    # all pre-commit hooks
```

Or use the Makefile shortcuts:

```bash
make install   # uv sync
make dev       # start dev server
make test      # pytest
make lint      # ruff check --fix
make format    # ruff format
make typecheck # pyrefly check
make check     # pre-commit run --all-files
make up        # docker compose up --build
make down      # docker compose down
make logs      # docker compose logs -f app
```

Tests run without Docker — they use in-memory fakes for the LLM, embeddings, and checkpointer.

## Architecture

The graph follows a tool-calling agent pattern:

```
START → agent ──(has tool calls)──→ tools → agent (loop)
              └──(no tool calls)──→ finalize → END
```

1. **`agent`** — the LLM (with `retrieve_context` tool bound) decides whether to retrieve docs or answer directly.
2. **`tools`** — executes `retrieve_context`, accumulates retrieved `Document` objects into `state["context"]`.
3. **`finalize`** — attaches citations from `state["context"]` to `AIMessage.response_metadata["citations"]`.

`RAGState` carries `messages` (full conversation via `add_messages`) and `context` (retrieved docs, replaced each turn). Session history is persisted per `thread_id` in Postgres via `AsyncPostgresSaver`.

Markdown files are split header-aware; plain text uses `RecursiveCharacterTextSplitter`. Both produce chunks with `sha256`-based IDs, making re-ingestion idempotent.
