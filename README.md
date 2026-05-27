# Company Policy Chatbot

A RAG-powered chatbot for answering questions about company policies. Upload plain-text or Markdown documents, then ask questions via a REST API. Answers include source citations and the conversation is stateful across turns.

## Stack

- **API**: FastAPI (async)
- **RAG pipeline**: LangChain + LangGraph (`retrieve → generate` graph with Postgres-backed session checkpointing)
- **Vector store**: pgvector (Postgres 17) via `langchain-postgres`
- **Embeddings**: Ollama `nomic-embed-text` (768-dim, local)
- **LLM**: OpenRouter by default; Ollama and LMStudio also supported
- **Streaming**: [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) over SSE
- **Observability**: LangSmith and/or OpenTelemetry (both env-toggled)

## Quick start

```bash
cp .env.example .env
# Add OPENROUTER_API_KEY to .env (or switch LLM_PROVIDER to ollama/lmstudio)

# Embeddings always use Ollama — ensure it is running on the host and pull the model:
ollama pull nomic-embed-text

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
| `POST` | `/chat` | Single-turn or multi-turn Q&A (JSON) |
| `POST` | `/chat/stream` | Streaming Q&A (AG-UI SSE) |
| `GET` | `/sessions/{session_id}/history` | Fetch conversation history |

### Ingest documents

```bash
curl -F "files=@policies/handbook.md" \
     -F "files=@policies/code-of-conduct.txt" \
     http://localhost:8000/documents/ingest
```

Re-uploading the same file is idempotent (chunk IDs are SHA-256 based).

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
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4-6` | Any model available on OpenRouter |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama host (chat + embeddings); use `http://localhost:11434` outside Docker |
| `POSTGRES_URL` | `postgresql://app:app@postgres:5432/policies` | Standard PostgreSQL URL |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `LANGSMITH_API_KEY` | — | Set to enable LangSmith tracing |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | Set to enable OpenTelemetry export |

## Local development

```bash
uv sync                        # install all deps including dev
uv run pytest                  # run tests
uv run pytest tests/test_api.py::test_chat_json   # single test
uv run ruff check --fix && uv run ruff format      # lint + format
uv run pyrefly check           # type check
uv run pre-commit run --all-files                  # all checks
```

Tests run without Docker — they use in-memory fakes for the LLM, embeddings, and checkpointer.
