from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_postgres.vectorstores import _get_embedding_collection_store
from pydantic import BaseModel
from sqlalchemy import select

from .checkpointer import lifespan_checkpointer
from .config import settings
from .deps import GraphDep, VectorstoreDep, resources
from .ingestion import ingest_files
from .observability import setup_observability
from .rag_graph import build_rag_graph
from .streaming import register_streaming_endpoint
from .vectorstore import build_vectorstore, init_vectorstore

# PydanticAI backend imports (loaded conditionally at runtime to avoid hard dependency)
# These are imported inside the lifespan/helpers when agent_backend == "pydantic_ai"

# ── Response schemas ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str


class IngestFileReport(BaseModel):
    filename: str
    chunks: int
    doc_id: str | None


class IngestResponse(BaseModel):
    files: list[IngestFileReport]
    chunks_total: int


class DocumentEntry(BaseModel):
    filename: str
    doc_id: str
    uploaded_at: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentEntry]


class Citation(BaseModel):
    source: str
    chunk: int
    doc_id: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]


class MessageEntry(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageEntry]


class ChatRequest(BaseModel):
    session_id: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_observability(app)

    vs = build_vectorstore()
    await init_vectorstore(vs)
    search_kwargs: dict = {"k": settings.retrieval_k}
    if settings.retrieval_search_type == "mmr":
        search_kwargs["fetch_k"] = settings.retrieval_k * 3
    retriever = vs.as_retriever(
        search_type=settings.retrieval_search_type,
        search_kwargs=search_kwargs,
    )
    resources.vectorstore = vs
    resources.retriever = retriever

    if settings.agent_backend == "pydantic_ai":
        from .pydantic_ai_agent import build_pydantic_agent
        from .pydantic_ai_sessions import lifespan_session_store
        from .pydantic_ai_streaming import register_pydantic_streaming

        async with lifespan_session_store() as session_store:
            agent = build_pydantic_agent()
            resources.pydantic_agent = agent
            resources.session_store = session_store
            register_pydantic_streaming(app, agent, retriever)
            yield
    else:
        async with lifespan_checkpointer() as checkpointer:
            graph = build_rag_graph(retriever, checkpointer)
            resources.graph = graph
            register_streaming_endpoint(app, graph, path="/chat/stream")
            yield


app = FastAPI(
    title="Company Policy Chatbot",
    description="RAG chatbot for company policy Q&A. Streaming via AG-UI protocol.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/health", tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


# ── Documents ────────────────────────────────────────────────────────────────


@app.post("/documents/ingest", status_code=201, tags=["documents"])
async def ingest_documents(
    files: list[UploadFile], vs: VectorstoreDep
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")
    allowed = {".txt", ".md"}
    for f in files:
        if Path(f.filename or "").suffix.lower() not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type: {f.filename}. Only .txt and .md are accepted.",
            )
    raw_files = [(f.filename or "unknown", await f.read()) for f in files]
    reports = await ingest_files(vs, raw_files)
    total_chunks = sum(r["chunks"] for r in reports)
    return IngestResponse(
        files=[IngestFileReport(**r) for r in reports],
        chunks_total=total_chunks,
    )


@app.get("/documents", tags=["documents"])
async def list_documents(vs: VectorstoreDep) -> DocumentsResponse:
    EmbeddingStore, _ = _get_embedding_collection_store(vs._embedding_length)
    async with vs._make_async_session() as session:
        result = await session.execute(
            select(
                EmbeddingStore.cmetadata["filename"].as_string().label("filename"),
                EmbeddingStore.cmetadata["doc_id"].as_string().label("doc_id"),
                EmbeddingStore.cmetadata["uploaded_at"]
                .as_string()
                .label("uploaded_at"),
            ).distinct(EmbeddingStore.cmetadata["doc_id"])
        )
        rows = result.all()
    return DocumentsResponse(
        documents=[
            DocumentEntry(
                filename=r.filename, doc_id=r.doc_id, uploaded_at=r.uploaded_at
            )
            for r in rows
        ]
    )


@app.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
async def delete_document(doc_id: str, vs: VectorstoreDep) -> None:
    EmbeddingStore, _ = _get_embedding_collection_store(vs._embedding_length)
    async with vs._make_async_session() as session:
        result = await session.execute(
            select(EmbeddingStore.id).where(
                EmbeddingStore.cmetadata["doc_id"].as_string() == doc_id
            )
        )
        ids = [str(row[0]) for row in result.fetchall()]
    if not ids:
        raise HTTPException(status_code=404, detail="Document not found.")
    await vs.adelete(ids=ids)


# ── Chat ─────────────────────────────────────────────────────────────────────


@app.post("/chat", tags=["chat"])
async def chat(req: ChatRequest, graph: GraphDep) -> ChatResponse:
    if settings.agent_backend == "pydantic_ai":
        return await _pydantic_ai_chat(req)
    assert graph is not None, (
        "graph is None — lifespan did not initialise LangGraph backend"
    )
    config = {"configurable": {"thread_id": req.session_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    ai_msg = result["messages"][-1]
    raw_citations = ai_msg.response_metadata.get("citations", [])
    return ChatResponse(
        session_id=req.session_id,
        answer=ai_msg.content,
        citations=[Citation(**c) for c in raw_citations],
    )


async def _pydantic_ai_chat(req: ChatRequest) -> ChatResponse:
    from .pydantic_ai_agent import AgentDeps

    agent = resources.pydantic_agent
    session_store = resources.session_store
    deps = AgentDeps(retriever=resources.retriever)
    history = await session_store.load(req.session_id)
    # NOTE: load→run→save is not atomic. Concurrent requests on the same session_id
    # can race and the second save will overwrite the first, silently losing a turn.
    # Acceptable for single-user sessions; fix with SELECT FOR UPDATE if multi-user
    # concurrent access on the same session becomes a requirement.
    result = await agent.run(
        req.message, deps=deps, message_history=history if history else None
    )
    await session_store.save(req.session_id, result.all_messages())
    return ChatResponse(
        session_id=req.session_id,
        answer=result.output,
        citations=[
            Citation(source=c.source, chunk=c.chunk, doc_id=c.doc_id)
            for c in deps.citations
        ],
    )


# ── Sessions ──────────────────────────────────────────────────────────────────


@app.get("/sessions/{session_id}/history", tags=["sessions"])
async def session_history(session_id: str, graph: GraphDep) -> SessionHistoryResponse:
    if settings.agent_backend == "pydantic_ai":
        return await _pydantic_ai_session_history(session_id)
    assert graph is not None, (
        "graph is None — lifespan did not initialise LangGraph backend"
    )
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)
    if not state.values:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = state.values.get("messages", [])
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[
            MessageEntry(role=getattr(m, "type", "unknown"), content=m.content)
            for m in messages
        ],
    )


async def _pydantic_ai_session_history(session_id: str) -> SessionHistoryResponse:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    session_store = resources.session_store
    all_messages = await session_store.load(session_id)
    if not all_messages:
        raise HTTPException(status_code=404, detail="Session not found.")

    entries: list[MessageEntry] = []
    for msg in all_messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = (
                        part.content
                        if isinstance(part.content, str)
                        else str(part.content)
                    )
                    entries.append(MessageEntry(role="human", content=content))
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    entries.append(MessageEntry(role="ai", content=part.content))
    return SessionHistoryResponse(session_id=session_id, messages=entries)
