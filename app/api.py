from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
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
from .streaming import register_copilotkit_endpoint, register_streaming_endpoint
from .vectorstore import build_vectorstore, init_vectorstore

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

    async with lifespan_checkpointer() as checkpointer:
        vs = build_vectorstore()
        await init_vectorstore(vs)
        search_kwargs: dict = {"k": settings.retrieval_k}
        if settings.retrieval_search_type == "mmr":
            search_kwargs["fetch_k"] = settings.retrieval_k * 3
        retriever = vs.as_retriever(
            search_type=settings.retrieval_search_type,
            search_kwargs=search_kwargs,
        )
        graph = build_rag_graph(retriever, checkpointer)

        resources.vectorstore = vs
        resources.graph = graph

        register_streaming_endpoint(app, graph, path="/chat/stream")
        register_copilotkit_endpoint(app, graph, path="/copilotkit")
        yield


app = FastAPI(
    title="Company Policy Chatbot",
    description="RAG chatbot for company policy Q&A. Streaming via AG-UI protocol.",
    version="0.1.0",
    lifespan=lifespan,
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


# ── Sessions ──────────────────────────────────────────────────────────────────


@app.get("/sessions/{session_id}/history", tags=["sessions"])
async def session_history(session_id: str, graph: GraphDep) -> SessionHistoryResponse:
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
