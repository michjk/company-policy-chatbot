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
from .streaming import register_streaming_endpoint
from .vectorstore import build_vectorstore, init_vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_observability(app)

    async with lifespan_checkpointer() as checkpointer:
        vs = build_vectorstore()
        await init_vectorstore(vs)
        retriever = vs.as_retriever(search_kwargs={"k": settings.retrieval_k})
        graph = build_rag_graph(retriever, checkpointer)

        resources.vectorstore = vs
        resources.graph = graph

        register_streaming_endpoint(app, graph, path="/chat/stream")
        yield


app = FastAPI(
    title="Company Policy Chatbot",
    description="RAG chatbot for company policy Q&A. Streaming via AG-UI protocol.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Documents ────────────────────────────────────────────────────────────────


@app.post("/documents/ingest")
async def ingest_documents(files: list[UploadFile], vs: VectorstoreDep):
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
    return {"files": reports, "chunks_total": total_chunks}


@app.get("/documents")
async def list_documents(vs: VectorstoreDep):
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
    return {
        "documents": [
            {"filename": r.filename, "doc_id": r.doc_id, "uploaded_at": r.uploaded_at}
            for r in rows
        ]
    }


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, vs: VectorstoreDep):
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


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest, graph: GraphDep):
    config = {"configurable": {"thread_id": req.session_id}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    ai_msg = result["messages"][-1]
    citations = ai_msg.response_metadata.get("citations", [])
    return {
        "session_id": req.session_id,
        "answer": ai_msg.content,
        "citations": citations,
    }


# ── Sessions ──────────────────────────────────────────────────────────────────


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str, graph: GraphDep):
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = state.values.get("messages", [])
    return {
        "session_id": session_id,
        "messages": [
            {"role": getattr(m, "type", "unknown"), "content": m.content}
            for m in messages
        ],
    }
