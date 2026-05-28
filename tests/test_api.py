from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver

from app import rag_graph as rg
from app.api import app
from app.deps import get_graph, get_vectorstore
from app.rag_graph import build_rag_graph
from tests.conftest import FakeListChatModelWithTools


@pytest.fixture
def stub_llm():
    return FakeListChatModelWithTools(responses=["PTO is 15 days per year."] * 20)


@pytest.fixture
def client(stub_llm, sample_docs):
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=sample_docs)

    vs = MagicMock()
    vs.as_retriever = MagicMock(return_value=retriever)
    vs.aadd_documents = AsyncMock(return_value=[])

    checkpointer = MemorySaver()
    with patch.object(rg, "build_chat_model", return_value=stub_llm):
        graph = build_rag_graph(retriever, checkpointer)

    app.dependency_overrides[get_vectorstore] = lambda: vs
    app.dependency_overrides[get_graph] = lambda: graph
    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health(client):
    async with client as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ingest_documents(client):
    async with client as c:
        files = [
            (
                "files",
                (
                    "policy.txt",
                    b"Employees receive 15 days PTO per year.",
                    "text/plain",
                ),
            )
        ]
        resp = await c.post("/documents/ingest", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert "files" in body
    assert body["chunks_total"] >= 1


@pytest.mark.asyncio
async def test_ingest_rejects_unsupported_type(client):
    async with client as c:
        files = [("files", ("report.pdf", b"%PDF content", "application/pdf"))]
        resp = await c.post("/documents/ingest", files=files)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_json(client):
    async with client as c:
        resp = await c.post(
            "/chat",
            json={"session_id": "s1", "message": "What is the PTO policy?"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["session_id"] == "s1"
    assert "citations" in body


@pytest.mark.asyncio
async def test_session_history(client):
    async with client as c:
        await c.post("/chat", json={"session_id": "s2", "message": "PTO policy?"})
        resp = await c.get("/sessions/s2/history")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s2"
    assert len(body["messages"]) >= 2
