"""API-level integration tests for the PydanticAI backend (AGENT_BACKEND=pydantic_ai)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic_ai.models.test import TestModel

from app import pydantic_ai_agent as pa_module
from app.api import app
from app.deps import get_vectorstore, resources


@pytest.fixture
def pydantic_ai_client(sample_docs):
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=sample_docs)

    vs = MagicMock()
    vs.as_retriever = MagicMock(return_value=retriever)
    vs.aadd_documents = AsyncMock(return_value=[])

    # Build an agent backed by TestModel so no real LLM is hit
    with patch.object(pa_module, "build_pydantic_model", return_value=TestModel()):
        agent = pa_module.build_pydantic_agent()

    session_store = MagicMock()
    session_store.load = AsyncMock(return_value=[])
    session_store.save = AsyncMock()

    resources.vectorstore = vs
    resources.retriever = retriever
    resources.pydantic_agent = agent
    resources.session_store = session_store

    app.dependency_overrides[get_vectorstore] = lambda: vs

    with patch("app.api.settings") as mock_settings:
        mock_settings.agent_backend = "pydantic_ai"
        # Pass through all other settings attributes used in the endpoints
        mock_settings.cors_origins = ["*"]
        yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    app.dependency_overrides.clear()
    resources.pydantic_agent = None
    resources.session_store = None
    resources.retriever = None


@pytest.mark.asyncio
async def test_chat_pydantic_ai_returns_answer(pydantic_ai_client):
    async with pydantic_ai_client as c:
        resp = await c.post(
            "/chat",
            json={"session_id": "pai-1", "message": "What is the PTO policy?"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["session_id"] == "pai-1"
    assert "citations" in body


@pytest.mark.asyncio
async def test_chat_pydantic_ai_saves_session(pydantic_ai_client):
    """Verify that a successful /chat call persists messages in the session store."""
    async with pydantic_ai_client as c:
        await c.post(
            "/chat",
            json={"session_id": "pai-2", "message": "PTO policy?"},
        )
    store = resources.session_store
    store.save.assert_called_once()
    call_args = store.save.call_args
    assert call_args[0][0] == "pai-2"


@pytest.mark.asyncio
async def test_session_history_pydantic_ai_not_found(pydantic_ai_client):
    """Unknown session returns 404."""
    async with pydantic_ai_client as c:
        resp = await c.get("/sessions/unknown-session/history")
    assert resp.status_code == 404
