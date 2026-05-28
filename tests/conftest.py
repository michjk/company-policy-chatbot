from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel


class FakeListChatModelWithTools(FakeListChatModel):
    """FakeListChatModel that accepts bind_tools without raising NotImplementedError."""

    def bind_tools(self, tools, **kwargs):
        return self


class FakeEmbeddings:
    """Deterministic 4-dim embeddings for tests (no Ollama needed)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10)] * 4 for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 10)] * 4

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            id="abc:0",
            page_content="Employees receive 15 days of PTO per year.",
            metadata={
                "filename": "pto.md",
                "doc_id": "abc",
                "chunk_index": 0,
                "uploaded_at": "2026-01-01T00:00:00+00:00",
            },
        ),
        Document(
            id="abc:1",
            page_content="New hires receive 5 days of PTO in their first year.",
            metadata={
                "filename": "pto.md",
                "doc_id": "abc",
                "chunk_index": 1,
                "uploaded_at": "2026-01-01T00:00:00+00:00",
            },
        ),
    ]


@pytest.fixture
def mock_vectorstore(sample_docs):
    vs = MagicMock()
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=sample_docs)
    vs.as_retriever = MagicMock(return_value=retriever)
    vs.aadd_documents = AsyncMock(return_value=["abc:0", "abc:1"])
    vs.adelete = AsyncMock()
    return vs
