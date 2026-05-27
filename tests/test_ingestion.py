from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from app.ingestion import _build_documents, _make_doc_id, ingest_files


def test_make_doc_id_deterministic():
    id1 = _make_doc_id("test.md", "hello world")
    id2 = _make_doc_id("test.md", "hello world")
    assert id1 == id2


def test_make_doc_id_different_for_different_content():
    id1 = _make_doc_id("test.md", "hello")
    id2 = _make_doc_id("test.md", "world")
    assert id1 != id2


def test_build_documents_txt():
    content = "A" * 100
    docs = _build_documents("policy.txt", content)
    assert len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.metadata["filename"] == "policy.txt" for d in docs)
    assert all(d.metadata["chunk_index"] == i for i, d in enumerate(docs))
    assert all(d.id is not None for d in docs)


def test_build_documents_md():
    content = "# Title\nSome policy text.\n\n## Section\nMore text here."
    docs = _build_documents("handbook.md", content)
    assert len(docs) >= 1
    assert all(d.metadata["filename"] == "handbook.md" for d in docs)


def test_build_documents_idempotent_ids():
    content = "Some policy content."
    docs1 = _build_documents("file.txt", content)
    docs2 = _build_documents("file.txt", content)
    assert [d.id for d in docs1] == [d.id for d in docs2]


@pytest.mark.asyncio
async def test_ingest_files_calls_vectorstore():
    vs = MagicMock()
    vs.aadd_documents = AsyncMock(return_value=[])
    files = [("policy.txt", b"This is the PTO policy. Employees get 15 days off.")]
    reports = await ingest_files(vs, files)
    assert len(reports) == 1
    assert reports[0]["filename"] == "policy.txt"
    assert reports[0]["chunks"] >= 1
    vs.aadd_documents.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_files_idempotent(mock_vectorstore):
    files = [("policy.txt", b"Same content")]
    reports1 = await ingest_files(mock_vectorstore, files)
    reports2 = await ingest_files(mock_vectorstore, files)
    # Same doc_id on re-upload
    assert reports1[0]["doc_id"] == reports2[0]["doc_id"]
