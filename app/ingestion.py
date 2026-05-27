import hashlib
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import settings


def _make_doc_id(filename: str, content: str) -> str:
    return hashlib.sha256(f"{filename}:{content}".encode()).hexdigest()


def _split_markdown(content: str) -> list[Document]:
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=False
    )
    sections = md_splitter.split_text(content)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return char_splitter.split_documents(sections)


def _split_text(content: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_text(content)


def _build_documents(filename: str, content: str) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    doc_id = _make_doc_id(filename, content)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    if suffix == ".md":
        chunks = _split_markdown(content)
        docs = []
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                source=filename,
                filename=filename,
                doc_id=doc_id,
                chunk_index=i,
                uploaded_at=uploaded_at,
            )
            chunk.id = f"{doc_id}:{i}"
            docs.append(chunk)
        return docs
    else:
        texts = _split_text(content)
        return [
            Document(
                id=f"{doc_id}:{i}",
                page_content=text,
                metadata=dict(
                    source=filename,
                    filename=filename,
                    doc_id=doc_id,
                    chunk_index=i,
                    uploaded_at=uploaded_at,
                ),
            )
            for i, text in enumerate(texts)
        ]


async def ingest_files(
    vectorstore: PGVector,
    files: list[tuple[str, bytes]],
) -> list[dict]:
    """Ingest a list of (filename, content_bytes) into the vectorstore.

    Re-uploading the same file is idempotent because IDs are sha256-based.
    Returns one report dict per file.
    """
    reports = []
    for filename, raw in files:
        content = raw.decode("utf-8", errors="replace")
        docs = _build_documents(filename, content)
        ids = [doc.id for doc in docs]
        await vectorstore.aadd_documents(docs, ids=ids)
        reports.append(
            {
                "filename": filename,
                "chunks": len(docs),
                "doc_id": docs[0].metadata["doc_id"] if docs else None,
            }
        )
    return reports
