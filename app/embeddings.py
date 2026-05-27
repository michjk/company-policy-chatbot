from langchain_ollama import OllamaEmbeddings

from .config import settings


def build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
