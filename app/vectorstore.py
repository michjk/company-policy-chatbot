from langchain_postgres import PGVector

from .config import settings
from .embeddings import build_embeddings


def build_vectorstore() -> PGVector:
    return PGVector(
        embeddings=build_embeddings(),
        connection=settings.sqlalchemy_url,
        collection_name=settings.pgvector_collection,
        embedding_length=settings.embedding_dim,
        use_jsonb=True,
        async_mode=True,
        create_extension=True,
    )


async def init_vectorstore(vs: PGVector) -> None:
    """Create vector extension and tables if they don't exist."""
    await vs.acreate_vector_extension()
    await vs.acreate_tables_if_not_exists()
    await vs.acreate_collection()
