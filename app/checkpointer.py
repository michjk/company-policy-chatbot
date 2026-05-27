from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from .config import settings


@asynccontextmanager
async def lifespan_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Async context manager that owns the connection pool and checkpointer lifecycle."""
    pool = AsyncConnectionPool(
        conninfo=settings.postgres_url,
        min_size=2,
        max_size=10,
        open=False,
        kwargs={"autocommit": True},
    )
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(conn=pool)
        await checkpointer.setup()
        yield checkpointer
    finally:
        await pool.close()
