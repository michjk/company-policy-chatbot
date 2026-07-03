import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from psycopg_pool import AsyncConnectionPool
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from .config import settings


class PydanticSessionStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def load(self, session_id: str) -> list[ModelMessage]:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT messages FROM pydantic_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = await cur.fetchone()
        if row is None:
            return []
        # psycopg3 deserialises JSONB to a Python object; round-trip through JSON
        # so ModelMessagesTypeAdapter can validate with its own schema
        return ModelMessagesTypeAdapter.validate_json(json.dumps(row[0]))

    async def save(self, session_id: str, messages: list[ModelMessage]) -> None:
        json_str = ModelMessagesTypeAdapter.dump_json(messages).decode()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO pydantic_sessions (session_id, messages, updated_at)
                    VALUES (%s, %s::jsonb, NOW())
                    ON CONFLICT (session_id)
                    DO UPDATE SET messages = %s::jsonb, updated_at = NOW()
                    """,
                    (session_id, json_str, json_str),
                )


@asynccontextmanager
async def lifespan_session_store() -> AsyncIterator[PydanticSessionStore]:
    pool = AsyncConnectionPool(
        conninfo=settings.postgres_url,
        min_size=2,
        max_size=10,
        open=False,
        kwargs={"autocommit": True},
    )
    await pool.open()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS pydantic_sessions (
                        session_id TEXT PRIMARY KEY,
                        messages   JSONB NOT NULL DEFAULT '[]',
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
        yield PydanticSessionStore(pool)
    finally:
        await pool.close()
