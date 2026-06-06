import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

from app.pydantic_ai_sessions import PydanticSessionStore


def _make_pool_with_row(row_value):
    """Build a minimal mock AsyncConnectionPool whose cursor returns row_value."""
    cur = AsyncMock()
    cur.fetchone = AsyncMock(return_value=row_value)
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)

    conn = AsyncMock()
    conn.cursor = MagicMock(return_value=cur)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, cur


@pytest.mark.asyncio
async def test_load_returns_empty_for_unknown_session():
    pool, _ = _make_pool_with_row(None)
    store = PydanticSessionStore(pool)
    messages = await store.load("nonexistent-session")
    assert messages == []


@pytest.mark.asyncio
async def test_load_deserialises_stored_messages():
    # Build a minimal ModelRequest (user message) and serialise it
    sample_messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
    json_obj = json.loads(ModelMessagesTypeAdapter.dump_json(sample_messages))
    pool, _ = _make_pool_with_row((json_obj,))

    store = PydanticSessionStore(pool)
    loaded = await store.load("session-1")
    assert len(loaded) == 1
    assert isinstance(loaded[0], ModelRequest)


@pytest.mark.asyncio
async def test_save_calls_upsert():
    pool, cur = _make_pool_with_row(None)
    store = PydanticSessionStore(pool)

    messages = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
    await store.save("session-2", messages)

    cur.execute.assert_called_once()
    sql, params = cur.execute.call_args[0]
    assert "INSERT INTO pydantic_sessions" in sql
    assert "ON CONFLICT" in sql
    assert params[0] == "session-2"


@pytest.mark.asyncio
async def test_serialisation_round_trip():
    """Messages serialised by save and deserialised by load should be equal."""
    original = [ModelRequest(parts=[UserPromptPart(content="Round-trip test")])]
    json_str = ModelMessagesTypeAdapter.dump_json(original).decode()
    json_obj = json.loads(json_str)

    pool, _ = _make_pool_with_row((json_obj,))
    store = PydanticSessionStore(pool)
    loaded = await store.load("session-3")

    assert len(loaded) == 1
    assert isinstance(loaded[0], ModelRequest)
    part = loaded[0].parts[0]
    assert isinstance(part, UserPromptPart)
    assert part.content == "Round-trip test"
