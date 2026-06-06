from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from app import pydantic_ai_agent as pa_module
from app.pydantic_ai_agent import AgentDeps, Citation, build_pydantic_agent


@pytest.fixture
def retriever(sample_docs):
    r = MagicMock()
    r.ainvoke = AsyncMock(return_value=sample_docs)
    return r


@pytest.fixture
def agent():
    with patch.object(pa_module, "build_pydantic_model", return_value=TestModel()):
        return build_pydantic_agent()


@pytest.mark.asyncio
async def test_agent_returns_answer(agent, retriever):
    with agent.override(model=TestModel()):
        deps = AgentDeps(retriever=retriever)
        result = await agent.run("What is the PTO policy?", deps=deps)
    assert isinstance(result.output, str)


@pytest.mark.asyncio
async def test_retrieve_context_populates_citations(agent, retriever):
    with agent.override(model=TestModel(call_tools=["retrieve_context"])):
        deps = AgentDeps(retriever=retriever)
        await agent.run("What is PTO?", deps=deps)
    assert len(deps.citations) > 0
    c = deps.citations[0]
    assert isinstance(c, Citation)
    assert c.source == "pto.md"
    assert c.doc_id == "abc"


@pytest.mark.asyncio
async def test_agent_preserves_history(agent, retriever):
    with agent.override(model=TestModel()):
        deps1 = AgentDeps(retriever=retriever)
        result1 = await agent.run("What is the PTO policy?", deps=deps1)
        history = result1.all_messages()

        deps2 = AgentDeps(retriever=retriever)
        result2 = await agent.run("How many days?", deps=deps2, message_history=history)
        all_msgs = result2.all_messages()

    # History from first run is included in second run's full message list
    assert len(all_msgs) > len(history)


@pytest.mark.asyncio
async def test_agent_citations_have_correct_fields(agent, retriever):
    with agent.override(model=TestModel(call_tools=["retrieve_context"])):
        deps = AgentDeps(retriever=retriever)
        await agent.run("PTO policy?", deps=deps)
    for citation in deps.citations:
        assert hasattr(citation, "source")
        assert hasattr(citation, "chunk")
        assert hasattr(citation, "doc_id")
