from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def fixed_retriever(sample_docs):
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=sample_docs)
    return retriever


@pytest.fixture
def stub_llm():
    from langchain_core.language_models import FakeListChatModel

    return FakeListChatModel(responses=["Employees get 15 days PTO."])


@pytest.mark.asyncio
async def test_rag_graph_returns_answer(
    fixed_retriever, memory_checkpointer, stub_llm, monkeypatch
):
    from app import rag_graph

    monkeypatch.setattr(rag_graph, "build_chat_model", lambda: stub_llm)

    from app.rag_graph import build_rag_graph

    graph = build_rag_graph(fixed_retriever, memory_checkpointer)

    config = {"configurable": {"thread_id": "test-session-1"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What is the PTO policy?")]},
        config=config,
    )
    ai_msg = result["messages"][-1]
    assert isinstance(ai_msg, AIMessage)
    assert len(ai_msg.content) > 0
    assert "citations" in ai_msg.response_metadata


@pytest.mark.asyncio
async def test_rag_graph_preserves_history(
    fixed_retriever, memory_checkpointer, stub_llm, monkeypatch
):
    from app import rag_graph

    monkeypatch.setattr(rag_graph, "build_chat_model", lambda: stub_llm)

    from langchain_core.language_models import FakeListChatModel

    stub_llm2 = FakeListChatModel(
        responses=["Employees get 15 days PTO.", "New hires get 5 days."]
    )
    monkeypatch.setattr(rag_graph, "build_chat_model", lambda: stub_llm2)

    from app.rag_graph import build_rag_graph

    graph = build_rag_graph(fixed_retriever, memory_checkpointer)

    config = {"configurable": {"thread_id": "test-session-2"}}
    await graph.ainvoke(
        {"messages": [HumanMessage(content="What is the PTO policy?")]},
        config=config,
    )
    result2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="How many days for new hires?")]},
        config=config,
    )
    # Both human turns should be in history
    all_msgs = result2["messages"]
    human_msgs = [m for m in all_msgs if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 2


@pytest.mark.asyncio
async def test_rag_graph_citations_contain_sources(
    fixed_retriever, memory_checkpointer, stub_llm, monkeypatch, sample_docs
):
    from app import rag_graph

    monkeypatch.setattr(rag_graph, "build_chat_model", lambda: stub_llm)

    from app.rag_graph import build_rag_graph

    graph = build_rag_graph(fixed_retriever, memory_checkpointer)

    config = {"configurable": {"thread_id": "test-session-3"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="PTO?")]},
        config=config,
    )
    ai_msg = result["messages"][-1]
    citations = ai_msg.response_metadata["citations"]
    assert len(citations) == len(sample_docs)
    assert all("source" in c for c in citations)
