from unittest.mock import AsyncMock, MagicMock

import dspy
import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def dummy_lm():
    lm = dspy.utils.DummyLM(
        [
            {"reformulated_query": "PTO policy details"},
            {
                "answer": "Employees receive 15 days PTO.",
                "reasoning": "Based on the context.",
            },
            {"citations": "pto.md"},
        ]
    )
    dspy.configure(lm=lm)


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(
        return_value=[
            Document(
                page_content="Employees receive 15 days of PTO per year.",
                metadata={"filename": "pto.md", "chunk_index": 0},
            )
        ]
    )
    return retriever


def test_policy_rag_forward_returns_prediction(mock_retriever):
    from eval.rag_program import PolicyRAG

    program = PolicyRAG(retriever=mock_retriever)
    result = program.forward(question="How many PTO days do I get?")

    assert hasattr(result, "answer")
    assert hasattr(result, "context")
    assert hasattr(result, "citations")
    assert len(result.answer) > 0


def test_policy_rag_calls_retriever(mock_retriever):
    from eval.rag_program import PolicyRAG

    program = PolicyRAG(retriever=mock_retriever)
    program.forward(question="What is the remote work policy?")

    mock_retriever.ainvoke.assert_called_once()


def test_policy_rag_context_contains_chunk_text(mock_retriever):
    from eval.rag_program import PolicyRAG

    program = PolicyRAG(retriever=mock_retriever)
    result = program.forward(question="PTO?")

    assert "15 days" in result.context
