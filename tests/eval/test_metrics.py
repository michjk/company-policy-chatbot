import dspy
import pytest


@pytest.fixture(autouse=True)
def dummy_lm():
    lm = dspy.utils.DummyLM([{"score": "0.9"}, {"score": "0.8"}])
    dspy.configure(lm=lm)
    yield
    dspy.settings.configure(lm=None)


def test_combined_score_returns_float():
    from eval.metrics import combined_score

    example = dspy.Example(
        question="What is PTO?",
        expected_answer="15 days.",
        context="Employees get 15 days.",
    ).with_inputs("question")

    pred = dspy.Prediction(
        answer="Employees receive 15 days PTO.",
        context="Employees get 15 days.",
        citations="pto.md",
    )

    score = combined_score(example, pred)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_combined_score_weighted():
    """Score = 0.6 * faithfulness + 0.4 * relevancy."""
    from eval.metrics import combined_score

    # DummyLM returns "0.9" then "0.8"
    example = dspy.Example(
        question="Q?", expected_answer="A.", context="C."
    ).with_inputs("question")
    pred = dspy.Prediction(answer="A.", context="C.", citations="")

    score = combined_score(example, pred)
    expected = 0.6 * 0.9 + 0.4 * 0.8
    assert abs(score - expected) < 0.01


def test_combined_score_accepts_trace_arg():
    from eval.metrics import combined_score

    example = dspy.Example(
        question="Q?", expected_answer="A.", context="C."
    ).with_inputs("question")
    pred = dspy.Prediction(answer="A.", context="C.", citations="")

    # Should not raise when trace is passed
    score = combined_score(example, pred, trace=None)
    assert isinstance(score, float)
