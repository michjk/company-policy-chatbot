from __future__ import annotations

import dspy

from eval.config import FAITHFULNESS_WEIGHT, RELEVANCY_WEIGHT


class FaithfulnessJudge(dspy.Signature):  # type: ignore[misc]
    """Score how faithfully the answer is grounded in the context. Return 1.0 if all claims are supported by the context, 0.0 if none are."""

    context: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Float between 0.0 and 1.0")


class RelevancyJudge(dspy.Signature):  # type: ignore[misc]
    """Score how directly the answer addresses the user's question. Return 1.0 if fully and correctly answered, 0.0 if not at all."""

    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Float between 0.0 and 1.0")


_faithfulness_judge = dspy.Predict(FaithfulnessJudge)
_relevancy_judge = dspy.Predict(RelevancyJudge)


def combined_score(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: object = None,
) -> float:
    """Weighted combination of faithfulness and answer relevancy. Used as the GEPA metric."""
    f_score = float(_faithfulness_judge(context=pred.context, answer=pred.answer).score)
    r_score = float(
        _relevancy_judge(question=example.question, answer=pred.answer).score
    )
    return FAITHFULNESS_WEIGHT * f_score + RELEVANCY_WEIGHT * r_score
