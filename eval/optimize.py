from __future__ import annotations

import json
import os
from typing import Any, Optional

import dspy
from dspy import Example, Prediction

from app.config import settings
from app.vectorstore import build_vectorstore
from eval.config import GEPA_BREADTH, GEPA_DEPTH, GEPA_MAX_ERRORS, configure_dspy_lm
from eval.metrics import combined_score
from eval.rag_program import PolicyRAG


def _build_retriever() -> Any:
    vs = build_vectorstore()
    return vs.as_retriever(search_kwargs={"k": settings.retrieval_k})


def _gepa_metric(
    gold: Example,
    pred: Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
) -> float:
    """Adapter wrapping combined_score to satisfy GEPAFeedbackMetric protocol."""
    return combined_score(gold, pred, trace)


def main() -> None:
    configure_dspy_lm()

    dataset_path = "eval/data/synthetic_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Run: make eval-data")
        return

    with open(dataset_path) as f:
        data: dict[str, list[dict[str, str]]] = json.load(f)

    train = [dspy.Example(**ex).with_inputs("question") for ex in data["train"]]
    dev = [dspy.Example(**ex).with_inputs("question") for ex in data["dev"]]

    retriever = _build_retriever()
    program = PolicyRAG(retriever=retriever)

    evaluator: dspy.Evaluate = dspy.Evaluate(
        devset=dev,
        metric=combined_score,
        num_threads=1,
        display_progress=True,
        max_errors=GEPA_MAX_ERRORS,
    )

    print("Scoring baseline on dev set…")
    baseline: float = evaluator(program)
    print(f"Baseline dev score: {baseline:.3f}")

    # breadth → reflection_minibatch_size (candidates per round)
    # depth   → max_full_evals (number of full-eval rounds)
    optimizer: dspy.GEPA = dspy.GEPA(
        metric=_gepa_metric,
        reflection_minibatch_size=GEPA_BREADTH,
        max_full_evals=GEPA_DEPTH,
    )

    print(
        f"\nRunning GEPA (reflection_minibatch_size={GEPA_BREADTH}, max_full_evals={GEPA_DEPTH})…"
    )
    compiled = optimizer.compile(program, trainset=train)

    print("\nScoring optimized program on dev set…")
    optimized: float = evaluator(compiled)
    print(f"Optimized dev score: {optimized:.3f}  (Δ {optimized - baseline:+.3f})")

    os.makedirs("eval/compiled", exist_ok=True)
    compiled.save("eval/compiled/optimized_rag.json")
    print("Saved → eval/compiled/optimized_rag.json")


if __name__ == "__main__":
    main()
