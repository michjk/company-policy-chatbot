import dspy


def test_configure_dspy_lm_openrouter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")

    from eval.config import configure_dspy_lm

    configure_dspy_lm()
    assert dspy.settings.lm is not None


def test_constants_are_positive():
    from eval.config import (
        DATASET_MAX_CHUNKS,
        DATASET_QA_PER_CHUNK,
        FAITHFULNESS_WEIGHT,
        GEPA_BREADTH,
        GEPA_DEPTH,
        RELEVANCY_WEIGHT,
    )

    assert DATASET_QA_PER_CHUNK > 0
    assert DATASET_MAX_CHUNKS > 0
    assert GEPA_BREADTH > 0
    assert GEPA_DEPTH > 0
    assert abs(FAITHFULNESS_WEIGHT + RELEVANCY_WEIGHT - 1.0) < 1e-9
