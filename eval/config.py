from __future__ import annotations

import dspy

from app.config import settings

# Dataset generation
DATASET_QA_PER_CHUNK: int = 3
DATASET_MAX_CHUNKS: int = 50

# GEPA optimizer
GEPA_BREADTH: int = 8
GEPA_DEPTH: int = 3
GEPA_MAX_ERRORS: int = 5

# Metric weights (must sum to 1.0)
FAITHFULNESS_WEIGHT: float = 0.6
RELEVANCY_WEIGHT: float = 0.4


def configure_dspy_lm(model_override: str | None = None) -> None:
    """Configure DSPy with the same LLM provider as the app."""
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        lm: dspy.LM = dspy.LM(
            model=model_override or f"openai/{settings.openrouter_model}",
            api_base=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
    elif provider == "ollama":
        lm = dspy.LM(
            model=model_override or f"ollama_chat/{settings.ollama_chat_model}",
            api_base=settings.ollama_base_url,
        )
    elif provider == "lmstudio":
        lm = dspy.LM(
            model=model_override or f"openai/{settings.lmstudio_model}",
            api_base=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    dspy.configure(lm=lm)
