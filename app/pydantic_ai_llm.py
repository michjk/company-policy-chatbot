from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .config import settings


def build_pydantic_model() -> OpenAIChatModel | OllamaModel:
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        return OpenAIChatModel(
            settings.openrouter_model,
            provider=OpenAIProvider(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
            ),
        )
    elif provider == "ollama":
        return OllamaModel(
            settings.ollama_chat_model,
            provider=OllamaProvider(base_url=settings.ollama_base_url),
        )
    elif provider == "lmstudio":
        return OpenAIChatModel(
            settings.lmstudio_model,
            provider=OpenAIProvider(
                base_url=settings.lmstudio_base_url,
                api_key=settings.lmstudio_api_key,
            ),
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. Choose openrouter, ollama, or lmstudio."
        )
