from langchain_core.language_models import BaseChatModel

from .config import settings


def build_chat_model() -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            streaming=True,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            streaming=True,
        )
    elif provider == "lmstudio":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
            model=settings.lmstudio_model,
            streaming=True,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. Choose openrouter, ollama, or lmstudio."
        )
