from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Agent backend: "langgraph" (default) or "pydantic_ai"
    agent_backend: Literal["langgraph", "pydantic_ai"] = "langgraph"

    # LLM provider
    llm_provider: str = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet-4-6"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ollama
    ollama_base_url: str = "http://ollama:11434"
    ollama_chat_model: str = "llama3.1"

    # LMStudio
    lmstudio_base_url: str = "http://host.docker.internal:1234/v1"
    lmstudio_model: str = "local-model"
    lmstudio_api_key: str = "lm-studio"

    # Embeddings
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Postgres
    postgres_url: str = "postgresql://app:app@postgres:5432/policies"
    pgvector_collection: str = "company_policies"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4
    retrieval_search_type: Literal["similarity", "mmr"] = "similarity"

    # CORS
    cors_origins: list[str] = ["*"]

    # LangSmith observability
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = "company-policy-chatbot"

    # OpenTelemetry observability
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "company-policy-chatbot"

    @field_validator("chunk_size", "retrieval_k", "embedding_dim")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @property
    def sqlalchemy_url(self) -> str:
        """Convert standard postgres:// URL to SQLAlchemy psycopg async format."""
        return self.postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)


settings = Settings()
