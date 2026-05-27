from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    # LangSmith observability
    langsmith_api_key: str = ""
    langsmith_project: str = "company-policy-chatbot"

    # OpenTelemetry observability
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "company-policy-chatbot"

    @property
    def sqlalchemy_url(self) -> str:
        """Convert standard postgres:// URL to SQLAlchemy psycopg async format."""
        return self.postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)


settings = Settings()
