"""
Configuration management for SentinelLM Gateway.

Loads settings from environment variables with sensible defaults.
Uses Pydantic Settings for validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Gateway configuration loaded from environment variables."""

    # --- Application ---
    app_name: str = "SentinelLM Gateway"
    debug: bool = False

    # --- LLM Backend ---
    llm_backend: str = Field(
        default="ollama",
        description="LLM backend type: 'ollama' or 'openai'"
    )
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        description="Base URL for Ollama API"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (if using OpenAI backend)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com",
        description="OpenAI API base URL"
    )
    llm_timeout: float = Field(
        default=120.0,
        description="Timeout in seconds for LLM requests"
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql+asyncpg://ppg:ppg@db:5432/ppg",
        description="Async PostgreSQL connection URL"
    )
    database_url_sync: str = Field(
        default="postgresql://ppg:ppg@db:5432/ppg",
        description="Sync PostgreSQL connection URL (for migrations)"
    )

    # --- Semantic Model ---
    model_path: str = Field(
        default="./model/trained",
        description="Path to fine-tuned DistilBERT model directory"
    )
    semantic_model_enabled: bool = Field(
        default=True,
        description="Enable semantic detector (disable if model not trained yet)"
    )

    # --- Policy ---
    policy_path: str = Field(
        default="./policies/default.yaml",
        description="Path to policy YAML file"
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {
        "env_prefix": "SENTINELLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton settings instance
settings = Settings()
