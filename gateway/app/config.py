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

    # --- Demo / Deployment Mode ---
    demo_mode: bool = Field(
        default=False,
        description=(
            "When True, the LLM proxy is disabled and /v1/chat/completions "
            "returns a clean 503. Used for the public Hugging Face Spaces demo, "
            "which showcases the detection pipeline via /scan and needs no LLM."
        ),
    )

    # --- LLM Backend ---
    llm_backend: str = Field(
        default="ollama", description="LLM backend type: 'ollama' or 'openai'"
    )
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        description="Base URL for Ollama API",
    )
    openai_api_key: str = Field(
        default="", description="OpenAI API key (if using OpenAI backend)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com", description="OpenAI API base URL"
    )
    llm_timeout: float = Field(
        default=120.0, description="Timeout in seconds for LLM requests"
    )

    # --- Database ---
    # Defaults to a self-contained async SQLite file so the app runs with no
    # external Postgres (e.g. on Hugging Face Spaces). docker-compose overrides
    # this with a Postgres URL for the full local stack.
    database_url: str = Field(
        default="sqlite+aiosqlite:///./sentinellm_audit.db",
        description="Async database connection URL (SQLite by default, Postgres in prod)",
    )
    database_url_sync: str = Field(
        default="sqlite:///./sentinellm_audit.db",
        description="Sync database connection URL (for migrations/tools)",
    )

    # --- Semantic Model ---
    model_path: str = Field(
        default="./model/trained",
        description="Path to fine-tuned DistilBERT model directory (local)",
    )
    semantic_model_id: str = Field(
        default="varshith145/sentinellm-pii-ner",
        description=(
            "Hugging Face Hub model id to load the semantic detector from when "
            "no local model directory is present (e.g. on Spaces)."
        ),
    )
    semantic_model_enabled: bool = Field(
        default=True,
        description="Enable semantic detector (disable if model not trained yet)",
    )

    # --- Policy ---
    policy_path: str = Field(
        default="./policies/default.yaml", description="Path to policy YAML file"
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
