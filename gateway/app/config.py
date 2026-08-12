"""
Configuration management for SentinelLM Gateway.

Loads settings from environment variables with sensible defaults.
Uses Pydantic Settings for validation.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


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
    inference_backend: Literal["torch", "onnx", "triton"] = Field(
        default="torch",
        description=(
            "Semantic detector inference backend. 'torch' loads the model "
            "directly (model_path / semantic_model_id, as before). 'onnx' runs "
            "the exported graph at onnx_model_path via ONNX Runtime — requires "
            "running model/export_onnx.py first, since no ONNX build is "
            "published to the Hub. 'triton' sends inference over gRPC to a "
            "Triton Inference Server serving that same ONNX graph (see "
            "triton_deploy/) — tokenization still happens locally, using the "
            "tokenizer files at onnx_model_path."
        ),
    )
    onnx_model_path: str = Field(
        default="./model/onnx",
        description="Path to the ONNX-exported model directory (produced by model/export_onnx.py)",
    )
    triton_url: str = Field(
        default="localhost:8101",
        description=(
            "Triton Inference Server gRPC endpoint (host:port). Default matches "
            "triton_deploy/run.sh's host port mapping for Triton's gRPC port "
            "(container 8001 -> host 8101, since the gateway itself owns 8000)."
        ),
    )
    triton_model_name: str = Field(
        default="sentinellm",
        description="Model name as registered in the Triton model repository (triton_deploy/models/).",
    )

    # --- Policy ---
    policy_path: str = Field(
        default="./policies/default.yaml", description="Path to policy YAML file"
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Console API (/api/v1/*) ---
    console_api_key: str = Field(
        default="",
        description=(
            "Required value of the X-API-Key header on /api/v1/* routes. "
            "Empty (the default) disables auth — set this in any deployment "
            "reachable outside localhost."
        ),
    )
    console_cors_origins: str = Field(
        default="http://localhost:5173",
        description=(
            "Comma-separated list of origins allowed to call /api/v1/* "
            "(the console's dev/prod URL). Never '*' — this project's premise "
            "is access control."
        ),
    )
    console_read_only: bool = Field(
        default=False,
        description=(
            "When True, POST/PATCH/DELETE on /api/v1/policies return 403 — "
            "GETs are unaffected. Used on the public Hugging Face Spaces "
            "deployment so a live console demo is safely browsable by "
            "strangers without letting them mutate the shared policy set. "
            "Same shape as demo_mode, applied to the console API instead of "
            "the LLM proxy."
        ),
    )

    # --- OpenTelemetry ---
    otel_enabled: bool = Field(
        default=False,
        description=(
            "Auto-instrument FastAPI and export traces via OTLP/HTTP. Off by "
            "default so pytest and plain `uvicorn` runs don't spend time "
            "retrying a collector that isn't there — docker-compose turns "
            "this on and points it at the bundled Jaeger service."
        ),
    )
    otel_exporter_endpoint: str = Field(
        default="http://jaeger:4318/v1/traces",
        description="OTLP/HTTP traces endpoint.",
    )
    otel_service_name: str = Field(default="sentinellm-gateway")

    model_config = {
        "env_prefix": "SENTINELLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton settings instance
settings = Settings()
