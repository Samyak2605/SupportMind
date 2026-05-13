from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "SupportMind"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"


class PathConfig(BaseModel):
    raw_data: Path
    processed_chunks: Path
    evaluation_dataset: Path
    vectorstore_dir: Path
    orders_db: Path
    checkpoints_db: Path
    llm_cache: Path


class ModelConfig(BaseModel):
    embedding_model: str
    reranker_model: str
    llm_model: str


class ChunkingConfig(BaseModel):
    chunk_size: int = 300
    chunk_overlap: int = 50


class RetrievalConfig(BaseModel):
    semantic_top_k: int = 12
    bm25_top_k: int = 12
    final_top_k: int = 6
    rerank_top_k: int = 8
    rrf_k: int = 60


class GenerationConfig(BaseModel):
    temperature: float = 0.1
    max_tokens: int = 700
    request_timeout_seconds: int = 45


class EvaluationConfig(BaseModel):
    sample_size: int = Field(default=40, ge=30, le=50)
    random_seed: int = 42


class OrdersSeedConfig(BaseModel):
    customer_count: int = 50
    order_count: int = 200
    failure_injection_rate: float = 0.05
    random_seed: int = 7


class AgentConfig(BaseModel):
    brain_model: str = "llama-3.3-70b-versatile"
    fallback_model: str = "gemini-flash-lite-latest"
    max_tool_iterations: int = 4
    refund_approval_cap: float = 200.0
    approval_timeout_hours: int = 24
    max_consecutive_failures: int = 2


class SupportMindConfig(BaseModel):
    app: AppConfig
    paths: PathConfig
    models: ModelConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    orders_seed: OrdersSeedConfig = OrdersSeedConfig()
    agent: AgentConfig = AgentConfig()


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str | None = None
    groq_eval_api_key: str | None = None
    gemini_api_key: str | None = None
    supportmind_config_path: str = "config/supportmind.yaml"

    # When set, points the Groq SDK's base_url at a self-hosted gateway
    # (Sarathi) instead of Groq's own API -- gets caching/routing/failover
    # for free since Groq's wire format is OpenAI-compatible and Sarathi
    # speaks the same shape. Unset by default (talks to Groq directly).
    groq_base_url: str | None = None

    def for_eval(self) -> "EnvironmentSettings":
        """A settings copy pointing Groq calls at GROQ_EVAL_API_KEY when set.

        RAGAS and the agent eval harness both make heavy, bursty batches of
        Groq calls; running them on the same key as the live demo/agent risks
        exhausting the demo's daily quota mid-eval. Falls back to the main
        key so evals still work with a single key configured."""
        return self.model_copy(update={"groq_api_key": self.groq_eval_api_key or self.groq_api_key})


def load_config(config_path: str | None = None) -> SupportMindConfig:
    env = EnvironmentSettings()
    path = Path(config_path or env.supportmind_config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = SupportMindConfig.model_validate(raw)

    for field_name, value in config.paths:
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        setattr(config.paths, field_name, resolved)

    return config


def load_settings() -> EnvironmentSettings:
    return EnvironmentSettings()
