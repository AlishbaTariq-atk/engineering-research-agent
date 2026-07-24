from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for anything environment-specific. Adapters
    and services take a Settings instance rather than reading os.environ
    directly, so tests can pass a throwaway Settings without touching real
    env vars, and switching providers (e.g. groq -> ollama) is a config
    change, not a code change."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # storage
    sqlite_path: Path = Path("data/knowledge_base.db")
    chroma_path: Path = Path("data/chroma")

    # embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # generation - provider selected here, not in code
    llm_provider: str = "groq"  # "groq" | "ollama"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    # ingestion
    github_token: str | None = None  # optional, but raises GitHub API rate limit from 60/hr to 5000/hr

    log_level: str = "INFO"


settings = Settings()
