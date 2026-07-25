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
    # False by default so a first-time clone can still download the model.
    # Once bge-small/the cross-encoder are cached locally, flipping this to
    # true skips the dozens of HEAD requests HF's client makes to re-verify
    # the cache on every load - found by timing a real server startup,
    # which took 40+ seconds mostly waiting on huggingface.co, not on
    # actual model loading. Matters for a live demo: startup shouldn't
    # depend on Hub latency once the weights are already on disk.
    hf_hub_offline: bool = False

    # generation - provider selected here, not in code
    llm_provider: str = "groq"  # "groq" | "ollama"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    # ingestion
    github_token: str | None = None  # optional, but raises GitHub API rate limit from 60/hr to 5000/hr

    # Comma-separated rather than a list type: pydantic-settings expects list
    # env vars as JSON arrays, which is an awkward .env authoring experience.
    # A plain comma-separated string means adding a source is "edit .env",
    # never "edit adapter code".
    github_repos: str = (
        "huggingface/transformers,vllm-project/vllm,langchain-ai/langchain,"
        "pytorch/pytorch,ollama/ollama,ggerganov/llama.cpp"
    )
    rss_feeds: str = (
        "https://huggingface.co/blog/feed.xml,"
        "https://pytorch.org/blog/feed.xml,"
        "https://bair.berkeley.edu/blog/feed.xml"
    )
    arxiv_categories: str = "cs.CL,cs.AI,cs.LG"

    log_level: str = "INFO"

    @property
    def github_repo_list(self) -> list[str]:
        return [r.strip() for r in self.github_repos.split(",") if r.strip()]

    @property
    def rss_feed_list(self) -> list[str]:
        return [u.strip() for u in self.rss_feeds.split(",") if u.strip()]

    @property
    def arxiv_category_list(self) -> list[str]:
        return [c.strip() for c in self.arxiv_categories.split(",") if c.strip()]


settings = Settings()
