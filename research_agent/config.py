"""Application configuration, loaded from environment variables or `.env`.

Every value has a working default, so the system runs without a `.env`
file except where an external credential is genuinely required.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Components receive a Settings instance rather than reading the
    environment directly, which keeps provider and path choices out of the
    code and makes them overridable in tests.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage locations.
    sqlite_path: Path = Path("data/knowledge_base.db")
    chroma_path: Path = Path("data/chroma")

    # Embedding and reranking models (both run locally on CPU).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Skips the model cache re-validation requests the Hugging Face client
    # makes on every load. Enable once the models are cached locally.
    hf_hub_offline: bool = False

    # Text generation. Groq is a hosted API; Ollama runs locally.
    llm_provider: str = "groq"  # groq | ollama
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_host: str = "http://localhost:11434"
    # Must be a model already pulled locally (`ollama pull qwen2.5:1.5b`).
    # Every reasoning step asks the model for a typed result, so whichever
    # model is used has to produce reliable structured output; smaller
    # models may not.
    ollama_model: str = "qwen2.5:1.5b"

    # Raises the GitHub API rate limit from 60 to 5000 requests/hour.
    github_token: str | None = None

    # Sources to ingest. Comma-separated so that adding one is an
    # environment change rather than a code change.
    arxiv_categories: str = "cs.CL,cs.AI,cs.LG"
    github_repos: str = (
        "huggingface/transformers,vllm-project/vllm,langchain-ai/langchain,"
        "pytorch/pytorch,ollama/ollama,ggerganov/llama.cpp,"
        "run-llama/llama_index,huggingface/peft"
    )
    rss_feeds: str = (
        "https://huggingface.co/blog/feed.xml,"
        "https://pytorch.org/blog/feed.xml,"
        "https://bair.berkeley.edu/blog/feed.xml"
    )

    @property
    def arxiv_category_list(self) -> list[str]:
        """arXiv categories as a list."""
        return _split(self.arxiv_categories)

    @property
    def github_repo_list(self) -> list[str]:
        """GitHub repositories as a list of 'owner/name' strings."""
        return _split(self.github_repos)

    @property
    def rss_feed_list(self) -> list[str]:
        """RSS feed URLs as a list."""
        return _split(self.rss_feeds)


def _split(value: str) -> list[str]:
    """Split a comma-separated setting, dropping blanks and stray spaces."""
    return [item.strip() for item in value.split(",") if item.strip()]
