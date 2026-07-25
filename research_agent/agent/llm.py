from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from research_agent.config import Settings


def get_llm(settings: Settings, temperature: float = 0.1) -> BaseChatModel:
    """Provider chosen by config (settings.llm_provider), not code. Groq is
    the default for demo-time speed; Ollama is the offline/zero-external-
    dependency fallback for when the venue's wifi can't be trusted. Both
    are plain LangChain chat models - this is exactly the "integration"
    use of LangChain the design settled on, not business logic."""
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=temperature)
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, temperature=temperature)
    raise ValueError(f"unknown llm_provider: {settings.llm_provider!r} (expected 'groq' or 'ollama')")
