from __future__ import annotations

from .types import RetrievedChunk

# Character budget, not a count tied to one tokenizer - which LLM answers a
# given query (Groq vs Ollama) is a runtime config choice, matching the
# same reasoning as chunking.py's token-count approximation.
DEFAULT_CONTEXT_CHAR_BUDGET = 8000


def build_context(chunks: list[RetrievedChunk], char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET) -> str:
    """Assembles retrieved chunks into one context block, each tagged with
    a [n] marker the synthesis prompt is instructed to cite by number - so
    every claim in the final report traces back to a specific chunk/URL
    instead of "the corpus said so."

    Chunks arrive already ranked by the reranker, so truncating to the
    budget by taking chunks in order is truncating by relevance, not by
    chronology or insertion order.
    """
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.title} ({chunk.source}, {chunk.publication_date or 'n.d.'}) - {chunk.canonical_url}"
        block = f"{header}\n{chunk.text}\n"
        if used + len(block) > char_budget and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def citation_map(chunks: list[RetrievedChunk]) -> dict[int, RetrievedChunk]:
    """The [n] -> chunk mapping report_generator needs to turn inline
    citation numbers back into real titles/URLs for the report's citation list."""
    return {i: chunk for i, chunk in enumerate(chunks, start=1)}
