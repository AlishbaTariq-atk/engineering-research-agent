"""MCP server exposing the knowledge base to other tools.

Any MCP-compatible client can connect over stdio and search the corpus,
fetch documents, and inspect how current the data is. See
`mcp_config.example.json` for client configuration.
"""

from __future__ import annotations

import os
from datetime import date

from research_agent.config import Settings

settings = Settings()

# Set before anything imports the embedding libraries, because the Hugging
# Face client reads this variable once at import time.
if settings.hf_hub_offline:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from research_agent import storage  # noqa: E402
from research_agent.models import SourceCategory  # noqa: E402
from research_agent.retrieval import Embedder, Reranker, search  # noqa: E402
from research_agent.retrieval.indexer import get_client, get_collection  # noqa: E402

mcp = FastMCP(
    "research-agent",
    instructions=(
        "Search a knowledge base of AI/ML technical literature (arXiv), practitioner "
        "knowledge (GitHub releases and READMEs, engineering blogs), and standards and "
        "regulations (NIST AI RMF, EU AI Act, US executive orders). Use "
        "search_knowledge_base to find passages, get_document to read a full record, "
        "and corpus_stats or source_freshness to see what is indexed and how current it is."
    ),
)

# Loaded once when the server starts, rather than on every search.
_embedder = Embedder(settings.embedding_model)
_reranker = Reranker(settings.reranker_model)


@mcp.tool()
def search_knowledge_base(
    query: str,
    categories: list[str] | None = None,
    top_k: int = 8,
    published_after: str | None = None,
    published_before: str | None = None,
    exclude_outdated: bool = True,
) -> list[dict]:
    """Search the knowledge base for passages relevant to a query.

    Args:
        query: What to search for, in natural language.
        categories: Limit the search to any of technical_literature,
            standards_regulations, or practitioner_knowledge. Omit to search all.
        top_k: How many passages to return.
        published_after: ISO date (YYYY-MM-DD); excludes anything older.
        published_before: ISO date (YYYY-MM-DD); excludes anything newer.
        exclude_outdated: Skip documents marked stale or superseded.

    Returns:
        Matching passages, most relevant first, each with its text, title,
        URL, source, category, publication date, and relevance score.
    """
    results = search(
        query,
        settings,
        categories=[SourceCategory(name) for name in categories] if categories else None,
        top_k=top_k,
        exclude_outdated=exclude_outdated,
        published_after=date.fromisoformat(published_after) if published_after else None,
        published_before=date.fromisoformat(published_before) if published_before else None,
        embedder=_embedder,
        reranker=_reranker,
    )
    return [
        {
            "chunk_id": result.chunk_id,
            "doc_id": result.doc_id,
            "title": result.title,
            "url": result.canonical_url,
            "source": result.source,
            "category": result.category,
            "publication_date": result.publication_date,
            "score": result.score,
            "text": result.text,
        }
        for result in results
    ]


@mcp.tool()
def get_document(doc_id: str) -> dict | None:
    """Fetch one complete document by its identifier.

    Args:
        doc_id: A document id, as returned by search_knowledge_base.

    Returns:
        Every stored field for that document, including its full text where
        available, or None if no such document exists.
    """
    conn = storage.connect(settings.sqlite_path)
    try:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


@mcp.tool()
def corpus_stats() -> dict:
    """Report what the knowledge base currently contains.

    Returns:
        Document counts broken down by category, source, storage mode, and
        status, plus how many searchable passages exist per category.
    """
    conn = storage.connect(settings.sqlite_path)
    try:
        stats = storage.corpus_stats(conn)
    finally:
        conn.close()

    client = get_client(settings)
    stats["indexed_passages"] = {
        category.value: get_collection(client, category).count() for category in SourceCategory
    }
    return stats


@mcp.tool()
def source_freshness() -> list[dict]:
    """Report when each source was last refreshed successfully.

    Lets a caller judge whether a source may be out of date before relying
    on what it returned.

    Returns:
        One entry per source, with the time of its last successful refresh
        and that run's document counts.
    """
    conn = storage.connect(settings.sqlite_path)
    try:
        return storage.source_freshness(conn)
    finally:
        conn.close()


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
