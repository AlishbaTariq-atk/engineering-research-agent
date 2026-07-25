from __future__ import annotations

import os
from datetime import date

from research_agent.config import Settings

settings = Settings()
if settings.hf_hub_offline:
    # Must be set before sentence_transformers/transformers is imported
    # anywhere, since huggingface_hub reads this env var at import time.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from research_agent.models import SourceCategory  # noqa: E402
from research_agent.retrieval import Embedder, Reranker  # noqa: E402
from research_agent.retrieval import search as retrieval_search  # noqa: E402
from research_agent.retrieval.indexer import get_chroma_client, get_collection  # noqa: E402
from research_agent.storage import get_connection  # noqa: E402

mcp = FastMCP(
    "research-agent",
    instructions=(
        "Search and inspect a knowledge base of AI/ML technical literature (arXiv), "
        "practitioner knowledge (GitHub releases/READMEs, engineering blogs), and "
        "standards/regulations (NIST AI RMF, EU AI Act, US executive orders). Use "
        "search_knowledge_base for semantic search, get_document for a full record "
        "by id, corpus_stats/source_freshness for questions about what's in the "
        "index and how current it is."
    ),
)

# Loaded once per server process (model weights), not per call - every
# other tool below is a cheap SQLite/Chroma lookup.
_embedder = Embedder(settings.embedding_model)
_reranker = Reranker()


@mcp.tool()
def search_knowledge_base(
    query: str,
    categories: list[str] | None = None,
    top_k: int = 8,
    published_after: str | None = None,
    published_before: str | None = None,
    exclude_stale: bool = True,
) -> list[dict]:
    """Semantic search over the knowledge base.

    Args:
        query: Natural-language search query.
        categories: Restrict to these categories - any of technical_literature,
            standards_regulations, practitioner_knowledge. Omit to search all three.
        top_k: Number of results to return after reranking.
        published_after: ISO date (YYYY-MM-DD). Excludes documents published before this.
        published_before: ISO date (YYYY-MM-DD). Excludes documents published after this.
        exclude_stale: If true (default), excludes documents marked stale or superseded.
    """
    cats = [SourceCategory(c) for c in categories] if categories else None
    results = retrieval_search(
        query,
        settings,
        categories=cats,
        top_k=top_k,
        exclude_stale=exclude_stale,
        published_after=date.fromisoformat(published_after) if published_after else None,
        published_before=date.fromisoformat(published_before) if published_before else None,
        embedder=_embedder,
        reranker=_reranker,
    )
    return [
        {
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "title": r.title,
            "url": r.canonical_url,
            "source": r.source,
            "category": r.category,
            "publication_date": r.publication_date,
            "score": r.score,
            "text": r.text,
        }
        for r in results
    ]


@mcp.tool()
def get_document(doc_id: str) -> dict | None:
    """Retrieve a full document record (all schema fields, including full_text
    if stored) by its doc_id, as returned in search_knowledge_base results."""
    conn = get_connection(settings.sqlite_path)
    try:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


@mcp.tool()
def corpus_stats() -> dict:
    """Corpus-wide statistics: document counts by category, source, storage
    mode, and status, plus indexed chunk counts per category collection."""
    conn = get_connection(settings.sqlite_path)
    try:
        by_category = dict(conn.execute("SELECT category, COUNT(*) FROM documents GROUP BY category").fetchall())
        by_source = dict(conn.execute("SELECT source, COUNT(*) FROM documents GROUP BY source").fetchall())
        by_storage_mode = dict(
            conn.execute("SELECT storage_mode, COUNT(*) FROM documents GROUP BY storage_mode").fetchall()
        )
        by_status = dict(conn.execute("SELECT status, COUNT(*) FROM documents GROUP BY status").fetchall())
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()

    client = get_chroma_client(settings)
    chunks_by_category = {cat.value: get_collection(client, cat).count() for cat in SourceCategory}

    return {
        "total_documents": total,
        "documents_by_category": by_category,
        "documents_by_source": by_source,
        "documents_by_storage_mode": by_storage_mode,
        "documents_by_status": by_status,
        "indexed_chunks_by_category": chunks_by_category,
    }


@mcp.tool()
def source_freshness() -> list[dict]:
    """Per-source freshness: when each source was last successfully
    refreshed and the stats from that run - lets a caller reason about
    whether a source's information might be out of date before trusting it."""
    conn = get_connection(settings.sqlite_path)
    try:
        rows = conn.execute(
            """
            SELECT source, trigger, status, finished_at, documents_fetched,
                   documents_new, documents_updated, documents_failed
            FROM ingestion_runs
            WHERE (source, finished_at) IN (
                SELECT source, MAX(finished_at) FROM ingestion_runs
                WHERE status = 'success' GROUP BY source
            )
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
