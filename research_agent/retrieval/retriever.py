from __future__ import annotations

from datetime import date

from research_agent.config import Settings
from research_agent.models import DocumentStatus, SourceCategory
from research_agent.storage import get_connection

from .embeddings import Embedder
from .indexer import get_chroma_client, get_collection
from .reranker import Reranker
from .types import RetrievedChunk


def search(
    query: str,
    settings: Settings,
    categories: list[SourceCategory] | None = None,
    top_k: int = 8,
    candidates_per_category: int = 20,
    exclude_stale: bool = False,
    published_after: date | None = None,
    published_before: date | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Category-aware retrieval: searches only the collections for the
    requested categories (default: all three) rather than one
    undifferentiated index, so a question routed to "practitioner_knowledge"
    never spends its top-k budget on arXiv abstracts that happen to be
    superficially similar.

    Two-stage: over-fetch `candidates_per_category` per category via fast
    vector similarity, pool everything, then rerank the pool with a
    cross-encoder and truncate to `top_k` (see reranker.py for why).
    """
    embedder = embedder or Embedder(settings.embedding_model)
    reranker = reranker or Reranker()
    client = get_chroma_client(settings)
    target_categories = categories or list(SourceCategory)

    query_embedding = embedder.embed_query(query)
    candidates: list[RetrievedChunk] = []

    for category in target_categories:
        collection = get_collection(client, category)
        count = collection.count()
        if count == 0:
            continue
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidates_per_category, count),
        )
        for i in range(len(result["ids"][0])):
            meta = result["metadatas"][0][i]
            pub_date = meta.get("publication_date") or None
            if published_after and pub_date and pub_date < published_after.isoformat():
                continue
            if published_before and pub_date and pub_date > published_before.isoformat():
                continue
            candidates.append(
                RetrievedChunk(
                    chunk_id=result["ids"][0][i],
                    doc_id=meta["doc_id"],
                    text=result["documents"][0][i],
                    title=meta["title"],
                    canonical_url=meta["canonical_url"],
                    source=meta["source"],
                    category=meta["category"],
                    publication_date=pub_date,
                    score=1.0 - result["distances"][0][i],  # Chroma: smaller distance = more similar
                    reranked=False,
                )
            )

    if exclude_stale and candidates:
        candidates = _filter_stale(candidates, settings)

    if not candidates:
        return []

    return reranker.rerank(query, candidates, top_k=top_k)


def _filter_stale(candidates: list[RetrievedChunk], settings: Settings) -> list[RetrievedChunk]:
    """Checked live against SQLite rather than a status field baked into
    Chroma metadata: a document can be marked stale/superseded without its
    content_hash changing, so a copy of status in Chroma could silently
    drift out of date. A fresh lookup on every query can't."""
    conn = get_connection(settings.sqlite_path)
    doc_ids = tuple({c.doc_id for c in candidates})
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT doc_id, status FROM documents WHERE doc_id IN ({placeholders})", doc_ids
    ).fetchall()
    conn.close()
    status_by_doc = {r["doc_id"]: r["status"] for r in rows}
    return [c for c in candidates if status_by_doc.get(c.doc_id) == DocumentStatus.ACTIVE.value]
