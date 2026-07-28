"""Searching the knowledge base.

Search runs in two stages. Vector similarity pulls a generous set of
candidates from each requested category, then a cross-encoder reranks that
small set and the best few are kept. The first stage is fast but compares
query and passage vectors computed separately; the second is slower but
reads the query and passage together, so it judges relevance more
accurately. Running it only over the shortlist keeps that cost bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sentence_transformers import CrossEncoder

from research_agent.config import Settings
from research_agent.models import DocumentStatus, SourceCategory
from research_agent.storage import connect

from .indexer import Embedder, get_client, get_collection

# Characters, not tokens, so the budget does not depend on which language
# model happens to be answering.
DEFAULT_CONTEXT_BUDGET = 8000


@dataclass
class SearchResult:
    """One retrieved passage, with everything needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    title: str
    canonical_url: str
    source: str
    category: str
    publication_date: str | None
    score: float


class Reranker:
    """Scores how well a passage answers a query, reading both together."""

    def __init__(self, model_name: str):
        """Load the cross-encoder model.

        Args:
            model_name: A sentence-transformers cross-encoder identifier.
        """
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Re-score candidates and keep the best ones.

        Args:
            query: The search query.
            candidates: Passages from the vector-similarity stage.
            top_k: How many to keep.

        Returns:
            The highest-scoring candidates, best first, with their scores
            replaced by the cross-encoder's. Scores are unbounded and may
            be negative; a strongly negative score means the passage does
            not answer the query.
        """
        if not candidates:
            return []

        scores = self._model.predict([(query, candidate.text) for candidate in candidates])
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:top_k]


def _drop_outdated(results: list[SearchResult], settings: Settings) -> list[SearchResult]:
    """Remove passages whose document is marked stale or superseded.

    Status is read from the database at query time rather than stored
    alongside the vectors, because a document can be marked outdated
    without its text changing — a copy held next to the vectors would
    quietly go out of date.

    Args:
        results: Candidate passages.
        settings: Application configuration.

    Returns:
        Only the passages whose documents are still active.
    """
    doc_ids = tuple({result.doc_id for result in results})
    conn = connect(settings.sqlite_path)
    try:
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(f"SELECT doc_id, status FROM documents WHERE doc_id IN ({placeholders})", doc_ids)
        status = {row["doc_id"]: row["status"] for row in rows}
    finally:
        conn.close()

    return [result for result in results if status.get(result.doc_id) == DocumentStatus.ACTIVE.value]


def search(
    query: str,
    settings: Settings,
    categories: list[SourceCategory] | None = None,
    top_k: int = 8,
    candidates_per_category: int = 20,
    exclude_outdated: bool = False,
    published_after: date | None = None,
    published_before: date | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> list[SearchResult]:
    """Find the passages that best answer a query.

    Args:
        query: What to search for.
        settings: Application configuration.
        categories: Restrict the search to these categories. Defaults to all.
        top_k: How many passages to return.
        candidates_per_category: How many to shortlist per category before
            reranking. Higher gives the reranker more to work with, at a cost.
        exclude_outdated: Drop passages from stale or superseded documents.
        published_after: Only include documents published on or after this date.
        published_before: Only include documents published on or before this date.
        embedder: Pre-loaded embedding model. Pass one when searching
            repeatedly, otherwise it is loaded per call.
        reranker: Pre-loaded reranker, same reasoning.

    Returns:
        Up to `top_k` passages, most relevant first.
    """
    embedder = embedder or Embedder(settings.embedding_model)
    reranker = reranker or Reranker(settings.reranker_model)
    client = get_client(settings)

    query_vector = embedder.embed_query(query)
    candidates: list[SearchResult] = []

    for category in categories or list(SourceCategory):
        collection = get_collection(client, category)
        stored = collection.count()
        if stored == 0:
            continue

        hits = collection.query(query_embeddings=[query_vector], n_results=min(candidates_per_category, stored))
        for position in range(len(hits["ids"][0])):
            metadata = hits["metadatas"][0][position]
            published = metadata.get("publication_date") or None

            if published_after and published and published < published_after.isoformat():
                continue
            if published_before and published and published > published_before.isoformat():
                continue

            candidates.append(
                SearchResult(
                    chunk_id=hits["ids"][0][position],
                    doc_id=metadata["doc_id"],
                    text=hits["documents"][0][position],
                    title=metadata["title"],
                    canonical_url=metadata["canonical_url"],
                    source=metadata["source"],
                    category=metadata["category"],
                    publication_date=published,
                    # Chroma returns distance, where smaller is closer.
                    # This stage's score is provisional; the reranker
                    # replaces it.
                    score=1.0 - hits["distances"][0][position],
                )
            )

    if exclude_outdated and candidates:
        candidates = _drop_outdated(candidates, settings)

    return reranker.rerank(query, candidates, top_k=top_k) if candidates else []


def build_context(results: list[SearchResult], char_budget: int = DEFAULT_CONTEXT_BUDGET) -> str:
    """Format retrieved passages into a single block for the language model.

    Each passage is numbered so the model can cite it by number, which is
    what lets every claim in the final report be traced to a specific
    source. Results arrive sorted by relevance, so trimming to the budget
    drops the least relevant passages.

    Args:
        results: Retrieved passages, most relevant first.
        char_budget: Rough ceiling on the size of the returned text.

    Returns:
        The numbered passages, each preceded by its title, source, date, and URL.
    """
    blocks: list[str] = []
    used = 0

    for number, result in enumerate(results, start=1):
        header = f"[{number}] {result.title} ({result.source}, {result.publication_date or 'n.d.'}) - {result.canonical_url}"
        block = f"{header}\n{result.text}\n"
        if used + len(block) > char_budget and blocks:
            break
        blocks.append(block)
        used += len(block)

    return "\n".join(blocks)


def citation_map(results: list[SearchResult]) -> dict[int, SearchResult]:
    """Map the numbers used in `build_context` back to their passages.

    Args:
        results: The same passages passed to `build_context`.

    Returns:
        Passage number to passage, so a citation the model writes can be
        resolved to a real title and URL.
    """
    return {number: result for number, result in enumerate(results, start=1)}
