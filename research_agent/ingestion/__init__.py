"""Source adapters and the shared ingestion pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from research_agent.config import Settings
from research_agent.models import Document, SourceName

from . import arxiv, github, rss, standards
from .pipeline import FetchFailure, IngestionResult, run_ingestion

# Every adapter exposes the same fetch(settings, **options) signature, so
# callers can run any source without knowing which one it is. Adding a
# source means writing one adapter module and adding one entry here.
ADAPTERS: dict[SourceName, Callable[..., Iterator[Document | FetchFailure]]] = {
    SourceName.ARXIV: arxiv.fetch,
    SourceName.GITHUB: github.fetch,
    SourceName.RSS_BLOG: rss.fetch,
    SourceName.GOV_STANDARDS: standards.fetch,
}


def ingest_source(
    source: SourceName,
    settings: Settings,
    trigger: str = "manual",
    **options,
) -> IngestionResult:
    """Fetch one source and store everything it returns.

    This is the single path into ingestion. A scheduled job, a manual
    refresh, and a historical backfill all call it and differ only in
    `trigger` and `options`, so none of them can drift apart in behaviour.

    Args:
        source: Which source to ingest.
        settings: Application configuration.
        trigger: What started this run: 'scheduled', 'manual', or 'backfill'.
        **options: Adapter-specific settings, such as arXiv's `max_results`
            and `start_offset`.

    Returns:
        Counts of documents fetched, added, updated, skipped, and failed.
    """
    documents = ADAPTERS[source](settings, **options)
    return run_ingestion(source, documents, str(settings.sqlite_path), trigger=trigger)


__all__ = ["ADAPTERS", "FetchFailure", "IngestionResult", "ingest_source", "run_ingestion"]
