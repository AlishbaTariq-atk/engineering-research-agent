from __future__ import annotations

import re
import time
from collections.abc import Iterator
from datetime import UTC, date, datetime

import feedparser
import httpx

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .deduplicator import compute_content_hash, make_doc_id
from .parser import extract_pdf_text

SOURCE = SourceName.ARXIV

ARXIV_API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 3.0  # arXiv's documented politeness window between calls

_ARXIV_ID_VERSION_RE = re.compile(r"/abs/(?P<id>[^v]+)v(?P<version>\d+)$")


def _parse_arxiv_id(entry_id: str) -> tuple[str, int]:
    """entry.id looks like 'http://arxiv.org/abs/2401.12345v2'. The bare id
    (without the version suffix) is used as source_id, so a paper revision
    (v1 -> v2) becomes a content_hash change on the SAME doc_id - exactly
    what the version-tracking pipeline exists to catch, not a new document."""
    match = _ARXIV_ID_VERSION_RE.search(entry_id)
    if match:
        return match.group("id"), int(match.group("version"))
    return entry_id.rsplit("/", 1)[-1], 1


def _find_pdf_url(entry) -> str | None:
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            return link.get("href")
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None


def fetch(
    settings: Settings,
    categories: tuple[str, ...] = ("cs.CL", "cs.AI", "cs.LG"),
    max_results: int = 1000,
    full_text_limit: int = 300,
) -> Iterator[Document]:
    """Yields one Document per matching arXiv paper.

    Metadata (title/abstract/tags) is fetched for every matching paper -
    cheap, and what actually gets the corpus to ~50k documents. Full PDF
    text is downloaded and parsed only for the `full_text_limit` most
    recent papers: the "curated full-text subset" from the storage-
    strategy decision. Parsing 50k PDFs would dominate the ingestion time
    budget for marginal retrieval benefit over the abstract alone.

    `settings` isn't used by this adapter (arXiv's API is fully public) but
    is part of every adapter's signature so the pipeline/scheduler can call
    any source generically - GitHub's adapter needs it for an API token.
    """
    query = " OR ".join(f"cat:{c}" for c in categories)
    fetched = 0
    start = 0

    with httpx.Client(timeout=30.0) as client:
        while fetched < max_results:
            page_size = min(PAGE_SIZE, max_results - fetched)
            response = client.get(
                ARXIV_API_URL,
                params={
                    "search_query": query,
                    "start": start,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            if not feed.entries:
                break

            for entry in feed.entries:
                source_id, arxiv_version = _parse_arxiv_id(entry.id)
                abstract = re.sub(r"\s+", " ", entry.summary).strip()
                pdf_url = _find_pdf_url(entry)

                full_text: str | None = None
                storage_mode = StorageMode.ABSTRACT_ONLY
                if fetched < full_text_limit and pdf_url:
                    try:
                        pdf_bytes = client.get(pdf_url, timeout=60.0).content
                        full_text = extract_pdf_text(pdf_bytes) or None
                        if full_text:
                            storage_mode = StorageMode.FULL_TEXT
                    except httpx.HTTPError:
                        full_text = None  # falls back to abstract_only

                now = datetime.now(UTC)
                yield Document(
                    doc_id=make_doc_id(SOURCE, source_id),
                    source=SOURCE,
                    source_id=source_id,
                    category=SourceCategory.TECHNICAL_LITERATURE,
                    canonical_url=f"https://arxiv.org/abs/{source_id}",
                    title=re.sub(r"\s+", " ", entry.title).strip(),
                    abstract=abstract,
                    full_text=full_text,
                    tags=[t["term"] for t in entry.get("tags", [])],
                    publication_date=_parse_date(entry.get("published")),
                    ingested_at=now,
                    last_checked_at=now,
                    updated_at=now,
                    content_hash=compute_content_hash(full_text or abstract),
                    storage_mode=storage_mode,
                    source_metadata={
                        "authors": [a.get("name") for a in entry.get("authors", [])],
                        "primary_category": entry.get("arxiv_primary_category", {}).get("term"),
                        "arxiv_version": arxiv_version,
                    },
                )
                fetched += 1
                if fetched >= max_results:
                    break

            start += page_size
            time.sleep(REQUEST_DELAY_SECONDS)
