from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .deduplicator import compute_content_hash, make_doc_id
from .parser import extract_pdf_text
from .pipeline import FetchFailure

SOURCE = SourceName.GOV_STANDARDS

CURATED_DOCUMENTS_PATH = Path(__file__).parent / "sources" / "standards_documents.json"

logger = logging.getLogger(__name__)


def _load_curated_documents() -> list[dict]:
    return json.loads(CURATED_DOCUMENTS_PATH.read_text())


def fetch(settings: Settings, documents: list[dict] | None = None) -> Iterator[Document | FetchFailure]:
    """Government/standards bodies don't expose a paginated discovery API
    the way arXiv or GitHub do - there's no query endpoint that returns
    "all AI regulations." So this source is modeled differently on
    purpose: a curated list of specific document URLs, kept in
    sources/standards_documents.json (a data file, not code) and
    re-checked for content changes on every run.

    "Backfill" for this source means adding entries to that JSON file, not
    paging through history - a real, documented difference from the other
    three sources rather than a limitation hidden by pretending this
    source works like the others.

    Each document is fetched independently; a dead link or transient
    network error yields a FetchFailure (so it lands in ingestion_failures,
    not just a console log line) rather than aborting the run.
    """
    entries = documents if documents is not None else _load_curated_documents()

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for entry in entries:
            try:
                response = client.get(entry["url"])
                response.raise_for_status()
                full_text = extract_pdf_text(response.content)
                if not full_text:
                    raise ValueError("no extractable text in PDF")
            except Exception as exc:
                logger.warning("standards adapter: failed to fetch %s (%s)", entry["url"], exc)
                yield FetchFailure(source_id=entry["source_id"], url=entry["url"], error_message=str(exc))
                continue

            published: date | None = None
            if entry.get("published"):
                published = date.fromisoformat(entry["published"])

            now = datetime.now(UTC)
            yield Document(
                doc_id=make_doc_id(SOURCE, entry["source_id"]),
                source=SOURCE,
                source_id=entry["source_id"],
                category=SourceCategory.STANDARDS_REGULATIONS,
                canonical_url=entry["url"],
                title=entry["title"],
                abstract=full_text[:1000],
                full_text=full_text,
                tags=entry.get("tags", []),
                publication_date=published,
                ingested_at=now,
                last_checked_at=now,
                updated_at=now,
                content_hash=compute_content_hash(full_text),
                storage_mode=StorageMode.FULL_TEXT,
                source_metadata={"curated": True},
            )
