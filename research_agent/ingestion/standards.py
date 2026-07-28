"""Standards and regulations adapter: government and standards-body publications.

Unlike arXiv or GitHub, these bodies publish no search API — there is no
endpoint that returns "every AI regulation". This adapter therefore works
from an explicit list of document URLs in `standards_documents.json` and
re-checks each one for changes on every run. Adding coverage means adding
an entry to that file, not changing this code.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .pipeline import FetchFailure, compute_content_hash, extract_pdf_text, make_doc_id

SOURCE = SourceName.GOV_STANDARDS
DOCUMENT_LIST = Path(__file__).parent / "standards_documents.json"

logger = logging.getLogger(__name__)


def fetch(settings: Settings) -> Iterator[Document | FetchFailure]:
    """Download and parse every document in the curated list.

    Each document is fetched independently, so a moved or broken link is
    reported and skipped without stopping the rest.

    Args:
        settings: Unused here, but part of the shared adapter signature so
            every source can be invoked the same way.

    Yields:
        One Document per publication, plus a FetchFailure for any that
        could not be downloaded or read.
    """
    entries = json.loads(DOCUMENT_LIST.read_text())

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for entry in entries:
            try:
                response = client.get(entry["url"])
                response.raise_for_status()
                text = extract_pdf_text(response.content)
                if not text:
                    raise ValueError("no readable text in PDF")
            except Exception as exc:
                logger.warning("Standards: could not fetch %s (%s)", entry["url"], exc)
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
                abstract=text[:1000],
                full_text=text,
                tags=entry.get("tags", []),
                publication_date=published,
                ingested_at=now,
                last_checked_at=now,
                updated_at=now,
                content_hash=compute_content_hash(text),
                storage_mode=StorageMode.FULL_TEXT,
                source_metadata={"curated": True},
            )
