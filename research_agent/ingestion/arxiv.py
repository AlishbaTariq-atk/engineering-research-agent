"""arXiv adapter: academic papers, fetched through the public arXiv API."""
# for clean modern type hints
from __future__ import annotations
# for pulling paper ID and version from URL pattern, and collapsing whitespace/line breaks
import re
#for pausing between requests to avoid rate-limiting
import time
# used for fetch() generator (returntype -> type hint -> Iterator)
from collections.abc import Iterator
# for arXiv's timestamps
from datetime import UTC, date, datetime
# XML/Atom feed parsing
import feedparser
# for HTTP requests to arXiv API and downloading PDFs
import httpx

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .pipeline import FetchFailure, compute_content_hash, extract_pdf_text, make_doc_id

SOURCE = SourceName.ARXIV
API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 3.0  # arXiv asks for at least three seconds between calls.
MAX_RETRIES = 6

_ID_WITH_VERSION = re.compile(r"/abs/(?P<id>[^v]+)v(?P<version>\d+)$")


def _parse_id(entry_id: str) -> tuple[str, int]:
    """Split an arXiv entry URL into its paper id and revision number.

    Entry ids look like 'http://arxiv.org/abs/2401.12345v2'. Dropping the
    version suffix means a revised paper keeps the same document id, so the
    revision is recorded as a content change rather than a second document.

    Args:
        entry_id: The `id` field from an arXiv API entry.

    Returns:
        The bare paper id and its revision number.
    """
    match = _ID_WITH_VERSION.search(entry_id)
    if match:
        return match.group("id"), int(match.group("version"))
    return entry_id.rsplit("/", 1)[-1], 1


def _pdf_url(entry) -> str | None:
    """Find the PDF link in an arXiv entry, if it has one."""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            return link.get("href")
    return None


def _parse_date(value: str | None) -> date | None:
    """Parse an arXiv timestamp into a date, returning None if unparseable."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").date()
    except ValueError:
        return None


def _get_page(client: httpx.Client, params: dict) -> httpx.Response:
    """Request one page of results, retrying if arXiv rate-limits us.

    Paging through thousands of results can trip the rate limiter even at
    the documented request spacing, so a 429 (Too Many Requests) is retried with a doubling
    delay instead of ending the run.

    Args:
        client: HTTP client to use.
        params: Query parameters for the arXiv API.

    Returns:
        A successful response.

    Raises:
        httpx.HTTPStatusError: If the request keeps failing after all retries.
    """
    response = client.get(API_URL, params=params)
    for attempt in range(MAX_RETRIES):
        if response.status_code != 429:
            response.raise_for_status()
            return response
        time.sleep(min(120.0, REQUEST_DELAY_SECONDS * (2**attempt) + 5))
        response = client.get(API_URL, params=params)
    response.raise_for_status()
    return response


def fetch(
    settings: Settings,
    max_results: int = 1000,
    full_text_limit: int = 300,
    start_offset: int = 0,
) -> Iterator[Document | FetchFailure]:
    """Fetch papers from the configured arXiv categories, newest first.

    Every paper contributes its title, abstract, and tags, which is cheap.
    Downloading and parsing PDFs is not, so full text is only retrieved for
    the first `full_text_limit` papers; the rest are stored abstract-only.

    Args:
        settings: Provides the list of arXiv categories to search.
        max_results: How many papers to fetch in total.
        full_text_limit: How many of them also get their PDF downloaded.
        start_offset: Where to start in the result list. Leave at 0 for the
            newest papers; use a larger value to backfill older ones.

    Yields:
        One Document per paper.
    """
    query = " OR ".join(f"cat:{category}" for category in settings.arxiv_category_list)
    fetched = 0
    start = start_offset

    with httpx.Client(timeout=30.0) as client:
        while fetched < max_results:
            page_size = min(PAGE_SIZE, max_results - fetched)
            response = _get_page(
                client,
                {
                    "search_query": query,
                    "start": start,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
            )
            entries = feedparser.parse(response.text).entries
            if not entries:
                return

            for entry in entries:
                source_id, revision = _parse_id(entry.id)
                abstract = re.sub(r"\s+", " ", entry.summary).strip()

                full_text = None
                storage_mode = StorageMode.ABSTRACT_ONLY
                pdf_link = _pdf_url(entry)
                if fetched < full_text_limit and pdf_link:
                    try:
                        full_text = extract_pdf_text(client.get(pdf_link, timeout=60.0).content) or None
                        if full_text:
                            storage_mode = StorageMode.FULL_TEXT
                    except httpx.HTTPError:
                        full_text = None  # Fall back to the abstract.

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
                    tags=[tag["term"] for tag in entry.get("tags", [])],
                    publication_date=_parse_date(entry.get("published")),
                    ingested_at=now,
                    last_checked_at=now,
                    updated_at=now,
                    content_hash=compute_content_hash(full_text or abstract),
                    storage_mode=storage_mode,
                    source_metadata={
                        "authors": [author.get("name") for author in entry.get("authors", [])],
                        "primary_category": entry.get("arxiv_primary_category", {}).get("term"),
                        "arxiv_version": revision,
                    },
                )
                fetched += 1
                if fetched >= max_results:
                    return

            start += page_size
            time.sleep(REQUEST_DELAY_SECONDS)
