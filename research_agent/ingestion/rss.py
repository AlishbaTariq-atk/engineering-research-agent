from __future__ import annotations

import logging
from calendar import timegm
from collections.abc import Iterator
from datetime import UTC, date, datetime

import feedparser

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .deduplicator import compute_content_hash, make_doc_id
from .parser import clean_html

SOURCE = SourceName.RSS_BLOG

logger = logging.getLogger(__name__)


def _entry_date(entry) -> date | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(timegm(parsed), tz=UTC).date()


def _entry_text(entry) -> tuple[str, str | None]:
    """Returns (abstract, full_text). Feeds that publish full content
    (entry.content) give us full_text for free; feeds that only publish a
    teaser (entry.summary) get abstract_only - the feed publisher's own
    choice decides storage_mode here, not ours."""
    summary = clean_html(entry.get("summary", ""))
    content_list = entry.get("content")
    if content_list:
        full = clean_html(content_list[0].get("value", ""))
        if len(full) > len(summary) + 50:  # meaningfully more than just the teaser
            return (summary or full[:500]), full
    return summary, None


def fetch(settings: Settings, feed_urls: list[str] | None = None) -> Iterator[Document]:
    """One Document per feed entry.

    Each feed is parsed independently - a dead/unreachable feed URL is
    logged and skipped, not fatal to the others.

    Real limitation, documented rather than worked around: RSS backfill is
    bounded by whatever the publisher's feed currently exposes (typically
    the last 10-50 posts). Unlike arXiv/GitHub there is no pagination for
    older entries here - deeper blog history would need a different
    mechanism (sitemap crawl, web archive), which is out of scope.

    Second real limitation: some feeds (e.g. Hugging Face's) publish only
    title/link/date, no summary or content. Those become METADATA_ONLY
    documents rather than a fabricated abstract - fetching and scraping the
    linked article page for a real excerpt would fix this but adds a
    per-entry HTTP request and a new failure mode, so it's left as a
    documented gap rather than built under time pressure.
    """
    urls = feed_urls if feed_urls is not None else settings.rss_feed_list

    for feed_url in urls:
        try:
            parsed = feedparser.parse(feed_url)
            if parsed.bozo and not parsed.entries:
                raise parsed.bozo_exception or ValueError("unparseable feed")
        except Exception as exc:
            logger.warning("rss adapter: failed to fetch/parse %s (%s)", feed_url, exc)
            continue

        feed_title = parsed.feed.get("title", feed_url)
        for entry in parsed.entries:
            source_id = entry.get("id") or entry.get("link")
            if not source_id:
                continue

            abstract, full_text = _entry_text(entry)
            title = entry.get("title", "(untitled)")
            now = datetime.now(UTC)

            # Some feeds (e.g. Hugging Face's blog) publish title/link/date
            # only - no summary, no content. Hashing "" there would collapse
            # every such entry onto the same content_hash and manufacture
            # thousands of false "duplicates" (this happened in testing).
            # These are honestly METADATA_ONLY - the schema's exclusion of
            # metadata_only docs from chunking/embedding is exactly this
            # case - and the hash is derived from stable metadata instead.
            if full_text or abstract:
                storage_mode = StorageMode.FULL_TEXT if full_text else StorageMode.ABSTRACT_ONLY
                content_hash = compute_content_hash(full_text or abstract)
            else:
                storage_mode = StorageMode.METADATA_ONLY
                content_hash = compute_content_hash(f"{source_id}:{title}")

            yield Document(
                doc_id=make_doc_id(SOURCE, source_id),
                source=SOURCE,
                source_id=source_id,
                category=SourceCategory.PRACTITIONER_KNOWLEDGE,
                canonical_url=entry.get("link", feed_url),
                title=title,
                abstract=abstract or None,
                full_text=full_text,
                tags=[t["term"] for t in entry.get("tags", [])] if entry.get("tags") else [],
                publication_date=_entry_date(entry),
                ingested_at=now,
                last_checked_at=now,
                updated_at=now,
                content_hash=content_hash,
                storage_mode=storage_mode,
                source_metadata={"feed_title": feed_title, "feed_url": feed_url},
            )
