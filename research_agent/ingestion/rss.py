"""RSS adapter: posts from engineering blogs.

Known limits, both imposed by the publishers rather than this code:
feeds expose only their most recent posts, so there is no way to page back
through older ones; and some feeds publish titles and links with no body
text at all.
"""

from __future__ import annotations

import logging
from calendar import timegm
from collections.abc import Iterator
from datetime import UTC, date, datetime

import feedparser

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory, SourceName, StorageMode

from .pipeline import FetchFailure, clean_html, compute_content_hash, make_doc_id

SOURCE = SourceName.RSS_BLOG

logger = logging.getLogger(__name__)


def _entry_date(entry) -> date | None:
    """Read an entry's publication date, if the feed provides one."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime.fromtimestamp(timegm(parsed), tz=UTC).date()


def _entry_text(entry) -> tuple[str, str | None]:
    """Pull the summary and, where available, the full post body.

    Feeds differ in how much they publish: some include the whole article,
    others only a teaser. Whatever the feed provides is what gets stored.

    Args:
        entry: A parsed feed entry.

    Returns:
        The summary text, and the full body if the feed carries one that is
        meaningfully longer than the summary.
    """
    summary = clean_html(entry.get("summary", ""))
    content = entry.get("content")
    if content:
        body = clean_html(content[0].get("value", ""))
        if len(body) > len(summary) + 50:
            return (summary or body[:500]), body
    return summary, None


def fetch(settings: Settings) -> Iterator[Document | FetchFailure]:
    """Fetch recent posts from every configured feed.

    Feeds are read independently, so an unreachable one is reported and
    skipped without affecting the others.

    Args:
        settings: Provides the list of feed URLs.

    Yields:
        One Document per post, plus a FetchFailure for any unreadable feed.
    """
    for feed_url in settings.rss_feed_list:
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo and not feed.entries:
                raise feed.bozo_exception or ValueError("feed could not be parsed")
        except Exception as exc:
            logger.warning("RSS: could not read %s (%s)", feed_url, exc)
            yield FetchFailure(source_id=feed_url, url=feed_url, error_message=str(exc))
            continue

        feed_title = feed.feed.get("title", feed_url)
        for entry in feed.entries:
            source_id = entry.get("id") or entry.get("link")
            if not source_id:
                continue

            summary, body = _entry_text(entry)
            title = entry.get("title", "(untitled)")

            if body or summary:
                storage_mode = StorageMode.FULL_TEXT if body else StorageMode.ABSTRACT_ONLY
                content_hash = compute_content_hash(body or summary)
            else:
                # Title-only entries have no text to search or hash. Hashing
                # the empty string would make every such post look like a
                # copy of the others, so the hash is taken over its metadata.
                storage_mode = StorageMode.METADATA_ONLY
                content_hash = compute_content_hash(f"{source_id}:{title}")

            now = datetime.now(UTC)
            yield Document(
                doc_id=make_doc_id(SOURCE, source_id),
                source=SOURCE,
                source_id=source_id,
                category=SourceCategory.PRACTITIONER_KNOWLEDGE,
                canonical_url=entry.get("link", feed_url),
                title=title,
                abstract=summary or None,
                full_text=body,
                tags=[tag["term"] for tag in entry.get("tags", [])] if entry.get("tags") else [],
                publication_date=_entry_date(entry),
                ingested_at=now,
                last_checked_at=now,
                updated_at=now,
                content_hash=content_hash,
                storage_mode=storage_mode,
                source_metadata={"feed_title": feed_title, "feed_url": feed_url},
            )
