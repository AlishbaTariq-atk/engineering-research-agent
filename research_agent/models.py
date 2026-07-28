"""Typed schema shared by every part of the system.

Each ingestion source produces `Document` objects; the retrieval layer
splits them into `Chunk` objects. Both are Pydantic models, so bad data
fails at the boundary instead of silently reaching the database.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class SourceCategory(StrEnum):
    """Top-level grouping a document belongs to. Each category is a
    separate vector collection, which is what makes search category-aware."""

    TECHNICAL_LITERATURE = "technical_literature"
    STANDARDS_REGULATIONS = "standards_regulations"
    PRACTITIONER_KNOWLEDGE = "practitioner_knowledge"


class SourceName(StrEnum):
    """Where a document was fetched from."""

    ARXIV = "arxiv"
    GITHUB = "github"
    RSS_BLOG = "rss_blog"
    GOV_STANDARDS = "gov_standards"


class StorageMode(StrEnum):
    """How much of a document's text was stored.

    Full text is expensive to fetch and embed, so it is only kept where a
    source provides it cheaply. METADATA_ONLY documents have no body text
    and are therefore skipped during indexing.
    """

    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    METADATA_ONLY = "metadata_only"


class DocumentStatus(StrEnum):
    """Whether a document should still be treated as current evidence.

    STALE means old with no known replacement; it can still be cited, with
    a freshness caveat. SUPERSEDED means a specific newer document replaces
    it (see `Document.superseded_by`) and it should normally be excluded.
    """

    ACTIVE = "active"
    STALE = "stale"
    SUPERSEDED = "superseded"


class Document(BaseModel):
    """A single item of source content in its canonical, source-agnostic form."""

    doc_id: str = Field(description="sha256(source:source_id), truncated. Stable across re-fetches.")
    source: SourceName
    source_id: str = Field(description="Identifier used by the source, e.g. an arXiv id or 'owner/repo@tag'.")
    category: SourceCategory
    canonical_url: HttpUrl

    title: str
    abstract: str | None = None
    full_text: str | None = None
    tags: list[str] = Field(default_factory=list)

    publication_date: date | None = Field(default=None, description="Null when the source publishes no date.")
    ingested_at: datetime = Field(description="Set once, when first stored.")
    last_checked_at: datetime = Field(description="Updated on every run that sees this document.")
    updated_at: datetime = Field(description="Updated only when the content itself changed.")

    content_hash: str = Field(description="sha256 of the normalised text. Drives change detection.")
    version: int = Field(default=1, ge=1, description="Incremented each time content_hash changes.")

    storage_mode: StorageMode
    status: DocumentStatus = DocumentStatus.ACTIVE
    superseded_by: str | None = Field(default=None, description="doc_id of the replacement, if any.")

    source_metadata: dict = Field(
        default_factory=dict,
        description="Source-specific extras (arXiv authors, GitHub repo, RSS feed name) that "
        "do not generalise across sources, so they are not promoted to real columns.",
    )


class Chunk(BaseModel):
    """A slice of a document's text, sized for embedding and retrieval.

    Fields such as title and URL are copied from the parent document rather
    than looked up, so building a citation from a search hit needs no
    database round-trip.
    """

    chunk_id: str = Field(description="'{doc_id}::{chunk_index}'. Re-chunking overwrites in place.")
    doc_id: str
    chunk_index: int = Field(ge=0)
    text: str
    token_count: int = Field(ge=0)

    # Copied from the parent document.
    source: SourceName
    category: SourceCategory
    title: str
    canonical_url: HttpUrl
    publication_date: date | None
    tags: list[str] = Field(default_factory=list)

    parent_version: int = Field(description="Document.version when this chunk was created.")
    embedding_model: str = Field(description="Model used, so mixed embedding spaces are detectable.")
    created_at: datetime

    def to_search_metadata(self) -> dict[str, str | int | float | bool]:
        """Flatten this chunk into metadata the vector store can hold.

        Returns:
            A dict of scalars only. The vector store rejects lists and
            nested objects, so tags are joined into a comma-separated
            string and a missing date becomes an empty string.
        """
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "source": self.source.value,
            "category": self.category.value,
            "title": self.title,
            "canonical_url": str(self.canonical_url),
            "publication_date": self.publication_date.isoformat() if self.publication_date else "",
            "tags": ",".join(self.tags),
            "parent_version": self.parent_version,
            "embedding_model": self.embedding_model,
        }
