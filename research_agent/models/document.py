from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .enums import DocumentStatus, SourceCategory, SourceName, StorageMode


class Document(BaseModel):
    """Canonical, source-agnostic document record. Every ingestion adapter
    (arXiv, GitHub, RSS, gov/standards) must produce one of these regardless
    of how different the upstream API/format is."""

    doc_id: str = Field(
        description="Deterministic: sha256(f'{source}:{source_id}')[:24]. "
        "Not a random UUID, so re-ingesting the same source item is a "
        "natural upsert (same doc_id) rather than a duplicate-detection problem."
    )
    source: SourceName
    source_id: str = Field(
        description="Source-specific identifier, e.g. arXiv id, 'owner/repo@tag', RSS entry guid."
    )
    category: SourceCategory
    canonical_url: HttpUrl

    title: str
    abstract: str | None = None
    full_text: str | None = None
    tags: list[str] = Field(default_factory=list)

    publication_date: date | None = Field(
        default=None,
        description="Nullable: some practitioner sources (e.g. doc pages) have no clear publish date.",
    )
    ingested_at: datetime = Field(description="Set once, on first ingestion. Never updated after.")
    last_checked_at: datetime = Field(
        description="Updated on every pipeline run that touches this doc, whether or not content changed."
    )
    updated_at: datetime = Field(description="Set only when content_hash actually changes.")

    content_hash: str = Field(
        description="sha256 of normalized full_text (or abstract if no full_text). "
        "Drives idempotency, duplicate detection, and version bumps."
    )
    version: int = Field(default=1, ge=1)

    storage_mode: StorageMode
    status: DocumentStatus = DocumentStatus.ACTIVE
    superseded_by: str | None = Field(
        default=None, description="doc_id of the newer document; set only when status == SUPERSEDED."
    )

    source_metadata: dict = Field(
        default_factory=dict,
        description="Source-specific extras that don't generalize across sources "
        "(arXiv categories, GitHub stars/license, RSS feed name, ...) kept as a "
        "bag rather than forcing every source into identical extra columns.",
    )

    @field_validator("last_checked_at")
    @classmethod
    def _checked_not_before_ingested(cls, v: datetime, info) -> datetime:
        ingested = info.data.get("ingested_at")
        if ingested and v < ingested:
            raise ValueError("last_checked_at cannot precede ingested_at")
        return v


class DocumentVersionRecord(BaseModel):
    """Append-only audit row written whenever a document's content_hash
    changes. Satisfies the spec's explicit 'document versioning' pipeline
    requirement as an actual history, not just a counter on Document."""

    doc_id: str
    version: int
    content_hash: str
    captured_at: datetime
    change_summary: str | None = Field(
        default=None, description="e.g. 'abstract updated', 'full_text replaced (arXiv v2)'"
    )
