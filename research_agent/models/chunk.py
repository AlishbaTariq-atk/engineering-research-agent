from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, HttpUrl

from .enums import SourceCategory, SourceName


class Chunk(BaseModel):
    """A retrievable unit of text. Carries copies (not references) of the
    parent Document fields retrieval/citation need, so the hot path never
    has to join back to the documents table."""

    chunk_id: str = Field(description="f'{doc_id}::{chunk_index}' - deterministic, re-chunking overwrites in place.")
    doc_id: str
    chunk_index: int = Field(ge=0)
    text: str
    token_count: int = Field(ge=0)

    # --- inherited from parent Document ---
    source: SourceName
    category: SourceCategory
    title: str
    canonical_url: HttpUrl
    publication_date: date | None
    tags: list[str] = Field(default_factory=list)

    parent_version: int = Field(
        description="Document.version at the time this chunk was embedded. If the "
        "parent's version has since incremented, this chunk is stale and due for re-embedding."
    )
    embedding_model: str = Field(
        description="Model that embedded this chunk - guards against silently mixing "
        "embedding spaces in one Chroma collection if the model is ever changed."
    )
    created_at: datetime

    def to_chroma_metadata(self) -> dict[str, str | int | float | bool]:
        """Chroma metadata values must be str/int/float/bool - no lists/dicts.
        Tags are comma-joined rather than JSON-encoded: keeps simple
        equality/substring filters usable directly in Chroma's `where` clause;
        the tradeoff is it would break if a tag itself contained a comma
        (acceptable here since tags are single keywords)."""
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
