from __future__ import annotations

from datetime import datetime, UTC

from langchain_text_splitters import RecursiveCharacterTextSplitter

from research_agent.models import Chunk, Document, StorageMode

# ~1000 chars / ~150 overlap is a standard middle ground: long enough to
# hold a full idea/paragraph for bge-small-en-v1.5 (512-token limit, and
# 1000 chars is comfortably under that for English text), short enough
# that a retrieved chunk doesn't drag in unrelated surrounding content.
# Applied uniformly across all sources rather than tuned per-category -
# a second tuning axis we'd only add once eval data showed uniform
# chunking was actually hurting a specific category.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _approx_token_count(text: str) -> int:
    """~4 chars/token is a standard English-text rule of thumb. Deliberately
    not tied to one specific tokenizer, since which LLM (Groq vs Ollama)
    answers a given query is a runtime config choice, not a fixed encoding."""
    return max(1, len(text) // 4)


def chunk_document(doc: Document, embedding_model: str) -> list[Chunk]:
    """Splits one Document into Chunks carrying copies of the parent fields
    retrieval/citation need, plus doc.version at chunk-creation time
    (Chunk.parent_version) so a later content change is detectable without
    a join back to `documents`.

    METADATA_ONLY documents have no body text to chunk (see StorageMode
    docstring) and are skipped - they exist for corpus stats/freshness only.
    """
    if doc.storage_mode == StorageMode.METADATA_ONLY:
        return []

    text = doc.full_text or doc.abstract
    if not text or not text.strip():
        return []

    now = datetime.now(UTC)
    pieces = _splitter.split_text(text)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            doc_id=doc.doc_id,
            chunk_index=i,
            text=piece,
            token_count=_approx_token_count(piece),
            source=doc.source,
            category=doc.category,
            title=doc.title,
            canonical_url=doc.canonical_url,
            publication_date=doc.publication_date,
            tags=doc.tags,
            parent_version=doc.version,
            embedding_model=embedding_model,
            created_at=now,
        )
        for i, piece in enumerate(pieces)
    ]
