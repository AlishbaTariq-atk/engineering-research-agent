from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime

import chromadb

from research_agent.config import Settings
from research_agent.models import Document, SourceCategory
from research_agent.storage import get_connection

from .chunking import chunk_document
from .embeddings import Embedder

logger = logging.getLogger(__name__)


def get_chroma_client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def get_collection(client: chromadb.ClientAPI, category: SourceCategory):
    """One collection per category - this *is* category-aware retrieval: a
    search scoped to one category never touches vectors from the others,
    rather than searching everything and filtering matches afterward."""
    return client.get_or_create_collection(name=category.value)


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        source=row["source"],
        source_id=row["source_id"],
        category=row["category"],
        canonical_url=row["canonical_url"],
        title=row["title"],
        abstract=row["abstract"],
        full_text=row["full_text"],
        tags=json.loads(row["tags"]),
        publication_date=date.fromisoformat(row["publication_date"]) if row["publication_date"] else None,
        ingested_at=datetime.fromisoformat(row["ingested_at"]),
        last_checked_at=datetime.fromisoformat(row["last_checked_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        content_hash=row["content_hash"],
        version=row["version"],
        storage_mode=row["storage_mode"],
        status=row["status"],
        superseded_by=row["superseded_by"],
        source_metadata=json.loads(row["source_metadata"]),
    )


def find_documents_needing_indexing(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    """A document needs (re)indexing if it has retrievable text
    (storage_mode != metadata_only) and either has never been indexed or
    has changed since its last index (version has moved past
    last_indexed_version). This is what makes indexing incremental: a full
    re-embed of the whole corpus only happens once per document, for as
    long as its content stays unchanged."""
    query = """
        SELECT * FROM documents
        WHERE storage_mode != 'metadata_only'
          AND (last_indexed_version IS NULL OR last_indexed_version != version)
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    return conn.execute(query).fetchall()


def index_document(
    doc: Document,
    embedder: Embedder,
    client: chromadb.ClientAPI,
    conn: sqlite3.Connection,
) -> int:
    """Returns the number of chunks written. Purges any chunks already
    indexed for this doc_id before writing fresh ones, so re-indexing an
    updated document is idempotent rather than appending stale duplicates
    alongside the new content."""
    collection = get_collection(client, SourceCategory(doc.category))
    collection.delete(where={"doc_id": doc.doc_id})

    chunks = chunk_document(doc, embedder.model_name)
    if chunks:
        embeddings = embedder.embed_documents([c.text for c in chunks])
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.to_chroma_metadata() for c in chunks],
        )

    conn.execute("UPDATE documents SET last_indexed_version = ? WHERE doc_id = ?", (doc.version, doc.doc_id))
    conn.commit()
    return len(chunks)


def run_indexing(settings: Settings, limit: int | None = None) -> dict:
    """Entrypoint the CLI calls: index everything currently due for
    (re)indexing. Returns simple counts for the caller to print/log."""
    conn = get_connection(settings.sqlite_path)
    client = get_chroma_client(settings)
    embedder = Embedder(settings.embedding_model)

    rows = find_documents_needing_indexing(conn, limit=limit)
    result = {"documents_considered": len(rows), "documents_indexed": 0, "chunks_written": 0, "failed": 0}

    for row in rows:
        try:
            doc = _row_to_document(row)
            n_chunks = index_document(doc, embedder, client, conn)
            result["documents_indexed"] += 1
            result["chunks_written"] += n_chunks
        except Exception as exc:
            result["failed"] += 1
            logger.warning("indexing failed for doc_id=%s: %s", row["doc_id"], exc)

    conn.close()
    return result
