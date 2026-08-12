"""Turning stored documents into searchable vectors.

Documents are split into overlapping chunks, embedded, and written to a
Chroma collection named after their category. Indexing is incremental:
only documents whose content has changed since they were last indexed are
re-embedded.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, date, datetime

import chromadb
# for doing chunk splitting of the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
# to load and run the embedding model
from sentence_transformers import SentenceTransformer

from research_agent.config import Settings
from research_agent.models import Chunk, Document, SourceCategory, StorageMode
from research_agent.storage import connect

logger = logging.getLogger(__name__)

# Around 1000 characters holds a full paragraph or idea while staying
# within the embedding model's 512-token input limit. The overlap keeps a
# sentence that straddles a boundary intact in at least one chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# bge-small is trained so that queries carry this prefix and indexed
# passages do not. Leaving it off measurably weakens retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


class Embedder:
    """Turns text into vectors using a local sentence-transformers model."""

    def __init__(self, model_name: str):
        """Load the embedding model.

        Args:
            model_name: A sentence-transformers model identifier.
        """
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing.

        Args:
            texts: Passage texts.

        Returns:
            One vector per input text.
        """
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Applies the model's query prefix, which indexed passages must not
        have, so queries and passages land in the same space correctly.

        Args:
            text: The search query.

        Returns:
            A single vector.
        """
        return self._model.encode(QUERY_PREFIX + text, convert_to_numpy=True, show_progress_bar=False).tolist()


def get_client(settings: Settings) -> chromadb.ClientAPI:
    """Open the on-disk vector store."""
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def get_collection(client: chromadb.ClientAPI, category: SourceCategory):
    """Get the collection holding one category's vectors.

    Categories are stored in separate collections, so a search limited to
    one category never scans the others' vectors.

    Args:
        client: An open vector store client.
        category: Which category's collection to return.

    Returns:
        The Chroma collection, created if it does not yet exist.
    """
    return client.get_or_create_collection(name=category.value)


def chunk_document(doc: Document, embedding_model: str) -> list[Chunk]:
    """Split a document into chunks ready for embedding.

    Args:
        doc: The document to split.
        embedding_model: Name of the model that will embed these chunks,
            recorded on each chunk.

    Returns:
        The document's chunks, or an empty list if it has no body text
        (metadata-only records, or an abstract that is blank).
    """
    if doc.storage_mode == StorageMode.METADATA_ONLY:
        return []

    text = doc.full_text or doc.abstract
    if not text or not text.strip():
        return []

    now = datetime.now(UTC)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::{index}",
            doc_id=doc.doc_id,
            chunk_index=index,
            text=piece,
            # Roughly four characters per token for English prose. Kept
            # approximate so it is not tied to one model's tokeniser.
            token_count=max(1, len(piece) // 4),
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
        for index, piece in enumerate(_splitter.split_text(text))
    ]


def _row_to_document(row: sqlite3.Row) -> Document:
    """Rebuild a Document from a database row."""
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


def index_document(doc: Document, embedder: Embedder, client: chromadb.ClientAPI, conn: sqlite3.Connection) -> int:
    """Embed one document's chunks and write them to the vector store.

    Existing chunks for the document are deleted first, so re-indexing an
    edited document replaces its old text instead of leaving both versions
    searchable.

    Args:
        doc: The document to index.
        embedder: Loaded embedding model.
        client: Open vector store client.
        conn: Open database connection, used to record that this version
            has been indexed.

    Returns:
        How many chunks were written.
    """
    collection = get_collection(client, SourceCategory(doc.category))
    collection.delete(where={"doc_id": doc.doc_id})

    chunks = chunk_document(doc, embedder.model_name)
    if chunks:
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embedder.embed_documents([chunk.text for chunk in chunks]),
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.to_search_metadata() for chunk in chunks],
        )

    conn.execute("UPDATE documents SET last_indexed_version = ? WHERE doc_id = ?", (doc.version, doc.doc_id))
    conn.commit()
    return len(chunks)


def run_indexing(settings: Settings, limit: int | None = None) -> dict:
    """Index every document that is new or has changed since last indexed.

    A document needs indexing when it has body text and its current version
    differs from the version last written to the vector store. Documents
    that have not changed are left alone, so repeat runs are cheap.

    Args:
        settings: Application configuration.
        limit: Maximum number of documents to process, for partial runs.

    Returns:
        Counts of documents considered, indexed, chunks written, and failures.
    """
    conn = connect(settings.sqlite_path)
    client = get_client(settings)
    embedder = Embedder(settings.embedding_model)

    query = """
        SELECT * FROM documents
        WHERE storage_mode != 'metadata_only'
          AND (last_indexed_version IS NULL OR last_indexed_version != version)
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()

    result = {"documents_considered": len(rows), "documents_indexed": 0, "chunks_written": 0, "failed": 0}
    for row in rows:
        try:
            result["chunks_written"] += index_document(_row_to_document(row), embedder, client, conn)
            result["documents_indexed"] += 1
        except Exception as exc:
            result["failed"] += 1
            logger.warning("Indexing failed for %s: %s", row["doc_id"], exc)

    conn.close()
    return result
