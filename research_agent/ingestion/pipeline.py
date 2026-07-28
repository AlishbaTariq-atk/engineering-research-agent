"""The shared ingestion pipeline.

Source adapters only fetch and map data into `Document` objects. Everything
that must behave identically across sources — hashing, change detection,
version history, duplicate detection, run logging — lives here, so it is
written once rather than repeated in four adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from pydantic import BaseModel

from research_agent.models import Document, SourceName
from research_agent.storage import connect


# --------------------------------------------------------------------------
# Text helpers used by the adapters
# --------------------------------------------------------------------------


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        The document text, or an empty string if the PDF is unreadable.
        Returning empty rather than raising keeps one bad PDF from ending
        the whole ingestion run.
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def clean_html(html: str) -> str:
    """Strip HTML markup down to readable text.

    Args:
        html: An HTML fragment or document.

    Returns:
        Visible text with script and style blocks removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def make_doc_id(source: SourceName, source_id: str) -> str:
    """Build the stable identifier for a source item.

    The same source and source_id always produce the same id, so re-fetching
    an item updates the existing row instead of inserting a near-duplicate.

    Args:
        source: Which source the item came from.
        source_id: The source's own identifier for the item.

    Returns:
        A 24-character hex string.
    """
    return hashlib.sha256(f"{source.value}:{source_id}".encode()).hexdigest()[:24]


def compute_content_hash(text: str) -> str:
    """Hash a document's text to detect real content changes.

    Whitespace is collapsed first, so a source re-serving the same text with
    different line wrapping does not register as an edit.

    Args:
        text: The document's full text, or its abstract if there is no full text.

    Returns:
        A hex sha256 digest.
    """
    normalised = re.sub(r"\s+", " ", text or "").strip()
    return hashlib.sha256(normalised.encode()).hexdigest()


# --------------------------------------------------------------------------
# Pipeline results
# --------------------------------------------------------------------------


@dataclass
class FetchFailure:
    """An item an adapter could not fetch.

    Adapters yield this instead of a `Document` so that failures are counted
    and stored by the pipeline, rather than disappearing into a log line.
    Keeping it a yielded value also means adapters never touch the database.
    """

    source_id: str
    url: str
    error_message: str


class IngestionResult(BaseModel):
    """Counts describing a single ingestion run."""

    run_id: str
    source: str
    fetched: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    duplicates_detected: int = 0


# --------------------------------------------------------------------------
# Database operations
# --------------------------------------------------------------------------


def _find_duplicate_content(conn: sqlite3.Connection, content_hash: str, exclude_doc_id: str) -> sqlite3.Row | None:
    """Look for the same content already stored under a different document.

    This catches cross-posted content, such as a paper mirrored on both
    arXiv and a project's GitHub. It is a different check from re-fetching
    the same item, which `make_doc_id` already handles.

    Args:
        conn: Open database connection.
        content_hash: Hash of the incoming document's text.
        exclude_doc_id: The incoming document's own id, so it cannot match itself.

    Returns:
        The existing row, or None if this content is new.
    """
    return conn.execute(
        "SELECT * FROM documents WHERE content_hash = ? AND doc_id != ? LIMIT 1",
        (content_hash, exclude_doc_id),
    ).fetchone()


def _insert(conn: sqlite3.Connection, doc: Document) -> None:
    """Insert a new document and open its version history."""
    conn.execute(
        """
        INSERT INTO documents (
            doc_id, source, source_id, category, canonical_url, title, abstract,
            full_text, tags, publication_date, ingested_at, last_checked_at,
            updated_at, content_hash, version, storage_mode, status,
            superseded_by, source_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc.doc_id, doc.source.value, doc.source_id, doc.category.value,
            str(doc.canonical_url), doc.title, doc.abstract, doc.full_text,
            json.dumps(doc.tags),
            doc.publication_date.isoformat() if doc.publication_date else None,
            doc.ingested_at.isoformat(), doc.last_checked_at.isoformat(),
            doc.updated_at.isoformat(), doc.content_hash, doc.version,
            doc.storage_mode.value, doc.status.value, doc.superseded_by,
            json.dumps(doc.source_metadata),
        ),
    )
    conn.execute(
        "INSERT INTO document_versions (doc_id, version, content_hash, captured_at, change_summary) "
        "VALUES (?, ?, ?, ?, 'initial ingestion')",
        (doc.doc_id, doc.version, doc.content_hash, doc.ingested_at.isoformat()),
    )


def _update(conn: sqlite3.Connection, doc: Document, new_version: int, change_summary: str) -> None:
    """Overwrite a document's content and append a version-history row."""
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """
        UPDATE documents SET
            title = ?, abstract = ?, full_text = ?, tags = ?, publication_date = ?,
            last_checked_at = ?, updated_at = ?, content_hash = ?, version = ?,
            storage_mode = ?, source_metadata = ?
        WHERE doc_id = ?
        """,
        (
            doc.title, doc.abstract, doc.full_text, json.dumps(doc.tags),
            doc.publication_date.isoformat() if doc.publication_date else None,
            now, now, doc.content_hash, new_version, doc.storage_mode.value,
            json.dumps(doc.source_metadata), doc.doc_id,
        ),
    )
    conn.execute(
        "INSERT INTO document_versions (doc_id, version, content_hash, captured_at, change_summary) "
        "VALUES (?, ?, ?, ?, ?)",
        (doc.doc_id, new_version, doc.content_hash, now, change_summary),
    )


def _record_failure(conn: sqlite3.Connection, run_id: str, source_id: str, url: str, message: str) -> None:
    """Store one failed item against the run that encountered it."""
    conn.execute(
        "INSERT INTO ingestion_failures (run_id, source_id, url, error_message, occurred_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, source_id, url, message, datetime.now(UTC).isoformat()),
    )


def run_ingestion(
    source: SourceName,
    items: Iterator[Document | FetchFailure],
    db_path: str,
    trigger: str = "manual",
) -> IngestionResult:
    """Store everything an adapter produces, and log the run.

    For each document: unchanged content is skipped (only its
    last-checked timestamp moves), changed content bumps the version and
    appends to the version history, and anything unseen is inserted.

    Each item is committed on its own. If the adapter fails partway through
    — a dropped connection mid-pagination, say — everything already stored
    stays stored, and the run is recorded as failed.

    Args:
        source: Which source is being ingested.
        items: Documents from the adapter, plus FetchFailure entries for
            items it could not retrieve.
        db_path: Path to the SQLite database.
        trigger: What started this run: 'scheduled', 'manual', or 'backfill'.

    Returns:
        Counts of documents fetched, added, updated, skipped, and failed.

    Raises:
        Exception: Re-raised if the adapter itself fails, after the run has
            been marked failed in the database.
    """
    run_id = str(uuid.uuid4())
    conn = connect(db_path)
    result = IngestionResult(run_id=run_id, source=source.value)

    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, trigger, started_at, status) VALUES (?, ?, ?, ?, 'running')",
        (run_id, source.value, trigger, datetime.now(UTC).isoformat()),
    )
    conn.commit()

    try:
        for item in items:
            result.fetched += 1

            if isinstance(item, FetchFailure):
                result.failed += 1
                _record_failure(conn, run_id, item.source_id, item.url, item.error_message)
                conn.commit()
                continue

            try:
                existing = conn.execute(
                    "SELECT content_hash, version FROM documents WHERE doc_id = ?", (item.doc_id,)
                ).fetchone()

                if existing is None:
                    if _find_duplicate_content(conn, item.content_hash, item.doc_id):
                        # Counted, but still stored: each source has its own
                        # URL and is worth citing independently.
                        result.duplicates_detected += 1
                    _insert(conn, item)
                    result.new += 1
                elif existing["content_hash"] != item.content_hash:
                    _update(conn, item, existing["version"] + 1, f"content changed ({source.value})")
                    result.updated += 1
                else:
                    conn.execute(
                        "UPDATE documents SET last_checked_at = ? WHERE doc_id = ?",
                        (datetime.now(UTC).isoformat(), item.doc_id),
                    )
                    result.skipped += 1
                conn.commit()
            except Exception as exc:
                conn.rollback()
                result.failed += 1
                _record_failure(conn, run_id, item.source_id, str(item.canonical_url), str(exc))
                conn.commit()

        conn.execute(
            "UPDATE ingestion_runs SET finished_at = ?, status = 'success', documents_fetched = ?, "
            "documents_new = ?, documents_updated = ?, documents_skipped = ?, documents_failed = ? "
            "WHERE run_id = ?",
            (
                datetime.now(UTC).isoformat(), result.fetched, result.new,
                result.updated, result.skipped, result.failed, run_id,
            ),
        )
        conn.commit()
    except Exception as exc:
        conn.execute(
            "UPDATE ingestion_runs SET finished_at = ?, status = 'failed', error_summary = ? WHERE run_id = ?",
            (datetime.now(UTC).isoformat(), str(exc), run_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()

    return result
