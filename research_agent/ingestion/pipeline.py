from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

from pydantic import BaseModel

from research_agent.models import Document, SourceName
from research_agent.storage import get_connection

from .deduplicator import find_cross_source_duplicate


class IngestionResult(BaseModel):
    """Exactly the log statistics the assessment asks for, plus the
    duplicate count since that's a separate reported capability."""

    run_id: str
    source: str
    fetched: int = 0
    new: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    duplicates_detected: int = 0


def _document_exists(conn: sqlite3.Connection, doc_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()


def _insert_document(conn: sqlite3.Connection, doc: Document) -> None:
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
        "VALUES (?, ?, ?, ?, ?)",
        (doc.doc_id, doc.version, doc.content_hash, doc.ingested_at.isoformat(), "initial ingestion"),
    )


def _update_document(conn: sqlite3.Connection, doc: Document, new_version: int, change_summary: str) -> None:
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


def _touch_last_checked(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute(
        "UPDATE documents SET last_checked_at = ? WHERE doc_id = ?",
        (datetime.now(UTC).isoformat(), doc_id),
    )


def run_ingestion(
    source_name: SourceName,
    documents: Iterator[Document],
    db_path: str,
    trigger: str = "manual",
) -> IngestionResult:
    """Generic upsert + logging loop shared by every adapter. Each adapter
    is responsible only for producing valid Document objects (fetch, parse,
    map fields); idempotency, versioning, duplicate detection, and run
    logging all live here exactly once, so every source gets them for free
    and can't drift from each other.

    Commits per-document (not once at the end): if the `documents`
    generator itself raises partway through (e.g. a network drop on page 6
    of an arXiv query), everything already processed in this run stays
    committed and the run is logged as 'failed' with a summary - a
    transient failure loses nothing already ingested.
    """
    run_id = str(uuid.uuid4())
    conn = get_connection(db_path)
    result = IngestionResult(run_id=run_id, source=source_name.value)
    started_at = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO ingestion_runs (run_id, source, trigger, started_at, status) VALUES (?, ?, ?, ?, 'running')",
        (run_id, source_name.value, trigger, started_at),
    )
    conn.commit()

    try:
        for doc in documents:
            result.fetched += 1
            try:
                existing = _document_exists(conn, doc.doc_id)
                if existing is None:
                    if find_cross_source_duplicate(conn, doc.content_hash, doc.doc_id) is not None:
                        result.duplicates_detected += 1
                        # Still stored, not merged: an independent source has its
                        # own canonical_url and citation value. See deduplicator.py.
                    _insert_document(conn, doc)
                    result.new += 1
                elif existing["content_hash"] != doc.content_hash:
                    _update_document(
                        conn, doc,
                        new_version=existing["version"] + 1,
                        change_summary=f"content changed on re-check (source={source_name.value})",
                    )
                    result.updated += 1
                else:
                    _touch_last_checked(conn, doc.doc_id)
                    result.skipped += 1
                conn.commit()
            except Exception as exc:
                conn.rollback()
                result.failed += 1
                conn.execute(
                    "INSERT INTO ingestion_failures (run_id, source_id, url, error_message, occurred_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id, doc.source_id, str(doc.canonical_url), str(exc),
                        datetime.now(UTC).isoformat(),
                    ),
                )
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
