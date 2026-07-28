"""SQLite storage: schema, connections, and the queries used across the app.

SQLite holds all document metadata, version history, and ingestion logs.
Embeddings live separately in the vector store (see `retrieval/`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

-- One row per document. Mirrors research_agent.models.Document.
CREATE TABLE IF NOT EXISTS documents (
    doc_id                TEXT PRIMARY KEY,
    source                TEXT NOT NULL,
    source_id             TEXT NOT NULL,
    category              TEXT NOT NULL,
    canonical_url         TEXT NOT NULL,
    title                 TEXT NOT NULL,
    abstract              TEXT,
    full_text             TEXT,
    tags                  TEXT NOT NULL DEFAULT '[]',   -- JSON array
    publication_date      TEXT,                         -- ISO date
    ingested_at           TEXT NOT NULL,
    last_checked_at       TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    content_hash          TEXT NOT NULL,
    version               INTEGER NOT NULL DEFAULT 1,
    storage_mode          TEXT NOT NULL,
    status                TEXT NOT NULL DEFAULT 'active',
    superseded_by         TEXT REFERENCES documents(doc_id),
    source_metadata       TEXT NOT NULL DEFAULT '{}',   -- JSON object

    -- Null until indexed. When this differs from `version`, the document's
    -- chunks in the vector store are out of date and need re-embedding.
    last_indexed_version  INTEGER,

    -- doc_id is derived from these two, so this states the natural key
    -- explicitly and guards against a collision in the derived id.
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Supports "has this same content already arrived from another source?",
-- which is a different question from "have we seen this source_id before?"
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- Append-only history: one row per content change, never updated or deleted.
CREATE TABLE IF NOT EXISTS document_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id),
    version         INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    change_summary  TEXT,
    UNIQUE (doc_id, version)
);

-- One row per ingestion run. Doubles as the source-freshness record:
-- the latest successful run per source is when that source was last updated.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id              TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    trigger             TEXT NOT NULL,                    -- scheduled | manual | backfill
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'running',  -- running | success | failed
    documents_fetched   INTEGER NOT NULL DEFAULT 0,
    documents_new       INTEGER NOT NULL DEFAULT 0,
    documents_updated   INTEGER NOT NULL DEFAULT 0,
    documents_skipped   INTEGER NOT NULL DEFAULT 0,
    documents_failed    INTEGER NOT NULL DEFAULT 0,
    error_summary       TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source ON ingestion_runs(source, finished_at);

-- Which individual items failed and why. The run row gives the count;
-- this gives the detail.
CREATE TABLE IF NOT EXISTS ingestion_failures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES ingestion_runs(run_id),
    source_id       TEXT NOT NULL,
    url             TEXT,
    error_message   TEXT NOT NULL,
    occurred_at     TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this app relies on.

    WAL mode and foreign keys are per-connection settings in SQLite, not
    per-database, so they must be set every time rather than once at setup.

    Args:
        db_path: Path to the SQLite file.

    Returns:
        A connection whose rows behave like dicts (`row["title"]`).
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """Create the database and its tables if they do not already exist.

    Safe to call on every startup: the schema uses `IF NOT EXISTS`
    throughout, so running it against a populated database does nothing.

    Args:
        db_path: Path to the SQLite file. Parent directories are created.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def corpus_stats(conn: sqlite3.Connection) -> dict:
    """Summarise what is currently stored.

    Args:
        conn: An open database connection.

    Returns:
        Total document count plus breakdowns by category, source, storage
        mode, and status.
    """

    def counts(column: str) -> dict[str, int]:
        return dict(conn.execute(f"SELECT {column}, COUNT(*) FROM documents GROUP BY {column}").fetchall())

    return {
        "total_documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "by_category": counts("category"),
        "by_source": counts("source"),
        "by_storage_mode": counts("storage_mode"),
        "by_status": counts("status"),
    }


def source_freshness(conn: sqlite3.Connection) -> list[dict]:
    """Report when each source was last refreshed successfully.

    Args:
        conn: An open database connection.

    Returns:
        One entry per source describing its most recent successful run,
        including that run's document counts.
    """
    rows = conn.execute(
        """
        SELECT source, trigger, status, finished_at, documents_fetched,
               documents_new, documents_updated, documents_failed
        FROM ingestion_runs
        WHERE (source, finished_at) IN (
            SELECT source, MAX(finished_at) FROM ingestion_runs
            WHERE status = 'success' GROUP BY source
        )
        """
    ).fetchall()
    return [dict(row) for row in rows]
