PRAGMA foreign_keys = ON;

-- Canonical document table. Mirrors research_agent.models.Document 1:1 so
-- there is exactly one place that defines "what a document is."
CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    category          TEXT NOT NULL,
    canonical_url     TEXT NOT NULL,
    title             TEXT NOT NULL,
    abstract          TEXT,
    full_text         TEXT,
    tags              TEXT NOT NULL DEFAULT '[]',    -- JSON array
    publication_date  TEXT,                          -- ISO date, nullable
    ingested_at       TEXT NOT NULL,
    last_checked_at   TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    content_hash      TEXT NOT NULL,
    version           INTEGER NOT NULL DEFAULT 1,
    storage_mode      TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active',
    superseded_by     TEXT REFERENCES documents(doc_id),
    source_metadata   TEXT NOT NULL DEFAULT '{}',    -- JSON object

    -- NULL until first indexed. Compared against `version` to decide
    -- whether this document's chunks in Chroma are stale and need
    -- re-embedding - the SQLite-side counterpart to Chunk.parent_version.
    last_indexed_version  INTEGER,

    -- doc_id is already derived from (source, source_id); this unique
    -- constraint makes that natural key explicit to a reader and guards
    -- against it independently of a hash bug/collision in doc_id itself.
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Cross-source duplicate detection (different source_id, same/near content)
-- is a distinct problem from idempotency (same source_id re-fetched, which
-- doc_id already handles via the UNIQUE above). This index makes "has this
-- content already been ingested from somewhere else" a cheap lookup.
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- Append-only version history: one row per content_hash change, never
-- updated or deleted. Satisfies "document versioning" as a real audit
-- trail rather than just the documents.version counter.
CREATE TABLE IF NOT EXISTS document_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id),
    version         INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    captured_at     TEXT NOT NULL,
    change_summary  TEXT,
    UNIQUE (doc_id, version)
);

-- One row per ingestion pipeline execution (scheduled, manual, or
-- backfill). Aggregate counters satisfy the spec's required log
-- statistics; per-run status/timestamps double as the source-freshness
-- data the MCP server needs ("when was this source last refreshed").
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id              TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    trigger             TEXT NOT NULL,                     -- 'scheduled' | 'manual' | 'backfill'
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL DEFAULT 'running',   -- 'running' | 'success' | 'failed'
    documents_fetched   INTEGER NOT NULL DEFAULT 0,
    documents_new       INTEGER NOT NULL DEFAULT 0,
    documents_updated   INTEGER NOT NULL DEFAULT 0,
    documents_skipped   INTEGER NOT NULL DEFAULT 0,
    documents_failed    INTEGER NOT NULL DEFAULT 0,
    error_summary       TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_source ON ingestion_runs(source, finished_at);

-- Per-document failure detail. ingestion_runs.documents_failed gives the
-- count; this gives the "which ones, and why" that failure reporting needs.
CREATE TABLE IF NOT EXISTS ingestion_failures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES ingestion_runs(run_id),
    source_id       TEXT NOT NULL,
    url             TEXT,
    error_message   TEXT NOT NULL,
    occurred_at     TEXT NOT NULL
);
