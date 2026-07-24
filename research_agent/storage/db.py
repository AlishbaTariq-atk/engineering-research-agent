from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """The one place a connection is opened, so the WAL/foreign-key pragmas
    (easy to forget per-connection, since SQLite scopes them to the
    connection, not the file) are never skipped."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """Idempotent: schema.sql is all CREATE TABLE/INDEX IF NOT EXISTS, so
    re-running this against an existing db is a no-op, not a reset."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        conn.close()
