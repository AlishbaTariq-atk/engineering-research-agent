from __future__ import annotations

import hashlib
import sqlite3

from research_agent.models import SourceName

from .parser import normalize_for_hash


def make_doc_id(source: SourceName, source_id: str) -> str:
    """Deterministic id: re-ingesting the same (source, source_id) always
    produces the same doc_id, making re-ingestion a natural upsert instead
    of a separate duplicate-detection problem."""
    digest = hashlib.sha256(f"{source.value}:{source_id}".encode()).hexdigest()
    return digest[:24]


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode()).hexdigest()


def find_cross_source_duplicate(
    conn: sqlite3.Connection, content_hash: str, exclude_doc_id: str
) -> sqlite3.Row | None:
    """Cross-source duplicate detection: a *different* source_id (so a
    different doc_id) with an identical content_hash - e.g. a paper
    mirrored on both arXiv and a lab's GitHub. Distinct from idempotency,
    which make_doc_id already handles for the same item re-fetched from
    the same source."""
    return conn.execute(
        "SELECT * FROM documents WHERE content_hash = ? AND doc_id != ? LIMIT 1",
        (content_hash, exclude_doc_id),
    ).fetchone()
