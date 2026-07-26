from __future__ import annotations

from research_agent.ingestion.deduplicator import compute_content_hash, make_doc_id, find_cross_source_duplicate
from research_agent.models import SourceName
from research_agent.storage import get_connection


def test_make_doc_id_is_deterministic():
    """The core idempotency guarantee: re-ingesting the same (source,
    source_id) must always produce the same doc_id, or every re-fetch
    would look like a new document instead of an upsert."""
    a = make_doc_id(SourceName.ARXIV, "2401.12345")
    b = make_doc_id(SourceName.ARXIV, "2401.12345")
    assert a == b


def test_make_doc_id_differs_by_source():
    """The same source_id string under two different sources must not
    collide - a GitHub repo happening to be named the same as an arXiv id
    should never merge into one document."""
    a = make_doc_id(SourceName.ARXIV, "12345")
    b = make_doc_id(SourceName.GITHUB, "12345")
    assert a != b


def test_compute_content_hash_ignores_whitespace_differences():
    """normalize_for_hash exists specifically so re-wrapped/re-formatted
    text (same content, different whitespace) doesn't register as a
    content change and trigger a spurious version bump."""
    a = compute_content_hash("Speculative decoding reduces latency.")
    b = compute_content_hash("Speculative   decoding\nreduces  latency.  ")
    assert a == b


def test_compute_content_hash_detects_real_changes():
    a = compute_content_hash("Version one of this abstract.")
    b = compute_content_hash("Version two of this abstract, substantially rewritten.")
    assert a != b


def test_find_cross_source_duplicate(db_path):
    """Cross-source duplicate detection is a content_hash lookup across
    *different* doc_ids - distinct from idempotency, which make_doc_id
    already guarantees for the same source re-fetched."""
    conn = get_connection(db_path)
    shared_hash = compute_content_hash("identical content mirrored on two sources")
    conn.execute(
        "INSERT INTO documents (doc_id, source, source_id, category, canonical_url, title, tags, "
        "ingested_at, last_checked_at, updated_at, content_hash, storage_mode) "
        "VALUES ('doc-a', 'arxiv', 'a', 'technical_literature', 'https://x/a', 'A', '[]', "
        "'2024-01-01T00:00:00', '2024-01-01T00:00:00', '2024-01-01T00:00:00', ?, 'abstract_only')",
        (shared_hash,),
    )
    conn.commit()

    dup = find_cross_source_duplicate(conn, shared_hash, exclude_doc_id="doc-b")
    assert dup is not None
    assert dup["doc_id"] == "doc-a"

    # Excluding the same doc_id that holds the hash must not match itself
    no_dup = find_cross_source_duplicate(conn, shared_hash, exclude_doc_id="doc-a")
    assert no_dup is None
