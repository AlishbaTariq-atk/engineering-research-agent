"""Tests for hashing, change detection, and the ingestion pipeline."""

from __future__ import annotations

from research_agent.ingestion import run_ingestion
from research_agent.ingestion.pipeline import FetchFailure, compute_content_hash, make_doc_id
from research_agent.models import SourceName
from research_agent.storage import connect

from .conftest import make_document


def test_doc_id_is_stable_for_the_same_item():
    """Re-fetching an item must produce the same id, or it would be stored twice."""
    assert make_doc_id(SourceName.ARXIV, "2401.12345") == make_doc_id(SourceName.ARXIV, "2401.12345")


def test_doc_id_differs_across_sources():
    """The same identifier used by two sources must not collide."""
    assert make_doc_id(SourceName.ARXIV, "12345") != make_doc_id(SourceName.GITHUB, "12345")


def test_content_hash_ignores_reformatting():
    """Re-wrapped text is the same content and must not look like an edit."""
    assert compute_content_hash("Reduces inference latency.") == compute_content_hash(
        "Reduces   inference\nlatency.  "
    )


def test_content_hash_detects_real_edits():
    assert compute_content_hash("The original text.") != compute_content_hash("The rewritten text.")


def test_new_document_is_stored_with_version_history(db_path):
    doc = make_document(doc_id=make_doc_id(SourceName.ARXIV, "a1"), source_id="a1")
    result = run_ingestion(SourceName.ARXIV, iter([doc]), db_path)

    assert (result.new, result.updated, result.skipped) == (1, 0, 0)

    conn = connect(db_path)
    assert conn.execute("SELECT version FROM documents WHERE doc_id = ?", (doc.doc_id,)).fetchone()[0] == 1
    assert len(conn.execute("SELECT * FROM document_versions WHERE doc_id = ?", (doc.doc_id,)).fetchall()) == 1


def test_unchanged_document_is_skipped_not_duplicated(db_path):
    """Running the same ingestion twice must not create a second copy."""
    doc = make_document(doc_id=make_doc_id(SourceName.ARXIV, "a2"), source_id="a2")
    run_ingestion(SourceName.ARXIV, iter([doc]), db_path)
    second = run_ingestion(SourceName.ARXIV, iter([doc]), db_path)

    assert (second.new, second.skipped) == (0, 1)

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_changed_content_bumps_version_and_records_history(db_path):
    """An edited source document keeps its id and gains a version."""
    doc_id = make_doc_id(SourceName.ARXIV, "a3")
    original = make_document(
        doc_id=doc_id, source_id="a3", abstract="Original.", content_hash=compute_content_hash("Original.")
    )
    revised = make_document(
        doc_id=doc_id, source_id="a3", abstract="Revised.", content_hash=compute_content_hash("Revised.")
    )

    run_ingestion(SourceName.ARXIV, iter([original]), db_path)
    result = run_ingestion(SourceName.ARXIV, iter([revised]), db_path)

    assert result.updated == 1

    conn = connect(db_path)
    row = conn.execute("SELECT version, abstract FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    assert row["version"] == 2
    assert row["abstract"] == "Revised."

    versions = conn.execute(
        "SELECT version FROM document_versions WHERE doc_id = ? ORDER BY version", (doc_id,)
    ).fetchall()
    assert [row["version"] for row in versions] == [1, 2]


def test_fetch_failure_is_recorded(db_path):
    """A failed item must be counted and stored, not silently dropped."""
    failure = FetchFailure(source_id="missing-repo", url="https://example.com/x", error_message="404")
    result = run_ingestion(SourceName.GITHUB, iter([failure]), db_path)

    assert (result.failed, result.new) == (1, 0)

    conn = connect(db_path)
    row = conn.execute("SELECT source_id, error_message FROM ingestion_failures").fetchone()
    assert (row["source_id"], row["error_message"]) == ("missing-repo", "404")


def test_one_failure_does_not_stop_the_run(db_path):
    """Good documents around a failed one must still be stored."""
    items = [
        make_document(doc_id=make_doc_id(SourceName.ARXIV, "g1"), source_id="g1"),
        FetchFailure(source_id="bad", url="https://example.com/bad", error_message="timeout"),
        make_document(doc_id=make_doc_id(SourceName.ARXIV, "g2"), source_id="g2"),
    ]
    result = run_ingestion(SourceName.ARXIV, iter(items), db_path)

    assert (result.new, result.failed) == (2, 1)

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


def test_cross_source_duplicate_is_counted_but_kept(db_path):
    """Identical content from two sources is flagged, and both are kept.

    Each source has its own URL and is worth citing separately, so the
    duplicate is reported rather than discarded.
    """
    shared = "This passage appears on two different sources."
    from_arxiv = make_document(
        doc_id=make_doc_id(SourceName.ARXIV, "d1"),
        source=SourceName.ARXIV,
        source_id="d1",
        abstract=shared,
        content_hash=compute_content_hash(shared),
    )
    from_github = make_document(
        doc_id=make_doc_id(SourceName.GITHUB, "d2"),
        source=SourceName.GITHUB,
        source_id="d2",
        abstract=shared,
        content_hash=compute_content_hash(shared),
    )

    run_ingestion(SourceName.ARXIV, iter([from_arxiv]), db_path)
    result = run_ingestion(SourceName.GITHUB, iter([from_github]), db_path)

    assert (result.new, result.duplicates_detected) == (1, 1)

    conn = connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
