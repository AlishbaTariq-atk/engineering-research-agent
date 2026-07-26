from __future__ import annotations

from research_agent.ingestion.deduplicator import compute_content_hash, make_doc_id
from research_agent.ingestion.pipeline import FetchFailure, run_ingestion
from research_agent.models import SourceName
from research_agent.storage import get_connection

from .conftest import make_document


def test_new_document_is_inserted_and_versioned(db_path):
    doc = make_document(doc_id=make_doc_id(SourceName.ARXIV, "x1"), source_id="x1")
    result = run_ingestion(SourceName.ARXIV, iter([doc]), db_path, trigger="manual")

    assert result.new == 1
    assert result.updated == 0
    assert result.skipped == 0

    conn = get_connection(db_path)
    row = conn.execute("SELECT version FROM documents WHERE doc_id = ?", (doc.doc_id,)).fetchone()
    assert row["version"] == 1
    version_rows = conn.execute(
        "SELECT * FROM document_versions WHERE doc_id = ?", (doc.doc_id,)
    ).fetchall()
    assert len(version_rows) == 1


def test_reingesting_unchanged_document_is_skipped_not_duplicated(db_path):
    """The core idempotency contract: running the same fetch twice must
    not create a second document or a second version row."""
    doc = make_document(doc_id=make_doc_id(SourceName.ARXIV, "x2"), source_id="x2")
    run_ingestion(SourceName.ARXIV, iter([doc]), db_path, trigger="manual")
    result2 = run_ingestion(SourceName.ARXIV, iter([doc]), db_path, trigger="manual")

    assert result2.new == 0
    assert result2.skipped == 1

    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) c FROM documents WHERE doc_id = ?", (doc.doc_id,)).fetchone()["c"]
    assert count == 1


def test_changed_content_bumps_version_and_logs_audit_row(db_path):
    """A real content change (e.g. an arXiv v1 -> v2 revision) must bump
    version and append a new document_versions row - this is the whole
    point of tracking content_hash at all."""
    doc_id = make_doc_id(SourceName.ARXIV, "x3")
    v1 = make_document(doc_id=doc_id, source_id="x3", abstract="Original abstract.",
                        content_hash=compute_content_hash("Original abstract."))
    v2 = make_document(doc_id=doc_id, source_id="x3", abstract="Substantially revised abstract.",
                        content_hash=compute_content_hash("Substantially revised abstract."))

    run_ingestion(SourceName.ARXIV, iter([v1]), db_path, trigger="manual")
    result2 = run_ingestion(SourceName.ARXIV, iter([v2]), db_path, trigger="manual")

    assert result2.updated == 1
    conn = get_connection(db_path)
    row = conn.execute("SELECT version, abstract FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    assert row["version"] == 2
    assert row["abstract"] == "Substantially revised abstract."
    version_rows = conn.execute(
        "SELECT version FROM document_versions WHERE doc_id = ? ORDER BY version", (doc_id,)
    ).fetchall()
    assert [r["version"] for r in version_rows] == [1, 2]


def test_fetch_failure_is_recorded_not_silently_dropped(db_path):
    """A FetchFailure yielded by an adapter must land in ingestion_failures
    (queryable, persisted) and count toward documents_failed - not just a
    console log line that vanishes."""
    failure = FetchFailure(source_id="broken-repo", url="https://example.com/broken", error_message="404")
    result = run_ingestion(SourceName.GITHUB, iter([failure]), db_path, trigger="manual")

    assert result.failed == 1
    assert result.new == 0

    conn = get_connection(db_path)
    row = conn.execute("SELECT source_id, error_message FROM ingestion_failures").fetchone()
    assert row["source_id"] == "broken-repo"
    assert row["error_message"] == "404"


def test_one_bad_item_does_not_abort_the_whole_run(db_path):
    """Mixing a FetchFailure between two good documents must not prevent
    the good ones from being ingested - this is the "one bad item can't
    take the whole run down" guarantee ingestion relies on."""
    good_1 = make_document(doc_id=make_doc_id(SourceName.ARXIV, "g1"), source_id="g1")
    bad = FetchFailure(source_id="bad", url="https://example.com/bad", error_message="timeout")
    good_2 = make_document(doc_id=make_doc_id(SourceName.ARXIV, "g2"), source_id="g2")

    result = run_ingestion(SourceName.ARXIV, iter([good_1, bad, good_2]), db_path, trigger="manual")

    assert result.new == 2
    assert result.failed == 1
    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    assert count == 2


def test_cross_source_duplicate_is_stored_and_counted(db_path):
    """A cross-source duplicate (different source_id, identical content) is
    detected and counted, but still stored - see deduplicator.py for why
    it's not merged away."""
    shared_text = "This exact passage appears on two different sources."
    doc_a = make_document(
        doc_id=make_doc_id(SourceName.ARXIV, "dup-a"), source=SourceName.ARXIV, source_id="dup-a",
        abstract=shared_text, content_hash=compute_content_hash(shared_text),
    )
    doc_b = make_document(
        doc_id=make_doc_id(SourceName.GITHUB, "dup-b"), source=SourceName.GITHUB, source_id="dup-b",
        abstract=shared_text, content_hash=compute_content_hash(shared_text),
    )

    run_ingestion(SourceName.ARXIV, iter([doc_a]), db_path, trigger="manual")
    result = run_ingestion(SourceName.GITHUB, iter([doc_b]), db_path, trigger="manual")

    assert result.new == 1
    assert result.duplicates_detected == 1
    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    assert count == 2  # stored, not merged
