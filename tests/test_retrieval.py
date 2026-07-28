"""Tests for splitting documents into chunks and building model context."""

from __future__ import annotations

from research_agent.models import StorageMode
from research_agent.retrieval import build_context, chunk_document, citation_map
from research_agent.retrieval.search import SearchResult

from .conftest import make_document


def _result(chunk_id: str, title: str, score: float = 1.0) -> SearchResult:
    """Build a search result for tests."""
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text="retrieved passage text",
        title=title,
        canonical_url=f"https://example.com/{chunk_id}",
        source="arxiv",
        category="technical_literature",
        publication_date="2024-01-01",
        score=score,
    )


def test_metadata_only_documents_are_not_chunked():
    """Documents with no body text have nothing to search."""
    doc = make_document(storage_mode=StorageMode.METADATA_ONLY, abstract=None, full_text=None)
    assert chunk_document(doc, "test-model") == []


def test_blank_text_produces_no_chunks():
    doc = make_document(storage_mode=StorageMode.ABSTRACT_ONLY, abstract="   ", full_text=None)
    assert chunk_document(doc, "test-model") == []


def test_short_text_becomes_one_chunk():
    doc = make_document(abstract="A short abstract.")
    chunks = chunk_document(doc, "test-model")
    assert len(chunks) == 1
    assert chunks[0].text == "A short abstract."


def test_long_text_is_split_and_full_text_wins_over_abstract():
    long_text = ("This sentence is repeated to make the document long. " * 200).strip()
    doc = make_document(storage_mode=StorageMode.FULL_TEXT, full_text=long_text, abstract="ignored")
    chunks = chunk_document(doc, "test-model")

    assert len(chunks) > 1
    assert "ignored" not in "".join(chunk.text for chunk in chunks)


def test_chunk_ids_are_predictable():
    """Stable ids mean re-indexing overwrites chunks instead of duplicating them."""
    doc = make_document(doc_id="fixed-id", abstract="Some text.")
    chunks = chunk_document(doc, "test-model")
    assert chunks[0].chunk_id == "fixed-id::0"
    assert chunks[0].chunk_index == 0


def test_chunks_carry_parent_details_for_citation():
    doc = make_document(title="Parent Title", version=3, abstract="Some text.")
    chunk = chunk_document(doc, "test-model")[0]

    assert chunk.title == "Parent Title"
    assert chunk.doc_id == doc.doc_id
    assert chunk.category == doc.category
    # Recorded so a later edit to the parent marks this chunk out of date.
    assert chunk.parent_version == 3


def test_context_numbers_passages_for_citation():
    context = build_context([_result("c1", "First Paper"), _result("c2", "Second Paper")])
    assert "[1] First Paper" in context
    assert "[2] Second Paper" in context


def test_context_stops_at_the_character_budget():
    """A small budget keeps the highest-ranked passages and drops the rest."""
    results = [_result(f"c{index}", f"Paper {index}") for index in range(20)]
    context = build_context(results, char_budget=200)
    assert len(context) < 600
    assert "[1] Paper 0" in context


def test_citation_map_matches_the_numbers_in_context():
    results = [_result("c1", "First Paper"), _result("c2", "Second Paper")]
    mapping = citation_map(results)
    assert mapping[1].title == "First Paper"
    assert mapping[2].title == "Second Paper"
