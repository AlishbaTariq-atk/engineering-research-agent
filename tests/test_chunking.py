from __future__ import annotations

from research_agent.models import StorageMode
from research_agent.retrieval.chunking import chunk_document

from .conftest import make_document


def test_metadata_only_document_produces_no_chunks():
    """METADATA_ONLY documents have no real text - chunking/embedding them
    would be indexing noise, not evidence (see StorageMode docstring)."""
    doc = make_document(storage_mode=StorageMode.METADATA_ONLY, abstract=None, full_text=None)
    assert chunk_document(doc, "test-model") == []


def test_short_abstract_produces_one_chunk():
    doc = make_document(storage_mode=StorageMode.ABSTRACT_ONLY, abstract="A short abstract.")
    chunks = chunk_document(doc, "test-model")
    assert len(chunks) == 1
    assert chunks[0].text == "A short abstract."


def test_long_full_text_is_split_into_multiple_overlapping_chunks():
    long_text = ("Speculative decoding reduces inference latency. " * 200).strip()
    doc = make_document(storage_mode=StorageMode.FULL_TEXT, full_text=long_text, abstract="short")
    chunks = chunk_document(doc, "test-model")

    assert len(chunks) > 1
    # full_text takes priority over abstract when both are present
    assert "Speculative decoding" in chunks[0].text
    assert "short" not in "".join(c.text for c in chunks)


def test_chunk_ids_are_deterministic_and_ordered():
    doc = make_document(doc_id="fixed-id", abstract="A short abstract.")
    chunks = chunk_document(doc, "test-model")
    assert chunks[0].chunk_id == "fixed-id::0"
    assert chunks[0].chunk_index == 0


def test_chunks_inherit_parent_fields():
    doc = make_document(title="Parent Title", version=3, abstract="Some text here.")
    chunks = chunk_document(doc, "test-model")
    assert chunks[0].title == "Parent Title"
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].parent_version == 3  # so a later version bump can be detected as stale
    assert chunks[0].category == doc.category
    assert chunks[0].source == doc.source


def test_empty_text_produces_no_chunks():
    doc = make_document(storage_mode=StorageMode.ABSTRACT_ONLY, abstract="   ", full_text=None)
    assert chunk_document(doc, "test-model") == []
