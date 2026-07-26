from __future__ import annotations

from research_agent.evaluation.metrics import (
    category_precision_at_k,
    citation_coverage,
    citation_validity,
    title_match_at_k,
)
from research_agent.models import SourceCategory
from research_agent.retrieval.types import RetrievedChunk


def _chunk(title: str, category: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c", doc_id="d", text="t", title=title, canonical_url="https://x",
        source="arxiv", category=category, publication_date=None, score=1.0, reranked=True,
    )


def test_category_precision_all_match():
    results = [_chunk("A", "technical_literature"), _chunk("B", "technical_literature")]
    assert category_precision_at_k(results, [SourceCategory.TECHNICAL_LITERATURE]) == 1.0


def test_category_precision_partial_match():
    results = [_chunk("A", "technical_literature"), _chunk("B", "practitioner_knowledge")]
    assert category_precision_at_k(results, [SourceCategory.TECHNICAL_LITERATURE]) == 0.5


def test_category_precision_empty_expected_is_zero_not_undefined():
    """A query with no expected category (e.g. a fully out-of-domain
    control) should score 0, not divide-by-zero or vacuously pass."""
    results = [_chunk("A", "technical_literature")]
    assert category_precision_at_k(results, []) == 0.0


def test_title_match_finds_expected_substring_anywhere_in_top_k():
    results = [_chunk("Unrelated Paper", "x"), _chunk("The Real Paper About Cyclic Peptides", "x")]
    assert title_match_at_k(results, "Cyclic Peptides") is True


def test_title_match_false_when_absent():
    results = [_chunk("Unrelated Paper", "x")]
    assert title_match_at_k(results, "Cyclic Peptides") is False


def test_title_match_vacuously_true_when_nothing_to_check():
    assert title_match_at_k([_chunk("Anything", "x")], None) is True


def test_citation_coverage_all_findings_cited():
    report = {"key_findings": [{"citations": [{"n": 1}]}, {"citations": [{"n": 2}]}]}
    assert citation_coverage(report) == 1.0


def test_citation_coverage_catches_uncited_finding():
    """Directly operationalizes the real bug found in live testing: a
    finding with citations: [] despite the synthesis prompt forbidding
    unsupported claims."""
    report = {"key_findings": [{"citations": [{"n": 1}]}, {"citations": []}]}
    assert citation_coverage(report) == 0.5


def test_citation_coverage_no_findings_is_zero():
    assert citation_coverage({"key_findings": []}) == 0.0


def test_citation_validity_all_resolved():
    report = {"key_findings": [{"citations": [{"n": 1, "url": "https://x"}]}], "conflicts": []}
    assert citation_validity(report) == 1.0


def test_citation_validity_nothing_cited_is_vacuously_valid():
    report = {"key_findings": [{"citations": []}], "conflicts": []}
    assert citation_validity(report) == 1.0
