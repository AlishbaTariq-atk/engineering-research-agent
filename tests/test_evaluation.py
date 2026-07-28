"""Tests for the evaluation metrics."""

from __future__ import annotations

from research_agent.evaluation import category_precision, citation_coverage, citation_validity, title_found
from research_agent.models import SourceCategory
from research_agent.retrieval.search import SearchResult


def _result(title: str, category: str) -> SearchResult:
    """Build a search result for tests."""
    return SearchResult(
        chunk_id="c",
        doc_id="d",
        text="text",
        title=title,
        canonical_url="https://example.com",
        source="arxiv",
        category=category,
        publication_date=None,
        score=1.0,
    )


def test_category_precision_when_everything_matches():
    results = [_result("A", "technical_literature"), _result("B", "technical_literature")]
    assert category_precision(results, [SourceCategory.TECHNICAL_LITERATURE]) == 1.0


def test_category_precision_when_some_results_are_off_target():
    results = [_result("A", "technical_literature"), _result("B", "practitioner_knowledge")]
    assert category_precision(results, [SourceCategory.TECHNICAL_LITERATURE]) == 0.5


def test_category_precision_is_zero_when_nothing_was_expected():
    """Out-of-domain queries expect no category, and must not score as passing."""
    assert category_precision([_result("A", "technical_literature")], []) == 0.0


def test_title_found_anywhere_in_results():
    results = [_result("Unrelated Paper", "x"), _result("A Paper About Cyclic Peptides", "x")]
    assert title_found(results, "Cyclic Peptides") is True


def test_title_not_found():
    assert title_found([_result("Unrelated Paper", "x")], "Cyclic Peptides") is False


def test_title_check_passes_when_there_is_nothing_to_look_for():
    assert title_found([_result("Anything", "x")], None) is True


def test_citation_coverage_when_every_finding_is_supported():
    brief = {"key_findings": [{"citations": [{"n": 1}]}, {"citations": [{"n": 2}]}]}
    assert citation_coverage(brief) == 1.0


def test_citation_coverage_flags_an_unsupported_claim():
    """A finding with no citations breaks the rule that claims must be
    grounded, and the metric has to surface it."""
    brief = {"key_findings": [{"citations": [{"n": 1}]}, {"citations": []}]}
    assert citation_coverage(brief) == 0.5


def test_citation_coverage_with_no_findings():
    assert citation_coverage({"key_findings": []}) == 0.0


def test_citation_validity_when_all_citations_resolve():
    brief = {"key_findings": [{"citations": [{"n": 1, "url": "https://example.com"}]}], "conflicts": []}
    assert citation_validity(brief) == 1.0


def test_citation_validity_with_nothing_cited():
    assert citation_validity({"key_findings": [{"citations": []}], "conflicts": []}) == 1.0
