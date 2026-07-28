"""Tests for assembling the final brief from the model's answer.

These cover the step that turns cited passage numbers into real sources,
which is what keeps citations honest.
"""

from __future__ import annotations

from research_agent.agent.nodes import Conflict, Finding, Synthesis, build_brief, build_out_of_scope_brief
from research_agent.retrieval.search import SearchResult


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


def _synthesis(**overrides) -> Synthesis:
    """Build a model answer, overriding only what a test cares about."""
    fields = {
        "executive_summary": "A summary.",
        "key_findings": [Finding(claim="A claim.", citation_ids=[1], confidence="high")],
        "conflicts": [],
        "overall_confidence": "high",
        "knowledge_gaps": ["a gap"],
        "follow_up_questions": ["a follow-up?"],
    }
    fields.update(overrides)
    return Synthesis(**fields)


def test_citations_resolve_to_real_retrieved_sources():
    """Titles and URLs come from the passages that were actually retrieved."""
    citations = {1: _result("c1", "Real Paper Title")}
    brief = build_brief("A question?", _synthesis(), citations)

    cited = brief["key_findings"][0]["citations"]
    assert len(cited) == 1
    assert cited[0]["title"] == "Real Paper Title"
    assert cited[0]["url"] == "https://example.com/c1"


def test_unknown_citation_numbers_are_dropped():
    """A number the model invented has no source behind it, so it is discarded
    rather than turned into a citation with made-up details."""
    citations = {1: _result("c1", "Real Paper")}
    synthesis = _synthesis(key_findings=[Finding(claim="A claim.", citation_ids=[1, 99], confidence="high")])

    brief = build_brief("A question?", synthesis, citations)

    cited = brief["key_findings"][0]["citations"]
    assert len(cited) == 1
    assert cited[0]["title"] == "Real Paper"


def test_uncited_findings_are_kept_as_they_are():
    """A claim with no evidence is a real problem, so the brief shows it
    plainly instead of hiding the finding."""
    citations = {1: _result("c1", "Real Paper")}
    synthesis = _synthesis(key_findings=[Finding(claim="Unsupported claim.", citation_ids=[], confidence="low")])

    brief = build_brief("A question?", synthesis, citations)

    assert brief["key_findings"][0]["citations"] == []
    assert brief["key_findings"][0]["claim"] == "Unsupported claim."


def test_source_list_covers_findings_and_conflicts_without_repeats():
    citations = {1: _result("c1", "Paper One"), 2: _result("c2", "Paper Two")}
    synthesis = _synthesis(
        key_findings=[Finding(claim="A claim.", citation_ids=[1], confidence="high")],
        conflicts=[Conflict(description="They disagree.", citation_ids=[1, 2])],
    )

    brief = build_brief("A question?", synthesis, citations)

    assert {citation["n"] for citation in brief["citations"]} == {1, 2}


def test_brief_contains_every_expected_section():
    brief = build_brief("A question?", _synthesis(), {1: _result("c1", "Paper")})

    for section in [
        "question",
        "executive_summary",
        "key_findings",
        "conflicts",
        "citations",
        "confidence",
        "knowledge_gaps",
        "follow_up_questions",
    ]:
        assert section in brief

    assert brief["out_of_scope"] is False


def test_out_of_scope_brief_makes_no_claims():
    """A declined question must produce no findings and no citations, so
    nothing irrelevant can be read as an answer."""
    nearest = [_result("c1", "An Unrelated Paper", score=-8.8)]
    brief = build_out_of_scope_brief(
        "What is the capital of France?", nearest, ["This evidence covers AI regulation, not geography."]
    )

    assert brief["out_of_scope"] is True
    assert brief["key_findings"] == []
    assert brief["citations"] == []
    assert brief["confidence"] == "none"


def test_out_of_scope_brief_shows_what_was_actually_found():
    """The nearest matches and the reviewer's reasoning are both reported,
    so the refusal is explainable rather than a bare no."""
    nearest = [_result("c1", "An Unrelated Paper", score=-8.8)]
    reasons = ["This evidence is about something else entirely."]
    brief = build_out_of_scope_brief("Something unrelated?", nearest, reasons)

    assert brief["nearest_matches"][0]["title"] == "An Unrelated Paper"
    assert brief["nearest_matches"][0]["score"] == -8.8
    assert brief["knowledge_gaps"] == reasons


def test_out_of_scope_brief_falls_back_when_no_reason_is_given():
    brief = build_out_of_scope_brief("A question?", [], [])
    assert brief["knowledge_gaps"]


def test_out_of_scope_brief_has_the_same_shape_as_a_normal_one():
    """Both kinds of brief share a shape, so printing and scoring need no
    special cases beyond the out_of_scope flag itself."""
    answered = build_brief("A question?", _synthesis(), {1: _result("c1", "Paper")})
    declined = build_out_of_scope_brief("Another question?", [], ["reason"])

    assert set(answered).issubset(set(declined))


def test_out_of_scope_brief_handles_no_results_at_all():
    brief = build_out_of_scope_brief("A question?", [], ["reason"])

    assert brief["out_of_scope"] is True
    assert brief["nearest_matches"] == []
    assert brief["knowledge_gaps"]
