from __future__ import annotations

from datetime import UTC, datetime

from research_agent.agent.report_generator import build_research_brief
from research_agent.agent.synthesizer import Conflict, Finding, SynthesisOutput
from research_agent.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text="some retrieved text",
        title=title,
        canonical_url=f"https://example.com/{chunk_id}",
        source="arxiv",
        category="technical_literature",
        publication_date="2024-01-01",
        score=1.0,
        reranked=True,
    )


def _synthesis(**overrides) -> SynthesisOutput:
    defaults = dict(
        executive_summary="Summary.",
        key_findings=[Finding(claim="A claim.", citation_ids=[1], confidence="high")],
        conflicts=[],
        overall_confidence="high",
        knowledge_gaps=["some gap"],
        follow_up_questions=["a follow up?"],
    )
    defaults.update(overrides)
    return SynthesisOutput(**defaults)


def test_citations_resolve_real_retrieved_chunks_only():
    """The hallucination-resistance guarantee: report_generator can only
    populate a citation from citation_map (built from actually-retrieved
    chunks) - it has no other source of titles/URLs to draw from."""
    citation_map = {1: _chunk("c1", "Real Paper Title")}
    synthesis = _synthesis(key_findings=[Finding(claim="X is true.", citation_ids=[1], confidence="high")])

    report = build_research_brief("question?", synthesis, citation_map)

    assert len(report["key_findings"][0]["citations"]) == 1
    cited = report["key_findings"][0]["citations"][0]
    assert cited["title"] == "Real Paper Title"
    assert cited["url"] == "https://example.com/c1"


def test_citation_id_not_in_map_is_dropped_not_fabricated():
    """If the model hallucinates a citation number that doesn't correspond
    to any retrieved chunk, it must be silently dropped, never invented
    with a placeholder title/URL."""
    citation_map = {1: _chunk("c1", "Real Paper")}
    synthesis = _synthesis(key_findings=[Finding(claim="X is true.", citation_ids=[1, 99], confidence="high")])

    report = build_research_brief("question?", synthesis, citation_map)

    assert len(report["key_findings"][0]["citations"]) == 1
    assert report["key_findings"][0]["citations"][0]["title"] == "Real Paper"


def test_finding_with_no_citations_is_preserved_not_hidden():
    """An uncited finding is a real faithfulness problem (see
    evaluation/metrics.py:citation_coverage) - the report builder's job is
    to represent it accurately, not paper over it."""
    citation_map = {1: _chunk("c1", "Real Paper")}
    synthesis = _synthesis(key_findings=[Finding(claim="Unsupported claim.", citation_ids=[], confidence="low")])

    report = build_research_brief("question?", synthesis, citation_map)

    assert report["key_findings"][0]["citations"] == []
    assert report["key_findings"][0]["claim"] == "Unsupported claim."


def test_top_level_citations_list_is_deduplicated_union():
    citation_map = {1: _chunk("c1", "Paper One"), 2: _chunk("c2", "Paper Two")}
    synthesis = _synthesis(
        key_findings=[Finding(claim="A.", citation_ids=[1], confidence="high")],
        conflicts=[Conflict(description="disagreement", citation_ids=[1, 2])],
    )

    report = build_research_brief("question?", synthesis, citation_map)

    assert {c["n"] for c in report["citations"]} == {1, 2}
