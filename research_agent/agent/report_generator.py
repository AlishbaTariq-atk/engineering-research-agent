from __future__ import annotations

from research_agent.agent.synthesizer import SynthesisOutput
from research_agent.retrieval.types import RetrievedChunk


def build_research_brief(
    question: str,
    synthesis: SynthesisOutput,
    citations: dict[int, RetrievedChunk],
) -> dict:
    """Deterministic assembly, not an LLM call: resolves the [n] citation
    numbers the synthesizer used into real title/url/source/date records
    from `citations` (built by retrieval.citation_map from the actual
    retrieved chunks). An LLM asked to also invent the citation list from
    scratch could hallucinate a URL; this cannot - it can only look one up
    in `citations` or drop it if the id doesn't exist."""

    def resolve(ids: list[int]) -> list[dict]:
        return [
            {
                "n": i,
                "title": citations[i].title,
                "url": citations[i].canonical_url,
                "source": citations[i].source,
                "publication_date": citations[i].publication_date,
            }
            for i in ids
            if i in citations
        ]

    all_cited_ids = sorted(
        {i for f in synthesis.key_findings for i in f.citation_ids}
        | {i for c in synthesis.conflicts for i in c.citation_ids}
    )

    return {
        "question": question,
        "executive_summary": synthesis.executive_summary,
        "key_findings": [
            {"claim": f.claim, "confidence": f.confidence, "citations": resolve(f.citation_ids)}
            for f in synthesis.key_findings
        ],
        "conflicts": [
            {"description": c.description, "citations": resolve(c.citation_ids)} for c in synthesis.conflicts
        ],
        "citations": resolve(all_cited_ids),
        "confidence_assessment": synthesis.overall_confidence,
        "knowledge_gaps": synthesis.knowledge_gaps,
        "follow_up_questions": synthesis.follow_up_questions,
    }
