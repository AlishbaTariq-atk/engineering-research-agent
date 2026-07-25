from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.models import SourceCategory


@dataclass
class EvalQuery:
    query: str
    expected_categories: list[SourceCategory]
    # Substring expected to appear in at least one top-k result's title -
    # only set where the corpus is known (by manual inspection, not
    # assumption) to contain a specific, clearly-matching document. Leave
    # empty for queries with no single "right answer" document.
    expected_title_substring: str | None = None
    # True only for queries deliberately chosen because the corpus has NO
    # good match in expected_categories (verified by inspecting actual
    # document titles, not guessed) - these test whether the system
    # honestly reports low confidence / a gap instead of forcing a
    # citation from the best-available-but-irrelevant candidate.
    known_gap: bool = False
    notes: str = ""


# Grounded in the actual corpus content (research_agent/ingestion/sources/
# standards_documents.json, the github_repos list in config.py, and a live
# inspection of the 15 arXiv papers actually ingested - not invented blind).
QUERY_SET: list[EvalQuery] = [
    EvalQuery(
        query="What is the NIST AI Risk Management Framework?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS],
        expected_title_substring="NIST AI Risk Management Framework",
        notes="Positive control: exact document exists in the curated standards list.",
    ),
    EvalQuery(
        query="What are the requirements of the EU AI Act for high-risk AI systems?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS],
        expected_title_substring="EU Artificial Intelligence Act",
        notes="Positive control.",
    ),
    EvalQuery(
        query="What executive order governs AI safety and security in the United States?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS],
        expected_title_substring="Executive Order 14110",
        notes="Positive control.",
    ),
    EvalQuery(
        query="What new features were added in recent vllm releases?",
        expected_categories=[SourceCategory.PRACTITIONER_KNOWLEDGE],
        expected_title_substring="vllm",
        notes="Positive control: vllm has ~150+ ingested releases.",
    ),
    EvalQuery(
        query="How is speculative decoding implemented in vllm?",
        expected_categories=[SourceCategory.PRACTITIONER_KNOWLEDGE],
        expected_title_substring="vllm",
        notes="Positive control: vllm release notes explicitly mention speculative decoding "
        "(confirmed via live agent test citing v0.5.4 correctly).",
    ),
    EvalQuery(
        query="What is LangChain and what is it used for?",
        expected_categories=[SourceCategory.PRACTITIONER_KNOWLEDGE],
        expected_title_substring="langchain",
        notes="Positive control: README ingested.",
    ),
    EvalQuery(
        query="What research exists on molecular ensemble modeling of cyclic peptides?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        expected_title_substring="Cyclic Peptides",
        notes="Positive control: exact arXiv paper title confirmed present.",
    ),
    EvalQuery(
        query="What does recent research say about sycophancy and moral reasoning in LLMs?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        expected_title_substring="Sycophancy",
        notes="Positive control: exact arXiv paper title confirmed present.",
    ),
    EvalQuery(
        query="What does recent academic research say about speculative decoding for LLM inference?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        known_gap=True,
        notes="Known gap: manually confirmed none of the 15 ingested arXiv papers are about "
        "inference optimization/speculative decoding - this is what actually produced the "
        "false-positive VLM-paper citation seen in live agent testing. Tests whether the "
        "system reports low confidence / a knowledge gap rather than forcing an irrelevant citation.",
    ),
    EvalQuery(
        query="What does the corpus say about gravure printing quality control automation?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        expected_title_substring="gravure printing",
        notes="Positive control on an intentionally narrow/unusual topic, to check retrieval "
        "isn't just succeeding on generic AI/ML queries.",
    ),
    EvalQuery(
        query="How do AI regulations like the EU AI Act relate to practical LLM deployment tools?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS, SourceCategory.PRACTITIONER_KNOWLEDGE],
        notes="Cross-category: tests whether a query spanning two categories actually retrieves "
        "from both rather than collapsing onto one.",
    ),
    EvalQuery(
        query="What is the capital of France?",
        expected_categories=[],
        known_gap=True,
        notes="Out-of-domain control: nothing in an AI/ML knowledge base should answer this "
        "confidently. Tests that the system doesn't fabricate relevance for a query with no "
        "legitimate match in the corpus at all.",
    ),
]
