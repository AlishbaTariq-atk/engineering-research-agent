"""Measuring whether the system actually works.

Two things are measured. Retrieval quality asks whether search finds the
right material, and runs without any language model, so it is cheap to
repeat. Citation faithfulness asks whether the written answer is properly
grounded in what was retrieved, and needs full agent runs.

The query set below is built from documents known to be in the corpus. It
deliberately includes questions the corpus cannot answer, because a system
that fabricates an answer for those is worse than one that admits it does
not know.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from research_agent.config import Settings
from research_agent.models import SourceCategory
from research_agent.retrieval import Embedder, Reranker, SearchResult, search


@dataclass
class EvalQuery:
    """One test question and what a good answer to it looks like."""

    query: str
    expected_categories: list[SourceCategory] = field(default_factory=list)
    expected_title: str | None = None  # Substring of a document that should surface.
    known_gap: bool = False  # True when the corpus genuinely cannot answer this.
    notes: str = ""


QUERY_SET: list[EvalQuery] = [
    EvalQuery(
        query="What is the NIST AI Risk Management Framework?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS],
        expected_title="NIST AI Risk Management Framework",
    ),
    EvalQuery(
        query="What are the requirements of the EU AI Act for high-risk AI systems?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS],
        expected_title="EU Artificial Intelligence Act",
    ),
    EvalQuery(
        query="What executive order governs AI safety and security in the United States?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS],
        expected_title="Executive Order 14110",
    ),
    EvalQuery(
        query="What new features were added in recent vllm releases?",
        expected_categories=[SourceCategory.PRACTITIONER_KNOWLEDGE],
        expected_title="vllm",
    ),
    EvalQuery(
        query="How is speculative decoding implemented in vllm?",
        expected_categories=[SourceCategory.PRACTITIONER_KNOWLEDGE],
        expected_title="vllm",
    ),
    EvalQuery(
        query="What is LangChain and what is it used for?",
        expected_categories=[SourceCategory.PRACTITIONER_KNOWLEDGE],
        expected_title="langchain",
    ),
    EvalQuery(
        query="What research exists on molecular ensemble modeling of cyclic peptides?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        expected_title="Cyclic Peptides",
    ),
    EvalQuery(
        query="What does recent research say about sycophancy and moral reasoning in LLMs?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        expected_title="Sycophancy",
    ),
    EvalQuery(
        query="How do AI regulations like the EU AI Act relate to practical LLM deployment tools?",
        expected_categories=[SourceCategory.STANDARDS_REGULATIONS, SourceCategory.PRACTITIONER_KNOWLEDGE],
        notes="Spans two categories; checks that both are searched.",
    ),
    EvalQuery(
        query="What does the corpus say about gravure printing quality control automation?",
        expected_categories=[SourceCategory.TECHNICAL_LITERATURE],
        expected_title="gravure printing",
        notes="Deliberately obscure, so success cannot come from general AI/ML familiarity.",
    ),
    EvalQuery(
        query="What is the capital of France?",
        known_gap=True,
        notes="Nothing in a technical AI corpus should answer this.",
    ),
]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def category_precision(results: list[SearchResult], expected: list[SourceCategory]) -> float:
    """Fraction of results that came from an expected category.

    This is a coarse measure: it checks that search looked in the right
    part of the corpus, not that each individual passage is useful, since
    there are no per-passage relevance labels.

    Args:
        results: Retrieved passages.
        expected: Categories a good answer should come from.

    Returns:
        A value from 0.0 to 1.0. Returns 0.0 when nothing was expected or
        nothing was found.
    """
    if not expected or not results:
        return 0.0
    wanted = {category.value for category in expected}
    return sum(1 for result in results if result.category in wanted) / len(results)


def title_found(results: list[SearchResult], expected_title: str | None) -> bool:
    """Whether a specific known-relevant document appeared in the results.

    Args:
        results: Retrieved passages.
        expected_title: Part of the title that should appear, or None when
            the query has no single right answer.

    Returns:
        True if found, or if there was nothing to look for.
    """
    if not expected_title:
        return True
    return any(expected_title.lower() in result.title.lower() for result in results)


def citation_coverage(brief: dict) -> float:
    """Fraction of the brief's findings that cite at least one source.

    The synthesis prompt requires every claim to be supported. A value
    below 1.0 means the model asserted something with no evidence behind
    it, which this catches without anyone reading the report.

    Args:
        brief: A finished research brief.

    Returns:
        A value from 0.0 to 1.0.
    """
    findings = brief.get("key_findings", [])
    if not findings:
        return 0.0
    return sum(1 for finding in findings if finding.get("citations")) / len(findings)


def citation_validity(brief: dict) -> float:
    """Fraction of citations that point at a real retrieved source.

    Expected to be 1.0, because citations are resolved by lookup rather
    than generated. It is measured anyway, so a regression that broke that
    guarantee would show up rather than be assumed away.

    Args:
        brief: A finished research brief.

    Returns:
        A value from 0.0 to 1.0, or 1.0 if nothing was cited.
    """
    citation_lists = [finding.get("citations", []) for finding in brief.get("key_findings", [])]
    citation_lists += [conflict.get("citations", []) for conflict in brief.get("conflicts", [])]

    total = sum(len(citations) for citations in citation_lists)
    if total == 0:
        return 1.0
    resolved = sum(1 for citations in citation_lists for citation in citations if citation.get("url"))
    return resolved / total


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------


def evaluate_retrieval(settings: Settings) -> list[dict]:
    """Measure search quality across the query set. No language model needed.

    Each query is searched twice. The unrestricted search covers all
    categories and is what category precision is scored against — scoring
    it against a search already limited to the expected category would
    guarantee a perfect result no matter how bad the ranking was. The
    restricted search is used to check that a specific known document
    surfaces once the right category has been chosen.

    Args:
        settings: Application configuration.

    Returns:
        One row per query with its scores, the top result, and timing.
    """
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model)
    rows = []

    for query in QUERY_SET:
        started = time.perf_counter()
        unrestricted = search(query.query, settings, top_k=5, embedder=embedder, reranker=reranker)
        elapsed = time.perf_counter() - started

        restricted = search(
            query.query,
            settings,
            categories=query.expected_categories or None,
            top_k=5,
            embedder=embedder,
            reranker=reranker,
        )

        rows.append(
            {
                "query": query.query,
                "known_gap": query.known_gap,
                "checked_for_title": query.expected_title is not None,
                "category_precision": category_precision(unrestricted, query.expected_categories),
                "title_found": title_found(restricted, query.expected_title),
                "top_result": unrestricted[0].title if unrestricted else None,
                "top_result_category": unrestricted[0].category if unrestricted else None,
                "top_result_score": round(unrestricted[0].score, 3) if unrestricted else None,
                "seconds": round(elapsed, 2),
            }
        )

    return rows


def evaluate_agent(settings: Settings, max_rounds: int = 2) -> list[dict]:
    """Run the full agent over the query set and score its citations.

    Each query is isolated, so one failure — a rate-limited API, say — is
    recorded and the rest still run, rather than losing the whole batch.

    Args:
        settings: Application configuration.
        max_rounds: Search rounds allowed per question.

    Returns:
        One row per query, either with its scores or with the error that
        stopped it.
    """
    from research_agent.agent import answer_question

    rows = []
    for query in QUERY_SET:
        started = time.perf_counter()
        try:
            brief = answer_question(query.query, settings, max_rounds=max_rounds)
        except Exception as exc:
            rows.append(
                {
                    "query": query.query,
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            continue

        declined = brief.get("out_of_scope", False)
        rows.append(
            {
                "query": query.query,
                "known_gap": query.known_gap,
                "declined": declined,
                # Declining is the right outcome for a question the corpus
                # cannot answer, and the wrong one for anything else.
                "handled_correctly": declined == query.known_gap,
                "citation_coverage": citation_coverage(brief),
                "citation_validity": citation_validity(brief),
                "confidence": brief.get("confidence"),
                "findings": len(brief.get("key_findings", [])),
                "knowledge_gaps": len(brief.get("knowledge_gaps", [])),
                "seconds": round(time.perf_counter() - started, 2),
            }
        )

    return rows


def _mean(rows: list[dict], key: str) -> float | None:
    """Average a column, ignoring rows where it is missing."""
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 3) if values else None


def run_evaluation(settings: Settings, include_agent: bool = True) -> dict:
    """Run the evaluation suite and summarise the results.

    Args:
        settings: Application configuration.
        include_agent: Whether to run the agent pass, which calls the
            language model. Set False for a quick retrieval-only check.

    Returns:
        A summary plus the per-query detail behind it.
    """
    retrieval_rows = evaluate_retrieval(settings)
    agent_rows = evaluate_agent(settings) if include_agent else []

    answerable = [row for row in retrieval_rows if not row["known_gap"]]
    unanswerable = [row for row in retrieval_rows if row["known_gap"]]
    title_checks = [row for row in retrieval_rows if row["checked_for_title"]]

    summary = {
        "queries": len(QUERY_SET),
        "retrieval": {
            "category_precision_answerable": _mean(answerable, "category_precision"),
            "category_precision_known_gaps": _mean(unanswerable, "category_precision"),
            "title_found_rate": (
                round(sum(row["title_found"] for row in title_checks) / len(title_checks), 3)
                if title_checks
                else None
            ),
            "mean_seconds": _mean(retrieval_rows, "seconds"),
        },
    }

    if include_agent:
        finished = [row for row in agent_rows if "error" not in row]
        # Citation metrics only apply to questions that were answered. A
        # declined question has no findings by design, and scoring it as
        # zero coverage would penalise the system for behaving correctly.
        answered = [row for row in finished if not row["declined"]]

        summary["agent"] = {
            "completed": len(finished),
            "failed": len(agent_rows) - len(finished),
            "errors": [row["error"] for row in agent_rows if "error" in row],
            "answered": len(answered),
            "declined": sum(1 for row in finished if row["declined"]),
            "scope_decisions_correct": (
                round(sum(row["handled_correctly"] for row in finished) / len(finished), 3)
                if finished
                else None
            ),
            "citation_coverage": _mean(answered, "citation_coverage"),
            "citation_validity": _mean(answered, "citation_validity"),
            "mean_seconds": _mean(agent_rows, "seconds"),
        }

    return {"summary": summary, "retrieval_detail": retrieval_rows, "agent_detail": agent_rows}
