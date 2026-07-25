from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research_agent.config import Settings
from research_agent.evaluation.metrics import category_precision_at_k, citation_coverage, citation_validity, title_match_at_k
from research_agent.evaluation.query_set import QUERY_SET
from research_agent.retrieval import Embedder, Reranker
from research_agent.retrieval import search as retrieval_search


def evaluate_retrieval(settings: Settings) -> list[dict]:
    """Fast pass: retrieval quality only, no LLM calls, no agent decomposition.

    Runs each query TWICE, deliberately:
    - unrestricted (categories=None, searches all three collections): this
      is what category_precision_at_5 is scored against. Scoring it against
      a category-restricted search would be tautological - filtering to
      the expected category before scoring "is the result in the expected
      category" is guaranteed to hit 1.0 by construction regardless of
      whether ranking is any good. This was caught by noticing a
      suspiciously perfect first result, not assumed safe.
    - restricted to expected_categories: this is what title_match_at_k is
      scored against, since that's testing something different and real -
      given correct category routing, does semantic search actually
      surface the one specific known-relevant document, which a filter
      alone can't guarantee.
    """
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker()
    rows = []
    for eq in QUERY_SET:
        start = time.perf_counter()
        unrestricted = retrieval_search(eq.query, settings, categories=None, top_k=5, embedder=embedder, reranker=reranker)
        elapsed = time.perf_counter() - start

        restricted = retrieval_search(
            eq.query, settings, categories=eq.expected_categories or None, top_k=5, embedder=embedder, reranker=reranker
        )

        rows.append(
            {
                "query": eq.query,
                "known_gap": eq.known_gap,
                "has_title_check": eq.expected_title_substring is not None,
                "category_precision_at_5": category_precision_at_k(unrestricted, eq.expected_categories),
                "title_match": title_match_at_k(restricted, eq.expected_title_substring),
                "top_unrestricted_result_title": unrestricted[0].title if unrestricted else None,
                "top_unrestricted_result_category": unrestricted[0].category if unrestricted else None,
                "top_unrestricted_result_score": round(unrestricted[0].score, 3) if unrestricted else None,
                "latency_seconds": round(elapsed, 2),
            }
        )
    return rows


def evaluate_agent(settings: Settings, max_iterations: int = 2) -> list[dict]:
    """Slow pass: runs the full agent (real LLM calls) for citation
    coverage/validity and end-to-end latency. Kept as a separate pass from
    evaluate_retrieval so a quick retrieval-only check never requires a
    live LLM backend at all.

    Each query is wrapped in its own try/except: found via a real run that
    a single Groq rate-limit error (429, daily token cap) was an uncaught
    exception that crashed the entire batch, losing every result already
    computed. Same "one bad item can't take down the whole run" principle
    already applied to ingestion (pipeline.py's FetchFailure), just missed
    here until it actually happened."""
    from research_agent.agent import run_research_agent

    rows = []
    for eq in QUERY_SET:
        start = time.perf_counter()
        try:
            report = run_research_agent(eq.query, settings, max_iterations=max_iterations)
        except Exception as exc:
            rows.append(
                {
                    "query": eq.query,
                    "known_gap": eq.known_gap,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_seconds": round(time.perf_counter() - start, 2),
                }
            )
            continue
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "query": eq.query,
                "known_gap": eq.known_gap,
                "citation_coverage": citation_coverage(report),
                "citation_validity": citation_validity(report),
                "overall_confidence": report.get("confidence_assessment"),
                "num_key_findings": len(report.get("key_findings", [])),
                "num_knowledge_gaps": len(report.get("knowledge_gaps", [])),
                "latency_seconds": round(elapsed, 2),
            }
        )
    return rows


def _avg(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def run_full_evaluation(settings: Settings, include_agent: bool = True) -> dict:
    retrieval_rows = evaluate_retrieval(settings)
    agent_rows = evaluate_agent(settings) if include_agent else []

    non_gap = [r for r in retrieval_rows if not r["known_gap"]]
    gap = [r for r in retrieval_rows if r["known_gap"]]
    title_checked = [r for r in retrieval_rows if r["has_title_check"]]

    summary = {
        "num_queries": len(QUERY_SET),
        "retrieval": {
            "mean_category_precision_at_5_normal_queries": _avg(non_gap, "category_precision_at_5"),
            "mean_category_precision_at_5_known_gap_queries": _avg(gap, "category_precision_at_5"),
            "title_match_rate": (
                sum(r["title_match"] for r in title_checked) / len(title_checked) if title_checked else None
            ),
            "mean_retrieval_latency_seconds": _avg(retrieval_rows, "latency_seconds"),
        },
        "agent": (
            {
                "queries_completed": sum(1 for r in agent_rows if "error" not in r),
                "queries_failed": sum(1 for r in agent_rows if "error" in r),
                "failure_reasons": [r["error"] for r in agent_rows if "error" in r],
                "mean_citation_coverage": _avg(agent_rows, "citation_coverage"),
                "mean_citation_validity": _avg(agent_rows, "citation_validity"),
                "mean_end_to_end_latency_seconds": _avg(agent_rows, "latency_seconds"),
            }
            if include_agent
            else None
        ),
    }

    return {"summary": summary, "retrieval_detail": retrieval_rows, "agent_detail": agent_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-agent", action="store_true", help="Retrieval-only pass, no LLM calls/API cost")
    parser.add_argument("--output", default="data/eval_report.json")
    args = parser.parse_args()

    settings = Settings()
    result = run_full_evaluation(settings, include_agent=not args.skip_agent)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull detail written to {out_path}")


if __name__ == "__main__":
    main()
