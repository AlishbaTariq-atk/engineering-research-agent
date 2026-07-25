from __future__ import annotations

from research_agent.models import SourceCategory
from research_agent.retrieval.types import RetrievedChunk


def category_precision_at_k(results: list[RetrievedChunk], expected_categories: list[SourceCategory]) -> float:
    """Fraction of the top-k results whose category is one of the expected
    categories. A coarse but honest metric: we don't have exhaustive
    per-document relevance judgments for the corpus, so this measures
    "did retrieval stay in the right part of the index" rather than
    per-document relevance - a real floor, not a substitute for human
    judgment at larger scale."""
    if not expected_categories or not results:
        return 0.0
    expected = {c.value for c in expected_categories}
    hits = sum(1 for r in results if r.category in expected)
    return hits / len(results)


def title_match_at_k(results: list[RetrievedChunk], expected_title_substring: str | None) -> bool:
    """True if a result whose title contains the expected substring
    appears anywhere in the top-k. Only meaningful for EvalQuery entries
    that set expected_title_substring (a specific, manually-confirmed
    document known to exist in the corpus) - returns True (vacuously) when
    no such document was specified, since there's nothing to check."""
    if not expected_title_substring:
        return True
    needle = expected_title_substring.lower()
    return any(needle in r.title.lower() for r in results)


def citation_coverage(report: dict) -> float:
    """Fraction of key_findings that carry at least one citation. Directly
    operationalizes a real bug found via live testing: the synthesizer
    stated a claim ("speculative decoding may compromise model/data
    integrity") with citations: [] despite an explicit system-prompt
    instruction not to state anything unsupported. A citation_coverage < 1
    on a given report is exactly that failure mode, made measurable."""
    findings = report.get("key_findings", [])
    if not findings:
        return 0.0
    covered = sum(1 for f in findings if f.get("citations"))
    return covered / len(findings)


def citation_validity(report: dict) -> float:
    """Fraction of citations across all findings/conflicts that resolved
    to a real title/url (i.e. report_generator.py's citation_map lookup
    succeeded). Expected to be 1.0 by construction - report_generator can
    only look citation_ids up in the map built from actually-retrieved
    chunks, never invent one. Reported anyway: a value below 1.0 would
    mean that guarantee broke, which is worth knowing immediately, not
    assuming still holds."""
    findings = report.get("key_findings", [])
    conflicts = report.get("conflicts", [])
    all_citation_lists = [f.get("citations", []) for f in findings] + [c.get("citations", []) for c in conflicts]
    total_ids = sum(len(c) for c in all_citation_lists)
    if total_ids == 0:
        return 1.0  # nothing cited, nothing to invalidate
    valid = sum(1 for citations in all_citation_lists for c in citations if c.get("url"))
    return valid / total_ids
