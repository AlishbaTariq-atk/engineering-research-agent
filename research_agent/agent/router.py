from __future__ import annotations

from pydantic import BaseModel, Field

from research_agent.agent.llm import get_llm
from research_agent.config import Settings
from research_agent.models import SourceCategory

ROUTER_SYSTEM_PROMPT = """You route research sub-questions to the knowledge \
base categories most likely to answer each one:
- technical_literature: academic papers (arXiv) - methods, benchmarks, theory
- standards_regulations: government/standards bodies - policy, compliance, risk frameworks
- practitioner_knowledge: GitHub releases/READMEs, engineering blogs - real-world implementation, tooling, changelogs

A sub-question can route to more than one category if genuinely relevant to \
both. Do not include a category just to be safe - unnecessary categories \
waste retrieval budget on irrelevant results.

You will be given a numbered list of sub-questions. Return one routing entry \
per sub-question, in the same order, each carrying the original question text \
back so the caller can match them up unambiguously."""


class RoutingEntry(BaseModel):
    question: str
    categories: list[str] = Field(
        description="subset of: technical_literature, standards_regulations, practitioner_knowledge"
    )


class RoutingOutput(BaseModel):
    routes: list[RoutingEntry]


def route_sub_questions(sub_questions: list[str], settings: Settings) -> dict[str, list[SourceCategory]]:
    """Second reasoning step: decide which categories are worth searching
    for each sub-question, so retrieval stays category-aware at the query
    level, not just at the index level (see indexer.py).

    Batched into one LLM call for all sub-questions rather than one call
    per sub-question: found via live testing that the per-question version
    pushed total LLM calls per agent run into the 8-10 range, risking
    Groq's free-tier rate limits (observed as an unexplained 138-second
    stall, almost certainly a silent retry-after-429) for no real benefit -
    routing every sub-question is one classification task, not several.
    """
    if not sub_questions:
        return {}
    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(RoutingOutput)
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, start=1))
    result: RoutingOutput = structured_llm.invoke([("system", ROUTER_SYSTEM_PROMPT), ("human", numbered)])

    # Matched by position, not by the echoed-back question text: the model
    # can reword/trim what it echoes, which would break a text-keyed match
    # even when it answered correctly. Position is exactly what "same
    # order" in the prompt asked for.
    valid = {c.value for c in SourceCategory}
    routing: dict[str, list[SourceCategory]] = {}
    for i, question in enumerate(sub_questions):
        raw_categories = result.routes[i].categories if i < len(result.routes) else []
        categories = [SourceCategory(c) for c in raw_categories if c in valid]
        # Fail open (search everything) for any sub-question whose entry is
        # missing, empty, or unparseable - never fail into silent zero results.
        routing[question] = categories or list(SourceCategory)
    return routing
