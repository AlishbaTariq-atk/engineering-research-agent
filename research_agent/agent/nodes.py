"""The individual reasoning steps the agent runs.

Each function here does one job and returns plain data. `graph.py` decides
the order they run in and when to loop. Every step that calls the language
model asks for a typed result, so a malformed reply fails immediately
rather than flowing downstream as a broken dict.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from research_agent.config import Settings
from research_agent.models import SourceCategory
from research_agent.retrieval import SearchResult

from . import prompts


def get_llm(settings: Settings, temperature: float = 0.1) -> BaseChatModel:
    """Build the language model client named in configuration.

    Args:
        settings: Application configuration, including which provider to use.
        temperature: Sampling temperature. Low by default, since these
            steps are closer to extraction than to open writing.

    Returns:
        A chat model. Groq is a hosted API; Ollama runs locally.

    Raises:
        ValueError: If the configured provider is not recognised.
    """
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=temperature)

    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, temperature=temperature)

    raise ValueError(f"Unknown llm_provider {settings.llm_provider!r}; expected 'groq' or 'ollama'.")


# --------------------------------------------------------------------------
# Typed model outputs
# --------------------------------------------------------------------------


class Plan(BaseModel):
    """The sub-questions a research question was broken into."""

    sub_questions: list[str] = Field(min_length=1, max_length=4)


class Route(BaseModel):
    """Which categories should be searched for one sub-question."""

    question: str
    categories: list[str] = Field(
        description="Any of: technical_literature, standards_regulations, practitioner_knowledge"
    )


class Routes(BaseModel):
    """Routing decisions for a whole round of sub-questions."""

    routes: list[Route]


class GapAssessment(BaseModel):
    """Whether the evidence is relevant, whether it is enough, and what is missing."""

    answerable: bool = Field(description="True if any evidence actually bears on the question")
    sufficient: bool = Field(description="True if no further searching is needed")
    gaps: list[str] = Field(min_length=1, description="Missing angles or limitations; never empty")


class Finding(BaseModel):
    """One claim, the passages backing it, and how confident it is."""

    claim: str
    citation_ids: list[int] = Field(description="Numbers of the passages supporting this claim")
    confidence: str = Field(description="high, medium, or low")


class Conflict(BaseModel):
    """A disagreement between sources."""

    description: str
    citation_ids: list[int] = Field(description="Numbers of the passages that disagree")


class Synthesis(BaseModel):
    """The answer the model produced from the gathered evidence."""

    executive_summary: str
    key_findings: list[Finding]
    conflicts: list[Conflict] = Field(default_factory=list)
    overall_confidence: str = Field(description="high, medium, or low")
    knowledge_gaps: list[str] = Field(min_length=1)
    follow_up_questions: list[str] = Field(min_length=1, max_length=5)


# --------------------------------------------------------------------------
# Reasoning steps
# --------------------------------------------------------------------------


def plan(question: str, settings: Settings) -> list[str]:
    """Break a research question into focused sub-questions.

    Args:
        question: The user's research question.
        settings: Application configuration.

    Returns:
        Two to four sub-questions, each searchable on its own.
    """
    llm = get_llm(settings).with_structured_output(Plan)
    return llm.invoke([("system", prompts.PLANNER), ("human", question)]).sub_questions


def route(sub_questions: list[str], settings: Settings) -> dict[str, list[SourceCategory]]:
    """Choose which categories to search for each sub-question.

    All sub-questions are routed in a single model call rather than one
    call each, so the number of requests per run stays flat no matter how
    many sub-questions were planned.

    Args:
        sub_questions: The sub-questions to route.
        settings: Application configuration.

    Returns:
        Each sub-question mapped to the categories worth searching. A
        sub-question the model leaves out, or answers unusably, falls back
        to all categories rather than returning nothing.
    """
    if not sub_questions:
        return {}

    llm = get_llm(settings).with_structured_output(Routes)
    numbered = "\n".join(f"{index}. {question}" for index, question in enumerate(sub_questions, start=1))
    reply = llm.invoke([("system", prompts.ROUTER), ("human", numbered)])

    # Matched by position rather than by the text the model echoes back,
    # since it may reword a question slightly even when routing it correctly.
    valid = {category.value for category in SourceCategory}
    routing: dict[str, list[SourceCategory]] = {}
    for index, question in enumerate(sub_questions):
        raw = reply.routes[index].categories if index < len(reply.routes) else []
        chosen = [SourceCategory(name) for name in raw if name in valid]
        routing[question] = chosen or list(SourceCategory)

    return routing


def assess_gaps(question: str, sub_questions: list[str], context: str, settings: Settings) -> GapAssessment:
    """Judge whether the gathered evidence is relevant, and whether it suffices.

    Two decisions come from this one call. Relevance decides whether the
    knowledge base can address the question at all; sufficiency drives the
    search loop, since gaps reported here become the next round's queries.

    Relevance is judged by the model rather than by a search score. Scores
    from the reranker rank passages within a single query and are not
    comparable between queries: a short, broad question scores low against
    long technical passages whatever the subject, so a numeric cutoff
    rejects valid questions as readily as invalid ones.

    Args:
        question: The original research question.
        sub_questions: The sub-questions searched so far.
        context: The evidence gathered so far, as formatted passages.
        settings: Application configuration.

    Returns:
        Whether the evidence is relevant, whether it is sufficient, and
        what is still missing.
    """
    llm = get_llm(settings).with_structured_output(GapAssessment)
    user_message = (
        f"Research question: {question}\n\nSub-questions:\n"
        + "\n".join(f"- {sub_question}" for sub_question in sub_questions)
        + f"\n\nEvidence gathered so far:\n{context}"
    )
    return llm.invoke([("system", prompts.CRITIC), ("human", user_message)])


def synthesise(question: str, sub_questions: list[str], context: str, settings: Settings) -> Synthesis:
    """Write the answer from the gathered evidence.

    All evidence is considered in one pass rather than per sub-question, so
    the model can notice when two sources disagree with each other.

    Args:
        question: The original research question.
        sub_questions: The sub-questions that were searched.
        context: All gathered evidence, as formatted passages.
        settings: Application configuration.

    Returns:
        The summary, findings with citations, conflicts, confidence,
        knowledge gaps, and follow-up questions.
    """
    llm = get_llm(settings).with_structured_output(Synthesis)
    user_message = (
        f"Research question: {question}\n\nSub-questions considered:\n"
        + "\n".join(f"- {sub_question}" for sub_question in sub_questions)
        + f"\n\nEvidence:\n{context}"
    )
    return llm.invoke([("system", prompts.SYNTHESIZER), ("human", user_message)])


def build_brief(question: str, synthesis: Synthesis, citations: dict[int, SearchResult]) -> dict:
    """Assemble the final report, resolving citation numbers to real sources.

    This step is deliberately ordinary Python rather than another model
    call. Titles and URLs are looked up from the passages that were
    actually retrieved, so a citation can only be real or dropped — the
    model has no opportunity to invent one.

    Args:
        question: The original research question.
        synthesis: The model's answer, citing passages by number.
        citations: Passage numbers mapped to the passages themselves.

    Returns:
        The finished research brief.
    """

    def resolve(numbers: list[int]) -> list[dict]:
        """Turn passage numbers into source records, ignoring unknown numbers."""
        return [
            {
                "n": number,
                "title": citations[number].title,
                "url": citations[number].canonical_url,
                "source": citations[number].source,
                "publication_date": citations[number].publication_date,
            }
            for number in numbers
            if number in citations
        ]

    cited_anywhere = sorted(
        {number for finding in synthesis.key_findings for number in finding.citation_ids}
        | {number for conflict in synthesis.conflicts for number in conflict.citation_ids}
    )

    return {
        "question": question,
        "executive_summary": synthesis.executive_summary,
        "key_findings": [
            {
                "claim": finding.claim,
                "confidence": finding.confidence,
                "citations": resolve(finding.citation_ids),
            }
            for finding in synthesis.key_findings
        ],
        "conflicts": [
            {"description": conflict.description, "citations": resolve(conflict.citation_ids)}
            for conflict in synthesis.conflicts
        ],
        "citations": resolve(cited_anywhere),
        "confidence": synthesis.overall_confidence,
        "knowledge_gaps": synthesis.knowledge_gaps,
        "follow_up_questions": synthesis.follow_up_questions,
        "out_of_scope": False,
    }


def build_out_of_scope_brief(question: str, nearest: list[SearchResult], reasons: list[str]) -> dict:
    """Report that the knowledge base cannot answer a question.

    Reached when the reviewer judged that none of the retrieved material
    bears on the question. Answering from those passages would produce
    confident text supported by irrelevant citations, so the system
    declines instead.

    The closest passages found are still listed, so the reader can see what
    the search actually turned up rather than getting a bare refusal.

    Args:
        question: The question that was asked.
        nearest: The best passages found, none of them relevant.
        reasons: The reviewer's explanation of what the evidence covers instead.

    Returns:
        A brief in the usual shape, with no findings and `out_of_scope` set.
    """
    return {
        "question": question,
        "executive_summary": (
            "The knowledge base does not hold material that answers this question. It "
            "covers AI and machine learning: research papers, engineering release notes "
            "and documentation, and AI standards and regulations. Rather than assemble an "
            "answer from unrelated sources, no answer is given."
        ),
        "key_findings": [],
        "conflicts": [],
        "citations": [],
        "confidence": "none",
        "knowledge_gaps": reasons or ["Nothing retrieved was relevant to this question."],
        "follow_up_questions": [
            "Ask about a specific model architecture, inference technique, or ML library.",
            "Ask about AI regulation, such as the EU AI Act or the NIST AI Risk Management Framework.",
            "Ask what changed in a recent release of a tracked project.",
        ],
        "out_of_scope": True,
        "nearest_matches": [
            {"title": result.title, "source": result.source, "score": round(result.score, 2)}
            for result in nearest[:3]
        ],
    }
