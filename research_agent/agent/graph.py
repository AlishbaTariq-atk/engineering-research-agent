"""The research workflow, wired together as a graph.

    plan -> search -> review -> write brief
                        │  └--> search again
                        └-----> decline, if nothing found is relevant

The loop back from the review step is the point of using a graph here: the
number of search rounds depends on what the evidence turns out to look
like, rather than being fixed in advance. LangGraph handles the wiring and
carries state between steps; every decision the agent makes lives in
`nodes.py`.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from research_agent.config import Settings
from research_agent.retrieval import Embedder, Reranker, SearchResult, build_context, citation_map, search

from . import nodes

MAX_ROUNDS = 2
RESULTS_PER_SUB_QUESTION = 6


class AgentState(TypedDict):
    """What the workflow carries from one step to the next."""

    question: str
    sub_questions: list[str]
    pending: list[str]  # What to search for in the current round.
    results: list[SearchResult]
    round_number: int
    gaps: list[str]
    answerable: bool
    sufficient: bool
    brief: dict


def build_graph(settings: Settings, max_rounds: int = MAX_ROUNDS):
    """Assemble the workflow.

    The embedding and reranking models are loaded once here and reused by
    every search in the run. Left to itself, `search` loads its own copy on
    each call, which is fine for a one-off query but wasteful across the
    several searches one run performs.

    Args:
        settings: Application configuration.
        max_rounds: Hard ceiling on search rounds, so a question the agent
            keeps finding gaps in still terminates.

    Returns:
        The compiled workflow, ready to invoke.
    """
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(settings.reranker_model)

    def plan_step(state: AgentState) -> dict:
        """Break the question into sub-questions to search for."""
        sub_questions = nodes.plan(state["question"], settings)
        return {"sub_questions": sub_questions, "pending": sub_questions}

    def search_step(state: AgentState) -> dict:
        """Search for every pending sub-question and pool the results.

        Results are keyed by chunk id, so a passage found by two different
        sub-questions, or found again in a later round, is kept once.
        """
        pooled = {result.chunk_id: result for result in state["results"]}
        routing = nodes.route(state["pending"], settings)

        for sub_question in state["pending"]:
            hits = search(
                sub_question,
                settings,
                categories=routing[sub_question],
                top_k=RESULTS_PER_SUB_QUESTION,
                embedder=embedder,
                reranker=reranker,
            )
            for hit in hits:
                pooled[hit.chunk_id] = hit

        return {"results": list(pooled.values()), "round_number": state["round_number"] + 1}

    def review_step(state: AgentState) -> dict:
        """Judge whether the evidence is relevant, and whether it is enough."""
        assessment = nodes.assess_gaps(
            state["question"], state["sub_questions"], build_context(state["results"]), settings
        )
        return {
            "gaps": assessment.gaps,
            "answerable": assessment.answerable,
            "sufficient": assessment.sufficient,
            "pending": assessment.gaps,
        }

    def next_step(state: AgentState) -> str:
        """Choose what happens after reviewing the evidence.

        Irrelevant evidence ends the run: if the knowledge base holds
        nothing on the subject, searching again will not change that.
        Otherwise the run either searches for what is missing or, once the
        evidence is sufficient or the round limit is reached, writes up.
        """
        if not state["answerable"]:
            return "decline"
        if state["sufficient"] or state["round_number"] >= max_rounds:
            return "write"
        return "search"

    def decline_step(state: AgentState) -> dict:
        """Report that the question is outside what the knowledge base covers."""
        brief = nodes.build_out_of_scope_brief(
            state["question"], state["results"], state.get("gaps", [])
        )
        return {"brief": brief}

    def write_step(state: AgentState) -> dict:
        """Write the brief from everything gathered."""
        results = state["results"]
        synthesis = nodes.synthesise(
            state["question"], state["sub_questions"], build_context(results), settings
        )
        brief = nodes.build_brief(state["question"], synthesis, citation_map(results))

        # The critic and the synthesiser each report gaps, and they tend to
        # notice different things, so both are kept, in order, without repeats.
        brief["knowledge_gaps"] = list(dict.fromkeys(brief["knowledge_gaps"] + state.get("gaps", [])))
        return {"brief": brief}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_step)
    graph.add_node("search", search_step)
    graph.add_node("review", review_step)
    graph.add_node("decline", decline_step)
    graph.add_node("write", write_step)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "review")
    graph.add_conditional_edges(
        "review", next_step, {"search": "search", "write": "write", "decline": "decline"}
    )
    graph.add_edge("write", END)
    graph.add_edge("decline", END)

    return graph.compile()


def answer_question(question: str, settings: Settings, max_rounds: int = MAX_ROUNDS) -> dict:
    """Research a question and return an evidence-backed brief.

    Args:
        question: The research question to answer.
        settings: Application configuration.
        max_rounds: Maximum search rounds before writing the answer.

    Returns:
        A brief containing an executive summary, cited findings, any
        conflicts between sources, a confidence rating, knowledge gaps, and
        suggested follow-up questions.
    """
    graph = build_graph(settings, max_rounds=max_rounds)
    initial: AgentState = {
        "question": question,
        "sub_questions": [],
        "pending": [],
        "results": [],
        "round_number": 0,
        "gaps": [],
        "answerable": True,
        "sufficient": False,
        "brief": {},
    }
    return graph.invoke(initial)["brief"]
