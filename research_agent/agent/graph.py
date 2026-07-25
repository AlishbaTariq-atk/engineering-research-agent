from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from research_agent.agent.critic import assess_gaps
from research_agent.agent.planner import decompose_question
from research_agent.agent.report_generator import build_research_brief
from research_agent.agent.router import route_sub_questions
from research_agent.agent.synthesizer import synthesize
from research_agent.config import Settings
from research_agent.retrieval import Embedder, Reranker, build_context, citation_map
from research_agent.retrieval import search as retrieval_search
from research_agent.retrieval.types import RetrievedChunk

DEFAULT_MAX_ITERATIONS = 2


class AgentState(TypedDict):
    question: str
    sub_questions: list[str]
    pending_sub_questions: list[str]  # sub-questions to retrieve for THIS round
    retrieved: list[RetrievedChunk]
    iteration: int
    gaps: list[str]
    sufficient: bool
    report: dict


def build_graph(settings: Settings, max_iterations: int = DEFAULT_MAX_ITERATIONS):
    """`settings` is closed over by the node functions rather than carried
    in AgentState - state should stay plain, run-specific data (what LangGraph's
    checkpointing/inspection is meant to work with), not a Settings object
    it has no reason to know about.

    Embedder/Reranker are loaded exactly once here too, for the same
    reason the MCP server loads them once at module level: retrieval.search()
    defaults to constructing its own if none are passed, which is correct
    for a single one-off call but would silently reload both models (and,
    without HF_HUB_OFFLINE, re-verify their cache online) on every one of
    the several retrieval calls a single agent run makes - found by timing
    a real run and seeing multi-second gaps between LLM calls that had no
    business being that slow.
    """
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker()

    def plan_node(state: AgentState) -> dict:
        sub_questions = decompose_question(state["question"], settings)
        return {"sub_questions": sub_questions, "pending_sub_questions": sub_questions}

    def retrieve_node(state: AgentState) -> dict:
        # Pool + dedupe by chunk_id across sub-questions and iterations, so
        # a chunk matched by two different sub-questions (or re-matched on
        # a later iteration) contributes to synthesis once, not twice.
        pooled = {c.chunk_id: c for c in state["retrieved"]}
        pending = state["pending_sub_questions"]
        routing = route_sub_questions(pending, settings)  # one LLM call for the whole round, not one per sub-question
        for sub_q in pending:
            results = retrieval_search(
                sub_q, settings, categories=routing[sub_q], top_k=6, embedder=embedder, reranker=reranker
            )
            for r in results:
                pooled[r.chunk_id] = r
        return {"retrieved": list(pooled.values()), "iteration": state["iteration"] + 1}

    def gap_check_node(state: AgentState) -> dict:
        context = build_context(state["retrieved"])
        assessment = assess_gaps(state["question"], state["sub_questions"], context, settings)
        return {"gaps": assessment.gaps, "sufficient": assessment.sufficient, "pending_sub_questions": assessment.gaps}

    def should_continue(state: AgentState) -> str:
        if state["sufficient"] or state["iteration"] >= max_iterations:
            return "synthesize"
        return "retrieve"

    def synthesize_node(state: AgentState) -> dict:
        chunks = state["retrieved"]
        context = build_context(chunks)
        cmap = citation_map(chunks)
        synthesis = synthesize(state["question"], state["sub_questions"], context, settings)
        report = build_research_brief(state["question"], synthesis, cmap)
        # Merge the critic's last-round gaps into the synthesizer's own
        # knowledge_gaps, deduped, rather than picking one source - both
        # are genuine signals about what's missing.
        report["knowledge_gaps"] = list(dict.fromkeys(report["knowledge_gaps"] + state.get("gaps", [])))
        return {"report": report}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("gap_check", gap_check_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "gap_check")
    graph.add_conditional_edges("gap_check", should_continue, {"retrieve": "retrieve", "synthesize": "synthesize"})
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_research_agent(question: str, settings: Settings, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> dict:
    graph = build_graph(settings, max_iterations=max_iterations)
    initial_state: AgentState = {
        "question": question,
        "sub_questions": [],
        "pending_sub_questions": [],
        "retrieved": [],
        "iteration": 0,
        "gaps": [],
        "sufficient": False,
        "report": {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["report"]
