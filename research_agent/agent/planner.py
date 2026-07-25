from __future__ import annotations

from pydantic import BaseModel, Field

from research_agent.agent.llm import get_llm
from research_agent.config import Settings

PLANNER_SYSTEM_PROMPT = """You are a research planning assistant. Given a \
technical research question, break it into 2-4 focused sub-questions that \
together cover the aspects needed to answer it well. Each sub-question \
should be answerable independently by searching a knowledge base. Avoid \
redundant sub-questions."""


class DecompositionOutput(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=4)


def decompose_question(question: str, settings: Settings) -> list[str]:
    """First reasoning step: turn one research question into several
    focused sub-questions. A flat "retrieve and summarize" system wouldn't
    need this - it's what makes multi-angle, category-aware retrieval
    possible instead of one similarity search against the whole corpus."""
    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(DecompositionOutput)
    result: DecompositionOutput = structured_llm.invoke(
        [("system", PLANNER_SYSTEM_PROMPT), ("human", question)]
    )
    return result.sub_questions
