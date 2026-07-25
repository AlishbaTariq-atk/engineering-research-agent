from __future__ import annotations

from pydantic import BaseModel, Field

from research_agent.agent.llm import get_llm
from research_agent.config import Settings

CRITIC_SYSTEM_PROMPT = """You assess whether retrieved evidence is enough to \
answer a research question well. Given the question, its sub-questions, and \
the evidence retrieved so far, decide if another round of retrieval is needed.

Always name at least one limitation, missing angle, or promising follow-up \
direction in `gaps`, even when the evidence is already sufficient to answer -
a research brief with zero acknowledged limitations is a red flag, not a \
sign of completeness."""


class GapAssessment(BaseModel):
    sufficient: bool = Field(description="true if no further retrieval is needed")
    gaps: list[str] = Field(min_length=1, description="missing angles or limitations - at least one, always")


def assess_gaps(question: str, sub_questions: list[str], context: str, settings: Settings) -> GapAssessment:
    """The node that makes retrieval iterative instead of one-shot: the LLM
    judges its own evidence and can request another round before
    synthesis, rather than synthesizing from whatever the first pass
    happened to find. `gaps` doubles as input to the next retrieval round
    (if not sufficient) and as the report's knowledge_gaps seed (if it is)."""
    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(GapAssessment)
    user_prompt = (
        f"Research question: {question}\n\nSub-questions:\n"
        + "\n".join(f"- {q}" for q in sub_questions)
        + f"\n\nEvidence retrieved so far:\n{context}"
    )
    return structured_llm.invoke([("system", CRITIC_SYSTEM_PROMPT), ("human", user_prompt)])
