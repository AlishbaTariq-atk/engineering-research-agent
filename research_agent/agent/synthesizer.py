from __future__ import annotations

from pydantic import BaseModel, Field

from research_agent.agent.llm import get_llm
from research_agent.config import Settings

SYNTHESIZER_SYSTEM_PROMPT = """You are a research analyst synthesizing evidence \
from a technical knowledge base to answer a research question. You are given \
numbered evidence passages [1], [2], etc, each tagged with its source and \
publication date.

Rules:
- Every claim in key_findings MUST cite the passage number(s) that support it \
  via citation_ids, e.g. a claim citing passages 3 and 7 has citation_ids: [3, 7].
- If passages disagree with each other, do not silently pick one - report it in \
  `conflicts` with the citation numbers of the disagreeing passages.
- If an older passage is contradicted or superseded by a newer one, say so \
  explicitly rather than treating them as equally current.
- Do not state anything not supported by the provided passages. If the evidence \
  is thin for part of the question, say so in knowledge_gaps rather than guessing.
- executive_summary should be readable on its own, 2-4 sentences.
- follow_up_questions should be concrete next research questions, not generic \
  ("investigate further")."""


class Finding(BaseModel):
    claim: str
    citation_ids: list[int] = Field(description="passage numbers from the evidence that support this claim")
    confidence: str = Field(description="one of: high, medium, low")


class Conflict(BaseModel):
    description: str
    citation_ids: list[int] = Field(description="passage numbers involved in the disagreement")


class SynthesisOutput(BaseModel):
    executive_summary: str
    key_findings: list[Finding]
    conflicts: list[Conflict] = Field(default_factory=list)
    overall_confidence: str = Field(description="one of: high, medium, low")
    knowledge_gaps: list[str] = Field(min_length=1)
    follow_up_questions: list[str] = Field(min_length=1, max_length=5)


def synthesize(question: str, sub_questions: list[str], context: str, settings: Settings) -> SynthesisOutput:
    """Cross-source comparison and conflict identification happen here, in
    one pass over all pooled evidence - not per-sub-question - so the model
    can actually notice when a GitHub changelog and an arXiv paper disagree,
    which it couldn't do if each sub-question's evidence were synthesized
    in isolation."""
    llm = get_llm(settings)
    structured_llm = llm.with_structured_output(SynthesisOutput)
    user_prompt = (
        f"Research question: {question}\n\nSub-questions considered:\n"
        + "\n".join(f"- {q}" for q in sub_questions)
        + f"\n\nEvidence:\n{context}"
    )
    return structured_llm.invoke([("system", SYNTHESIZER_SYSTEM_PROMPT), ("human", user_prompt)])
