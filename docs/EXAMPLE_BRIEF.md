# Example Output

A real session, reproduced as it ran. It includes a wrong citation, kept
here and explained rather than replaced with a cleaner run.

```
$ python main.py

Research Assistant
Ask a technical question and get an evidence-backed brief.
Type 'exit' or press Ctrl-C to quit.

Knowledge base: 4828 documents, 4200 indexed for search.

Question> What is speculative decoding, what are its tradeoffs for LLM
          inference, and are there relevant AI regulatory considerations?
```

Six model calls, 31.5 seconds: one to plan, two rounds of routing, two gap
checks, one to write.

---

## Summary

> Speculative decoding is a technique used in LLM inference that allows for
> self-speculative decoding, where the model decodes with earlier layers
> and verifies with remaining layers. The tradeoffs include a potential
> decrease in speed, but it can also lead to increased efficiency and
> accuracy. Relevant AI regulatory considerations include ensuring that its
> use does not compromise the fairness, transparency, and accountability of
> AI systems.

## Key findings

| Claim | Confidence | Source |
|---|---|---|
| Speculative decoding allows a model to decode with earlier layers and verify with the rest. | high | [2] transformers v4.47.0 release notes |
| Tradeoffs include a potential decrease in speed. | medium | [7] *3D-Aware VLMs with Implicit and Explicit Geometries* |
| It can lead to increased efficiency and accuracy. | medium | [8] vllm v0.5.4 release notes |

**Citation [7] is wrong.** That paper is about vision-language model
geometry and has nothing to do with speculative decoding. The agent had
routed that sub-question to technical literature alone, and the corpus held
no paper on inference optimisation, so search returned the closest thing
available rather than nothing. This is a category-routing gap, not a
scope-detection one - see the next section for the distinction, and
`EVALUATION.md` for the full analysis.

## Sources

| # | Title | Source | Date |
|---|---|---|---|
| 2 | huggingface/transformers v4.47.0 | github | 2024-12-05 |
| 7 | 3D-Aware VLMs with Implicit and Explicit Geometries | arxiv | 2026-07-23 |
| 8 | vllm-project/vllm v0.5.4 | github | 2024-08-05 |

## Overall confidence

Medium.

## Knowledge gaps

- The impact of speculative decoding on AI regulatory considerations is not fully understood.
- The evidence does not give a comprehensive picture of the technique, its tradeoffs, and the regulatory position together.
- There is little on the risks involved, such as bias or errors introduced during decoding.
- What was retrieved is largely technical, with little on ethical or societal implications.

## Suggested follow-ups

- What specific AI regulatory considerations apply to speculative decoding in LLM inference?
- How can speculative decoding be optimised to reduce its speed cost while keeping its efficiency and accuracy gains?

---

Worth noting: the system correctly flagged the regulatory angle as thin,
and it was. The three standards documents in the corpus are general
governance frameworks and say nothing about inference-time techniques.
The gap detection worked in the same run where a citation did not, which
is why both appear here.

## A question the knowledge base cannot answer

Asking something clearly outside the domain - "What is the capital of
France?" - stops after the review step rather than assembling an answer
from whatever search came closest. The model is shown the retrieved
passages next to the question and judges that none of them, an AI
regulation and a couple of unrelated technical papers among them, actually
address it. The reply names what was found and why it does not answer the
question, in the same `OUT OF SCOPE` format `main.py` prints for any
declined question.

*A verbatim transcript belongs here once one is captured against the
current code. An earlier version of this section showed output from a
different approach to this check - comparing a search score against a
fixed cutoff - which produced wrong results on ordinary questions and was
replaced with the judgement described above. See `DESIGN.md` and
`REFLECTION.md` for what went wrong with it and why. Reusing that old
transcript here would describe behaviour the system no longer has.*
