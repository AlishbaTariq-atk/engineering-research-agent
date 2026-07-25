# Example Research Brief

Real captured output from `research-agent ask`, run against the live
system (Groq `llama-3.3-70b-versatile`, ~1800-document test corpus).
Reproduced verbatim - including a real quality issue, annotated rather
than edited out, since an honest example is more useful than a flattering
one.

**Command:**
```bash
research-agent ask "What is speculative decoding, what are its tradeoffs for LLM inference, and are there relevant AI regulatory considerations?"
```

**Total time**: 31.5s · **LLM calls**: 6 (1 plan, 2 routing rounds, 2 gap-checks, 1 synthesis)

---

## Executive Summary

> Speculative decoding is a technique used in LLM inference that allows for
> self-speculative decoding, where the model decodes with earlier layers
> and verifies with remaining layers. The tradeoffs of using speculative
> decoding include a potential decrease in speed, but it can also lead to
> increased efficiency and accuracy. Relevant AI regulatory considerations
> include ensuring that the use of speculative decoding does not compromise
> the fairness, transparency, and accountability of AI systems.

## Key Findings

| Claim | Confidence | Citation |
|---|---|---|
| Speculative decoding is a technique used in LLM inference that allows for self-speculative decoding. | High | [2] transformers v4.47.0 release notes |
| The tradeoffs of using speculative decoding include a potential decrease in speed. | Medium | [7] *3D-Aware VLMs with Implicit and Explicit Geometries* (arXiv) |
| Speculative decoding can lead to increased efficiency and accuracy. | Medium | [8] vllm v0.5.4 release notes |

**⚠️ Known issue, not edited out:** citation [7] is wrong. That arXiv paper
is about vision-language model geometry, not speculative decoding - it was
cited because the test corpus's `technical_literature` category (15
arbitrary recent papers) has no real paper on inference optimization, and
the router restricted that sub-question's search to `technical_literature`
only. See `docs/EVALUATION.md` for the root-cause analysis: an unrestricted
search for the same underlying question correctly finds a genuinely
relevant *practitioner_knowledge* result instead. This is a corpus-scale
and category-routing limitation, documented and measured, not silently
shipped.

## Conflicts

None identified in this run (the corpus at test scale didn't surface
disagreeing sources for this question).

## Citations

| # | Title | Source | Date |
|---|---|---|---|
| 2 | huggingface/transformers v4.47.0 | github | 2024-12-05 |
| 7 | 3D-Aware VLMs with Implicit and Explicit Geometries | arxiv | 2026-07-23 |
| 8 | vllm-project/vllm v0.5.4 | github | 2024-08-05 |

## Confidence Assessment

**Medium.**

## Identified Knowledge Gaps

- The impact of speculative decoding on AI regulatory considerations is not fully understood.
- The current evidence does not provide a comprehensive overview of speculative decoding, its tradeoffs for LLM inference, and relevant AI regulatory considerations. Further research is needed.
- There is a lack of information on the potential risks and challenges associated with speculative decoding, such as potential biases or errors in the decoding process.
- The evidence retrieved so far is mostly focused on the technical aspects of speculative decoding, with little discussion on its ethical and societal implications.

## Recommended Follow-up Questions

- What are the specific AI regulatory considerations that need to be addressed when using speculative decoding in LLM inference?
- How can speculative decoding be optimized to minimize its potential decrease in speed while maximizing its benefits in terms of efficiency and accuracy?

---

Note the system's own `knowledge_gaps` output correctly flagged that the
regulatory angle was thin - which it was: none of the three curated
standards documents (NIST AI RMF, EU AI Act, US EO 14110) discuss
inference-time techniques specifically, they're general AI governance
frameworks. The gap-detection mechanism worked as designed even though the
citation-precision mechanism didn't, in the same run - both are reported
honestly in `docs/EVALUATION.md` rather than only showing the parts that worked.
