# Evaluation Report

Two metric categories, per the assessment's minimum of two meaningful
aspects: **retrieval precision** and **citation faithfulness**. Both are
measured, not assumed - including one case where a real system failure
(an exhausted API quota) happened mid-evaluation and is reported as data,
not hidden.

Run against the 12-query set in `research_agent/evaluation/query_set.py`,
grounded in the actual ~1800-document test corpus (not the full ~50k
target - see "Corpus scale" in `docs/DESIGN.md`).

```bash
research-agent eval               # full run
research-agent eval --skip-agent  # retrieval-only, no LLM calls/API cost
```

## 1. Retrieval precision (12/12 queries completed)

`category_precision_at_5` is scored against an **unrestricted** search
(all three Chroma collections), not one pre-filtered to the expected
category - filtering first would make the metric 1.0 by construction
regardless of ranking quality. This was an actual bug in the first version
of this eval, caught by noticing a suspiciously perfect score before
trusting it (see commit history) rather than assumed correct.

| Precision | Title match | Top result score | Query |
|---|---|---|---|
| 1.0 | ✓ | 8.176 | What is the NIST AI Risk Management Framework? |
| 1.0 | ✓ | 7.726 | What are the requirements of the EU AI Act for high-risk AI systems? |
| 1.0 | ✓ | 6.080 | What executive order governs AI safety and security in the US? |
| 1.0 | ✓ | 6.320 | What new features were added in recent vllm releases? |
| 1.0 | ✓ | 4.115 | How is speculative decoding implemented in vllm? |
| 1.0 | ✓ | 7.531 | What is LangChain and what is it used for? |
| 1.0 | ✓ | 6.893 | What research exists on molecular ensemble modeling of cyclic peptides? |
| 1.0 | ✓ | 2.921 | What does recent research say about sycophancy and moral reasoning in LLMs? |
| **0.0** | ✓ | **-0.405** | *(known gap)* What does recent **academic** research say about speculative decoding? |
| 1.0 | ✓ | 2.342 | What does the corpus say about gravure printing quality control automation? |
| 1.0 | ✓ | 1.158 | *(cross-category)* How do AI regulations like the EU AI Act relate to practical LLM deployment tools? |
| **0.0** | ✓ | **-8.845** | *(out-of-domain control)* What is the capital of France? |

**Summary**: mean precision@5 = **1.0** on the 10 queries with a genuine
in-corpus answer, **0.0** on both deliberate known-gap queries. Title
match rate (does the one specific known-correct document actually surface
in the top 5, given correct category routing) = **1.0** across all 8
queries where a specific document was checkable.

**Interpretation - this is a real result, not a trivially perfect one**:
the 0.0 scores on the known-gap queries are not failures, they're the
metric correctly confirming what manual corpus inspection already
established - `technical_literature` genuinely has no paper about
inference optimization at this corpus scale, and nothing in the corpus
answers "capital of France." The score gap is the informative part: the
academic-research known-gap query's best unrestricted match scored
**-0.405** (weak but not absurd - it found a genuinely adjacent
practitioner-knowledge document about efficient inference), while the
fully out-of-domain control scored **-8.845** (nearly 20x more negative).
The cross-encoder's score is a real, usable confidence signal that
distinguishes "weak tangential match" from "no match exists" - not
currently used as a threshold anywhere in the system, which is a concrete,
scoped next step (see `docs/DESIGN.md` §9).

## 2. Citation faithfulness

### What happened: the formal batch hit a real infra limit

Running `research-agent eval` (agent pass, 12 queries × ~6 LLM calls each)
hit Groq's free-tier daily cap (**100,000 tokens/day**) partway through
development testing on the same day. The retry, after the fix below, is
itself informative: **all 12 queries failed cleanly** with a recorded
`RateLimitError` reason each, and the run still completed and wrote a full
report - rather than crashing and losing everything, which is what
happened on the *first* attempt before a fix (below).

This is reported as a real finding, not smoothed over: **the free tier is
not sufficient for iterative development plus a full evaluation batch in
the same day.** A production evaluation cadence would need a paid tier,
evaluation spread across multiple days, or a run against the Ollama
fallback (architecturally identical - same `.with_structured_output()`
interface - but not yet exercised live in this environment).

### Bug found and fixed via this exact failure

The first attempt at the agent-pass batch didn't fail cleanly - a single
query's `RateLimitError` was an uncaught exception that crashed the entire
loop, discarding every result already computed (about 8 of 12 queries'
worth of real Groq calls, wasted). Fixed by wrapping each query in
`evaluate_agent()` in its own try/except, recording failures as explicit
rows instead of raising - the same "one bad item can't take the whole run
down" principle already applied to ingestion (`FetchFailure` in
`pipeline.py`), just missed here until it happened for real. The second
attempt, shown above, demonstrates the fix working under a genuine
failure, not a simulated one.

### Real citation-quality data (3 informal spot-checks)

Since the formal 12-query batch couldn't complete against the exhausted
quota, citation metrics here come from three real end-to-end runs
captured earlier in development (same question, evolving code across
fixes - see commit history), scored with the same `citation_coverage`/
`citation_validity` functions used by the formal framework:

| Run | citation_coverage | citation_validity | Confidence | Note |
|---|---|---|---|---|
| 1 (pre-fix, unbatched routing) | 0.667 | 1.0 | medium | 1 of 3 findings had zero citations |
| 2 (post-fix, batched routing) | 0.667 | 1.0 | medium | Same gap reproduced independently |
| 3 (post-fix, re-run) | 1.0 | 1.0 | medium | All findings cited this time |

**Mean citation_coverage: 0.778. Mean citation_validity: 1.0 (all 3 runs).**

**Interpretation**:

- `citation_validity` = 1.0 across every run, always, by construction -
  `report_generator.py` can only resolve a citation ID against chunks
  that were actually retrieved, never invent one. This is a hallucination-
  resistance guarantee, not a measured tendency, and the eval confirms it
  held in practice, not just in theory.
- `citation_coverage` < 1.0 in 2 of 3 runs is a real fault, not noise: the
  synthesis prompt explicitly instructs "do not state anything not
  supported by the provided passages," and the model violated it (a
  finding about regulatory risk, asserted with `citations: []`) in the
  same run, twice independently. Low confidence was assigned to that
  finding both times, which is a partial mitigation, but a claim with zero
  citations is a faithfulness violation regardless of its confidence
  label. This is exactly the kind of failure a citation_coverage metric
  exists to catch mechanically instead of relying on a human noticing it
  once in a demo.
- The **same wrong citation** (an unrelated VLM paper, for a
  speculative-decoding claim) appeared in runs 1 and 3, independently.
  Combined with the retrieval-precision finding above, this has a precise
  diagnosis: the agent's router restricts a sub-question's search to
  `technical_literature` specifically, and that category has no good
  match for this question at current corpus scale - so retrieval is
  forced to return the least-bad in-category candidate instead of
  surfacing the better cross-category match an unrestricted search finds.
  **This is a category-routing limitation compounding a corpus-scale
  limitation, not a reranker defect in isolation** - the same reranker,
  given an unrestricted candidate pool, correctly scored that match's true
  relevance near zero rather than confidently wrong.

## 3. Latency

Not originally planned as a headline metric, but became one after two
real bugs were found by timing actual runs rather than assuming
performance was fine:

| Configuration | Wall time | LLM calls |
|---|---|---|
| Before fix: no persistent embedder/reranker, per-sub-question routing | 5m 7s | 9-10 |
| After fix: persistent embedder/reranker + batched routing | **31.5s** | **6** |

Root causes (both in `research_agent/agent/graph.py`, see commit history
for detail): the retrieval node was constructing a fresh `Embedder`/
`Reranker` on every call instead of reusing one instance across the run,
and routing made one LLM call per sub-question instead of one call for
the whole round. The second bug is also what most likely caused an
unexplained 138-second silent stall observed during testing before the
fix - consistent with a Groq rate-limit retry, since `httpx` only logs a
successful request, never a failed/retried attempt.

## Strengths

- Citation hallucination-resistance is real and verified, not assumed:
  `citation_validity` was 1.0 across every run, and the mechanism (lookup-
  only, never generative) makes a failure structurally hard to produce.
- Retrieval precision is strong (1.0) whenever the corpus genuinely has a
  relevant document, across a deliberately varied set of topics (policy
  documents, GitHub tooling, and an intentionally obscure paper about
  gravure printing) - this isn't retrieval succeeding only on easy,
  generic AI/ML queries.
- The cross-encoder's raw score is a real, usable relevance signal (a
  ~20x gap between a weak tangential match and a true non-match), not
  currently exploited but validated as available for future use.
- Failure handling is resilient at every layer that's been tested against
  a real failure: ingestion (dead URLs, a broken GitHub repo), and now
  evaluation (an exhausted API quota) - all degrade to a recorded,
  queryable failure instead of crashing.

## Weaknesses and failure cases

- Citation coverage is not guaranteed: 2 of 3 real runs produced at least
  one finding with zero supporting citations, directly contradicting an
  explicit system-prompt instruction. The model's own confidence labeling
  partially compensates (both cases were marked "low") but doesn't fully
  fix it.
- Category-restricted routing can force a worse answer than an
  unrestricted search would give, specifically when the router's chosen
  category is sparse for the question at hand. No fallback exists yet for
  "the restricted search came back weak, try broader."
- The formal agent-level evaluation batch could not complete in this
  environment due to real free-tier quota exhaustion - the numbers above
  are a smaller, real sample (3 runs), not the full 12-query set the
  framework is built to run once quota allows.
- All findings here are at ~1800-document corpus scale, not the ~50k
  target - several results (technical_literature sparsity specifically)
  are plausibly corpus-scale artifacts that a full-scale run could
  resolve or could reveal are more fundamental. This can't be
  distinguished without actually running it.

## Future improvements

See `docs/DESIGN.md` §9 for the full list; the two most directly motivated
by this evaluation are a reranker-score confidence threshold (the -8.845
vs. -0.405 gap is sitting right there, unused) and a fallback from
restricted to unrestricted search when the restricted result set is weak.
