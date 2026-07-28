# Reflection

## What holds up best

Citations cannot be fabricated, and that is structural rather than
hopeful. The step that assembles the final brief is ordinary Python, not
another model call: it resolves cited numbers against the passages that
were actually retrieved, so an invented number resolves to nothing and
disappears. Validity measured 1.0 across every run, including runs where
the model failed in other ways. The guarantee held exactly where it was
supposed to, and did not paper over the failure sitting next to it.

The second thing is the failure-handling pattern: record what failed, keep
going, never let one bad item end a run. It started in ingestion, for dead
URLs and unreachable repositories, and turned out to fit the indexer and
the evaluation runner just as well. When a free-tier API quota ran out
mid-evaluation, the run recorded the error per query and still produced a
report instead of losing everything computed up to that point.

The third, less comfortably, is finding out that the first version of the
out-of-scope check was wrong, and finding out quickly. It was built from a
real gap in the evaluation, scored well against the evaluation query set,
and read as a solid result in an earlier draft of this document. One
question outside that set — "what is AI" — broke it immediately: it
scored lower than the out-of-domain example the whole approach was
calibrated against, because reranker scores compare passages within a
query, not across different ones, and nothing about the evaluation set had
a way to catch that. The fix was to move the judgement to the model, which
was already reviewing the evidence for sufficiency and could just as well
review it for relevance. What holds up about this isn't the first attempt;
it's that testing past the evaluation set caught the mistake before it
shipped as a claimed result.

## What would change with more time

**Broaden the evaluation query set**, specifically with more than one
out-of-domain example and some broad, simple questions alongside the
specific ones. A set with a single question of each awkward kind can make
a broken approach look validated, which is exactly what happened here.

**Add a fallback when category routing comes back thin.** The router
commits to categories before searching. When the chosen category turns out
to be poorly covered, there is currently no way to widen the search for
that sub-question.

**Grow the corpus and re-measure.** Some evaluation findings reflect how
much was ingested rather than how the system is built, and only a larger
run separates the two.

**Run the full evaluation against the local model, not just a manual
question or two.** Running it manually is what surfaced the scope-check
problem in the first place, which says the local path is worth exercising
more, not less. A structured comparison against the hosted model's results
would show whether a small local model's structured output is reliable
enough for every reasoning step, not just the one that happened to get
tried.

## The tradeoff that mattered most

Category-aware retrieval is not a free improvement, and measurement is
what revealed the cost.

Restricting a sub-question to relevant categories works: precision was 1.0
on every query the corpus could answer. But the same restriction produced
the worst result in the evaluation. Asked about speculative decoding, the
agent routed to technical literature, where the corpus had nothing on the
topic, and search returned the closest available paper — which was about
vision-language models and entirely irrelevant. The same query without the
category restriction finds a genuinely useful document in a different
category. Same embeddings, same reranker, same corpus; the only variable
was whether search was allowed to look outside the category already chosen.

Restricting the search space is the obvious way to make retrieval
category-aware, and it is right most of the time. The cost only became
visible after a result that looked slightly off got examined instead of
accepted. That is the part worth keeping: a plausible answer and a correct
one look identical until something exists that can tell them apart.
