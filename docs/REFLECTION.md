# Reflection

## What I'm most satisfied with

The citation hallucination-resistance design, because it held up under
real testing rather than just sounding good on paper. `report_generator.py`
is deliberately not an LLM call - it resolves the synthesizer's citation
IDs against a `citation_map` built from the chunks that were actually
retrieved, so a citation can only be looked up or dropped, never invented.
The evaluation confirmed `citation_validity` at 1.0 across every real run,
including the one where the model *did* fail (a finding with zero
citations, discussed below) - the guarantee held exactly where it was
supposed to and didn't paper over the failure that was outside its scope.

Close second: the failure-handling pattern (log the specific failure,
keep going, never let one bad item take down a whole run) turned out to
generalize cleanly across the entire system - ingestion adapters, the
indexing pipeline, and finally the evaluation framework itself, where it
was proven under a real failure (an exhausted Groq quota) rather than a
simulated one.

## What I'd improve with more time

1. **Run at actual scale.** Everything here was built and evaluated
   against ~1800 documents, not the ~50k target, to keep iteration and API
   costs bounded during development. Several evaluation findings -
   particularly the sparse `technical_literature` coverage that produced
   the wrong VLM-paper citation - are plausibly artifacts of that small,
   arbitrary sample (15 "most recent" arXiv papers, not selected for
   topical relevance) rather than fundamental design flaws. I can't tell
   which without actually running the full ingestion at scale, which the
   pipeline supports (no code changes needed) but which I didn't spend the
   API/time budget on this round.
2. **Use the reranker's confidence signal.** The evaluation surfaced a
   nearly 20x score gap between a weak tangential match (-0.405) and a
   genuine non-match (-8.845) that the system doesn't currently act on
   anywhere. A threshold below which the agent reports "no relevant
   evidence" instead of forcing a citation seems like a small change with
   a real, already-measured payoff.
3. **A fallback when category-restricted retrieval comes back weak.** The
   router commits to categories before retrieval runs; when the chosen
   category turns out sparse for that specific question, there's currently
   no way back to a broader search within the same sub-question.
4. **Test the Ollama fallback live.** It's architecturally identical to
   the Groq path (same LangChain chat model interface), and exists
   specifically for offline resilience, but I never actually exercised it
   end-to-end in this environment - only reasoned that it should work.

## The largest engineering tradeoff I encountered

Category-aware retrieval is a real tradeoff, not a strict improvement, and
I only found this out by testing rather than by reasoning about it in
advance. Restricting a sub-question's search to specific categories (via
the router) measurably helps: the evaluation's precision@5 was 1.0 on
every query where the corpus had genuine coverage. But the same mechanism
measurably hurt on the one query where the "correct" category was sparse -
restricting to `technical_literature` for a speculative-decoding question
forced the system to cite the least-bad in-category match (an unrelated
VLM paper) instead of a genuinely relevant document that an *unrestricted*
search found without any trouble in a different category. Same embeddings,
same reranker, same corpus - the only variable was whether the search was
allowed to look outside the category the router had already committed to.

I didn't design around this tradeoff going in; I designed category-aware
retrieval because it's explicitly requested by the assessment and because
restricting the candidate pool is the obvious way to make a search
"category-aware" at all. The cost side only became visible once I ran a
real evaluation against a real (if small) corpus and looked hard at a
result that seemed slightly off, rather than accepting a query that
returned *an* answer as evidence the system worked. That's probably the
main thing this project reinforced for me: a plausible-looking answer and
a correct one aren't the same thing, and the gap between them is usually
invisible until you build the measurement that can see it.
