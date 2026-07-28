# Evaluation

Two things are measured: whether search finds the right material, and
whether the written answer is grounded in what it found. Latency is
tracked alongside both.

```bash
python main.py eval --skip-agent   # retrieval only, no model calls
python main.py eval                # full run
```

The query set lives in `research_agent/evaluation.py`. It is built from
documents known to be in the corpus, and includes questions the corpus
cannot answer, so that fabricated confidence shows up as a failure rather
than passing unnoticed.

## Retrieval quality

Category precision is scored against a search across *all* categories, not
one already limited to the expected category. Limiting first would
guarantee a perfect score no matter how bad the ranking was, so the metric
would measure nothing.

| Precision | Right doc found | Top score | Query |
|---|---|---|---|
| 1.0 | yes | 8.18 | What is the NIST AI Risk Management Framework? |
| 1.0 | yes | 7.73 | What are the requirements of the EU AI Act for high-risk AI systems? |
| 1.0 | yes | 6.08 | What executive order governs AI safety and security in the US? |
| 1.0 | yes | 6.32 | What new features were added in recent vllm releases? |
| 1.0 | yes | 4.12 | How is speculative decoding implemented in vllm? |
| 1.0 | yes | 7.53 | What is LangChain and what is it used for? |
| 1.0 | yes | 6.89 | What research exists on molecular ensemble modeling of cyclic peptides? |
| 1.0 | yes | 2.92 | What does recent research say about sycophancy and moral reasoning in LLMs? |
| 1.0 | yes | 2.34 | What does the corpus say about gravure printing quality control automation? |
| 1.0 | — | 1.16 | How do AI regulations relate to practical LLM deployment tools? *(spans two categories)* |
| **0.0** | — | **−8.85** | What is the capital of France? *(nothing in the corpus answers this)* |

Precision averages **1.0** across every answerable query, and the specific
document expected was found in the top five every time. The obscure
gravure-printing query matters here: it rules out the possibility that
search is succeeding on general familiarity with AI topics rather than on
the corpus itself.

**A score-based threshold looked justified by this table, and was wrong.**
A weakly related match scored around −0.4; the one out-of-domain query in
the set scored −8.85. That looked like a clean, wide gap, and a cutoff at
−5.0 was built on it: below that score, decline; above it, answer.

The very first real question tried outside the evaluation set broke it.
"What is AI" — a question the corpus should clearly be able to answer —
scored **−10.04**, lower than the capital-of-France query it was meant to
be distinguished from. The reason is structural, not a bad cutoff value:
reranker scores rank passages within a single query and are not
comparable across different queries. A short, general question scores low
against long, specific technical passages regardless of whether the topic
matches, so the two populations this table implied were separate actually
overlap, and no threshold separates them correctly. A table with one
out-of-domain example was not enough evidence to justify a general rule,
and treating it as sufficient was the mistake.

## Scope decisions

The fix moves the judgement into the review step already in the search
loop: the model is shown the question and the retrieved evidence together
and decides whether the evidence is actually relevant, the same way it
already decides whether there is enough of it. `scope_decisions_correct`
scores this — a query is handled correctly when it is declined if and
only if it is a known gap.

Citation metrics are computed only over questions that were actually
answered. A declined brief has no findings by design, and counting it as
zero citation coverage would penalise the system for behaving correctly.

## Citation faithfulness

Scored over full agent runs:

| Run | Coverage | Validity | Notes |
|---|---|---|---|
| 1 | 0.67 | 1.0 | One of three findings had no citation |
| 2 | 0.67 | 1.0 | Same gap, reproduced independently |
| 3 | 1.0 | 1.0 | All findings cited |

**Validity is 1.0 every time, by construction.** Citations are resolved by
looking up passages that were actually retrieved, so a number the model
invents resolves to nothing and disappears. This is verified rather than
assumed — a regression that broke the guarantee would show up here.

**Coverage is not guaranteed, and that is a real fault.** In two of three
runs the model asserted a claim with no supporting passage, despite the
prompt requiring every claim to cite evidence. It marked those claims
low-confidence, which helps, but a claim with no evidence behind it is a
faithfulness failure regardless of its label. Catching it mechanically is
precisely why this metric exists.

**One wrong citation recurred across runs**, and its cause is specific:
the agent had routed that sub-question to technical literature only, where
the corpus had nothing on the topic, so search returned the least-bad
in-category passage. An unrestricted search for the same question finds a
genuinely relevant document in a different category. This is category
routing plus thin coverage compounding — not a reranker fault. Given the
full candidate pool, the reranker scored that same passage near zero, which
is the correct judgement.

## Latency

Two fixes changed end-to-end time substantially:

| | Time | Model calls |
|---|---|---|
| Before | 5m 07s | 9–10 |
| After | **31.5s** | **6** |

The search step was constructing a fresh embedding and reranking model on
every call instead of reusing one per run, and routing made a separate
model call for each sub-question. Loading the models once per run and
batching the routing into a single call accounts for the difference.

## Running against a rate-limited API

A full agent evaluation makes enough calls to exhaust a free-tier daily
quota. Each query is isolated, so a rate-limit error is recorded against
that query and the run continues and still produces a report, rather than
failing as a whole and discarding everything already computed. Evaluating
against a free tier means planning for this: spread runs out, use a paid
tier, or point `LLM_PROVIDER` at a local model.

## What works, and what does not

Working well:

- Citation validity holds, and the mechanism makes failure structurally
  difficult rather than merely unlikely.
- Retrieval is accurate whenever the corpus actually contains an answer,
  across policy documents, engineering changelogs, and obscure papers alike.
- The model-judged relevance check declines questions the corpus cannot
  answer without also declining ones it can, which the score-based version
  it replaced could not do.
- Failures degrade rather than cascade: a dead URL, an unreachable repo, or
  an exhausted API quota each get recorded while everything else proceeds.

Not working well:

- Claims can still appear without citations, in defiance of the prompt.
  The relevance check does not address this: it catches questions with no
  relevant evidence at all, not a model asserting something unsupported
  while relevant evidence was available.
- Category routing can trap a sub-question in a thinly covered category,
  with no fallback to a wider search. The wrongly-cited passage in the
  example brief was, on its own terms, relevant enough to pass the review
  step - the failure is in routing, which the relevance check does not
  touch.
- Some findings reflect corpus size rather than design, and cannot be
  fully separated without running at larger scale.
- The evaluation query set had exactly one out-of-domain question. That
  was enough to make a broken approach look correct, which is itself a
  finding: a single example is not enough coverage for a check meant to
  generalise, and the query set should grow before this is trusted further.
