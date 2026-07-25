# Design Document

## 1. Problem framing and domain choice

The assessment leaves domain choice open but requires it be defensible. I
chose **AI/ML systems** (LLMs, RAG, agents, inference infrastructure) for
three reasons: the source variety is genuinely rich across all three
required categories (arXiv for literature, GitHub for practitioner
knowledge, NIST/EU/US government publications for standards); it's a
domain I can personally judge content relevance for during evaluation,
which matters because judgment quality is exactly what's being assessed;
and it's self-referential in a useful way - the corpus is *about* the kind
of system being built.

**Scope decision**: given the 3-5 day estimate against ingestion + retrieval
+ MCP + agent + eval, I chose to go deep on ingestion, retrieval, and
evaluation - the doc explicitly calls ingestion "the most important
portion," and evaluation is what's most commonly skipped or superficial.
The agent and MCP server are real and fully functional, but intentionally
smaller in surface area (one graph, four MCP tools) rather than expanded
further. This is a deliberate tradeoff, not an oversight.

## 2. Architecture

```
CLI ──┐
      ├──►  Service layer (plain Python)  ──►  LangGraph agent  ──►  Retriever  ──►  Chroma + SQLite
MCP ──┘
```

No HTTP API. The assessment requires a locally runnable system with an MCP
server; it does not require a web interface. A service layer (plain Python
functions/classes) gives the same separation-of-concerns benefit as a REST
API without a network hop, another process to keep alive during a live
demo, or FastAPI request/response schemas duplicating the Pydantic schema
that already exists. FastAPI would be a straightforward addition later if
browser access became a real requirement - the point is not building it
before anything needs it.

## 3. Data engineering

**Schema** (`research_agent/models/`): one `Document` type every source
adapter maps into, one `Chunk` type that copies (not references) the
parent fields retrieval needs. Two decisions carry the most weight:

- `doc_id` is `sha256(source:source_id)`, not a random UUID, so
  re-ingesting the same item is a natural upsert rather than a separate
  duplicate-detection problem.
- `content_hash` (of normalized text) drives idempotency and versioning:
  unchanged → skip; changed → bump `version`, write an append-only
  `document_versions` audit row. arXiv paper revisions (v1→v2) are handled
  as a content_hash change on the *same* doc_id, not a new document -
  version tracking doing exactly the job it was built for on a real case,
  not a hypothetical one.

**Storage strategy**: full text is expensive to fetch, parse, and embed at
scale, so `storage_mode` (`full_text` / `abstract_only` / `metadata_only`)
is decided per-source based on what's actually cheap for that source:
GitHub release notes and READMEs are already plain text from the API, so
always `full_text`; arXiv fetches abstracts for every matching paper but
only downloads/parses PDF full text for a curated recent subset; RSS
follows whatever the publisher's own feed exposes (some publish full
content, some only a title); gov/standards documents are few enough in
number that all are stored as `full_text`. `metadata_only` documents are
excluded from chunking/embedding entirely - they exist for corpus
stats/freshness only.

**Four sources, deliberately not uniform**: arXiv and GitHub have real
paginated discovery APIs; RSS has no pagination for older entries (a real,
documented limitation, not one hidden by a workaround); gov/standards
bodies have no discovery API at all, so that source is a curated,
config-driven JSON list of document URLs, not a crawler - "backfill" for
that source means adding a URL to the list, not paging through history.

**Pipeline** (`ingestion/pipeline.py`): one generic upsert+log loop shared
by all four adapters, so idempotency/versioning/duplicate-detection/
logging can't drift between sources. Commits per-item, so a mid-run
network failure loses nothing already ingested. Cross-source duplicate
detection (same content, different `source_id`) is a content_hash lookup
distinct from idempotency (same `source_id` re-fetched, handled by
`doc_id`). Failures are recorded per-item in a queryable `ingestion_failures`
table, not just logged to console - found to matter in practice when an
adapter's per-item error handling only logged a warning and a broken item
silently vanished from the run's stats instead of counting as failed.

## 4. Retrieval

**Chunking**: 1000 chars / 150 overlap, recursive split on paragraph→
sentence boundaries (`langchain_text_splitters`, used as a plain utility,
not business logic), uniform across sources. A documented simplification -
per-category tuning is a natural next step once eval data justifies it,
not before.

**Embeddings**: `BAAI/bge-small-en-v1.5` via a thin direct wrapper over
`sentence-transformers`, not `langchain-huggingface`. bge-small was
trained asymmetrically for retrieval - queries need an instruction prefix,
indexed passages don't - and that behavior needed to be exact and visible
in our own code rather than configured through an extra wrapper layer.

**Vector store**: ChromaDB, one collection per category. This is what
makes retrieval category-aware structurally - a search scoped to one
category never touches another category's vectors, rather than searching
everything and filtering after.

**Reranking**: over-fetch per category via fast vector similarity, pool,
rerank the small pool with a cross-encoder (`ms-marco-MiniLM-L-6-v2`,
CPU-friendly), truncate to top-k.

**Freshness filtering**: checked live against SQLite (`status` column),
not a copy baked into Chroma metadata - a document can be marked
stale/superseded without its content changing, so a metadata copy could
silently drift out of date.

## 5. Agent design

Five distinct reasoning steps, not a retrieve-and-summarize pass:
**planner** (decompose into 2-4 sub-questions) → **retrieve** (category-
routed search, pooled/deduped across sub-questions and iterations) →
**critic** (judge whether evidence is sufficient; if not, its gaps become
the next round's sub-questions - this is the iterative-retrieval loop,
capped at 2 rounds) → **synthesizer** (claims with mandatory citation IDs,
a dedicated conflicts list for cross-source disagreement) →
**report_generator** (deliberately *not* an LLM call - plain Python that
resolves citation IDs against the actual retrieved chunks, so a citation
can be looked up or dropped, never invented).

**LangGraph vs. a hand-rolled loop**: the workflow is genuinely graph-
shaped (a conditional loop back to retrieval), so LangGraph owns control
flow/state-passing. Every planning, routing, and synthesis *decision* is
our own prompt and Pydantic schema - no prebuilt LangChain agent or
`AgentExecutor` is used anywhere, specifically so every reasoning step
stays inspectable and explainable rather than delegated to a framework's
internal loop.

**LLM provider**: chosen by config (`llm_provider: groq | ollama`), both
plain LangChain chat models via `.with_structured_output()`. Groq for
demo-time speed, Ollama as the offline/zero-external-dependency fallback.
This is the one place LangChain is used as originally planned - a thin
integration layer, not business logic.

## 6. MCP server

Four tools via the official Python SDK (`FastMCP`, stdio transport):
`search_knowledge_base` (semantic search with category/date filtering and
stale-exclusion), `get_document` (by ID), `corpus_stats`, and
`source_freshness` (read live from `ingestion_runs`, not a separate table).
See `mcp_config.example.json` for how any MCP client connects.

## 7. Evaluation

See `docs/EVALUATION.md` for the full report. Two categories of metric:
retrieval precision (against a 12-query set grounded in the actual
ingested corpus, including deliberate "known gap" queries to test honest
abstention) and citation faithfulness (coverage/validity, directly
targeting a real bug found during testing - a synthesized claim with zero
supporting citations despite an explicit prompt instruction against it).

## 8. Scope and known limitations

Stated plainly rather than discovered by the reader:

- **Corpus scale**: development and testing used a small sample (~1800
  documents, not the ~50k target) to keep iteration fast and API costs
  bounded. The ingestion pipeline itself has no scale ceiling built in -
  `--max-results`/pagination are the only things standing between this
  and a full run - but a small, somewhat arbitrary arXiv sample (15 most-
  recent papers across three categories) means `technical_literature` has
  genuinely sparse coverage of systems/inference topics specifically, which
  is directly responsible for the weakest result in the evaluation report.
- **RSS backfill** is bounded by whatever a publisher's feed currently
  exposes (typically last 10-50 posts) - no pagination for older entries.
- **RSS title-only feeds** (e.g. Hugging Face's blog) produce
  `metadata_only` documents rather than a fabricated abstract; scraping the
  linked article page for a real excerpt would fix this but adds a
  per-entry HTTP request and a new failure mode.
- **Gov/standards source** is a curated list of specific URLs, not a
  crawler - by design (no discovery API exists), but it means coverage is
  only as broad as the list, currently three documents.
- **Category-restricted retrieval** can force a bad match when the
  "correct" category has sparse coverage for a given query - the router
  commits to categories before retrieval runs, so a genuinely relevant
  document in a different category is invisible to that sub-question's
  search. Confirmed directly during evaluation (see EVALUATION.md).
- **Groq free-tier rate limits** (100k tokens/day) are a real constraint
  encountered during testing, not theoretical - the batched-routing fix
  (one LLM call per retrieval round instead of one per sub-question) was
  made specifically to reduce exposure to this, and evaluation runs budget
  for it explicitly (see EVALUATION.md).

## 9. What I'd improve with more time

1. Scale the corpus toward the ~50k target and re-run evaluation - several
   of the current findings (sparse technical_literature coverage
   specifically) are corpus-scale artifacts, not fixed properties of the
   design, and the evaluation report can't fully distinguish "the design is
   wrong" from "the test corpus is small" until this happens.
2. A confidence threshold on reranker score below which the agent reports
   "no relevant evidence" instead of forcing a citation - the evaluation
   run's out-of-domain control query scored dramatically lower
   (-8.845) than any in-domain query, suggesting the score is a genuinely
   usable signal for this that isn't being used yet.
3. Per-category chunk size tuning, once eval data (not intuition) justifies
   it over the current uniform 1000/150 split.
4. A real fallback path when the router's category restriction produces a
   sparse result set - e.g., falling back to an unrestricted search when
   the restricted one returns very few or very low-scoring candidates.
