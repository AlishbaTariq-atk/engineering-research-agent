# Engineering Research Agent

An AI research platform that ingests technical content from multiple public
sources (arXiv, GitHub, RSS/engineering blogs, and government/standards
publications) into a structured, versioned knowledge base, and answers
research questions through an agentic retrieval workflow, producing
evidence-backed briefs with citations, confidence assessment, and identified
knowledge gaps.

Built for an AI Engineering Practical Assessment (domain: AI/ML systems).
See [docs/DESIGN.md](docs/DESIGN.md) for the full design rationale and
tradeoffs, [docs/EVALUATION.md](docs/EVALUATION.md) for the evaluation
report, and [docs/EXAMPLE_BRIEF.md](docs/EXAMPLE_BRIEF.md) for a real
end-to-end research brief.

## Architecture

```
CLI (research-agent) ──┐
                        ├──►  Service layer (plain Python)  ──►  LangGraph agent  ──►  Retriever  ──►  Chroma + SQLite
MCP Server ─────────────┘                                          (plan/route/retrieve/critique/synthesize)
                                                                                          │
                                                          ┌───────────────┬───────────────┼───────────────┐
                                                     arXiv API      GitHub API      RSS feeds      curated gov/standards docs
```

No HTTP API / FastAPI: the CLI and MCP server both call the same in-process
service layer directly (see docs/DESIGN.md for why).

## Setup

Requires Python 3.12+.

```bash
git clone <this-repo>
cd Engineering_Research_Agent
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

cp .env.example .env
# Fill in GROQ_API_KEY (free tier: console.groq.com) - required unless
# LLM_PROVIDER=ollama. GITHUB_TOKEN is optional (raises the GitHub API
# rate limit from 60/hr to 5000/hr). Once models are cached locally after
# a first run, set HF_HUB_OFFLINE=true to skip ~40s of redundant
# HuggingFace cache-verification requests on every startup.
```

All commands log to `data/logs/research-agent.log` in addition to the
console - watch progress live from any terminal with:

```bash
tail -f data/logs/research-agent.log
```

## Usage

```bash
# Ingest a small sample from every source (drop --max-results for a full run)
.venv/bin/research-agent ingest arxiv --max-results 50
.venv/bin/research-agent ingest github
.venv/bin/research-agent ingest rss_blog
.venv/bin/research-agent ingest gov_standards

# Historical backfill (arXiv only - pages deeper into history)
.venv/bin/research-agent ingest arxiv --backfill --start-offset 5000 --max-results 1000

# Chunk + embed + index everything not yet indexed
.venv/bin/research-agent index

# Ask the research agent a question end-to-end
.venv/bin/research-agent ask "What are the tradeoffs of speculative decoding for LLM inference?"

# Run the evaluation suite
.venv/bin/research-agent eval               # full: retrieval + agent (real LLM calls)
.venv/bin/research-agent eval --skip-agent  # retrieval-only, no LLM calls/API cost

# Run the recurring scheduler (all sources, every 24h by default)
.venv/bin/research-agent scheduler --interval-hours 24
```

## MCP server

Exposes `search_knowledge_base`, `get_document`, `corpus_stats`, and
`source_freshness` to any MCP-compatible client over stdio.

```bash
.venv/bin/research-agent-mcp
```

See [mcp_config.example.json](mcp_config.example.json) for how to connect
a client (e.g. Claude Desktop) - fill in the absolute path to this
project's `.venv/bin/python`.

## Project layout

```
research_agent/
  models/        Document/Chunk schema (Pydantic) - the one shared shape every source maps into
  storage/       SQLite schema + connection helper
  ingestion/     One adapter per source (arxiv, github, rss, standards) + shared pipeline
  scheduler/     APScheduler wiring - scheduled/manual/backfill are one code path
  retrieval/     Chunking, embeddings, Chroma indexing, reranking, context construction
  agent/         LangGraph agent: planner, router, critic, synthesizer, report_generator
  mcp_server/    MCP tool server
  evaluation/    Query set + metrics + eval runner
  cli.py         Single CLI entrypoint for all of the above
docs/            Design document, evaluation report, example brief
```

## Status

All 5 parts (ingestion, retrieval, MCP server, agent, evaluation) are
implemented and verified end-to-end against real data and a live LLM
backend - see docs/EVALUATION.md for what was actually measured, including
what didn't work well.
