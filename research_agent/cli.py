from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from research_agent.config import Settings
from research_agent.models import SourceName
from research_agent.scheduler import ADAPTERS, build_scheduler, run_source
from research_agent.storage import init_db


def _fetch_kwargs_for(source: SourceName, args: argparse.Namespace) -> dict:
    """--max-results and --start-offset only exist on arxiv.fetch()'s
    signature (the other three adapters have no equivalent concept - a
    GitHub repo list or a curated standards-doc list isn't "capped" the
    same way a paginated query is). Forwarding them unconditionally to
    every adapter would raise TypeError; a shared **kwargs catch-all on
    every adapter would silently swallow a typo'd flag instead. So the CLI
    decides per source which flags are even meaningful."""
    if source != SourceName.ARXIV:
        return {}
    kwargs = {}
    if args.max_results is not None:
        kwargs["max_results"] = args.max_results
    if args.start_offset is not None:
        kwargs["start_offset"] = args.start_offset
    return kwargs


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    init_db(settings.sqlite_path)
    sources = list(ADAPTERS) if args.source == "all" else [SourceName(args.source)]
    trigger = "backfill" if args.backfill else "manual"

    for source in sources:
        result = run_source(source, settings, trigger=trigger, **_fetch_kwargs_for(source, args))
        print(result.model_dump_json(indent=2))


def _cmd_index(args: argparse.Namespace, settings: Settings) -> None:
    # Imported lazily: this pulls in sentence-transformers, which only the
    # index/agent/eval commands need. Also means main() gets to set
    # HF_HUB_OFFLINE (below) before this import chain ever touches
    # huggingface_hub, not after.
    from research_agent.retrieval import run_indexing

    init_db(settings.sqlite_path)
    result = run_indexing(settings, limit=args.limit)
    print(result)


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    import json

    from research_agent.agent import run_research_agent

    report = run_research_agent(args.question, settings, max_iterations=args.max_iterations)
    print(json.dumps(report, indent=2, default=str))


def _cmd_scheduler(args: argparse.Namespace, settings: Settings) -> None:
    init_db(settings.sqlite_path)
    scheduler = build_scheduler(settings, interval_hours=args.interval_hours)
    scheduler.start()
    print(f"Scheduler running - refreshing every {args.interval_hours}h across all sources. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Run ingestion for one source or all sources (manual refresh / backfill)")
    p_ingest.add_argument("source", choices=[*(s.value for s in ADAPTERS), "all"])
    p_ingest.add_argument("--backfill", action="store_true", help="Log this run's trigger as 'backfill'")
    p_ingest.add_argument("--start-offset", type=int, default=None, help="arXiv only: page offset for deeper history")
    p_ingest.add_argument("--max-results", type=int, default=None, help="Cap on documents fetched this run")

    p_index = sub.add_parser("index", help="Chunk/embed/index documents due for (re)indexing into Chroma")
    p_index.add_argument("--limit", type=int, default=None, help="Cap on documents indexed this run")

    p_ask = sub.add_parser("ask", help="Run the research agent end-to-end and print a research brief")
    p_ask.add_argument("question")
    p_ask.add_argument("--max-iterations", type=int, default=2)

    p_sched = sub.add_parser("scheduler", help="Run the recurring ingestion scheduler in the foreground")
    p_sched.add_argument("--interval-hours", type=int, default=24)

    return parser


def main() -> None:
    # Always logs to a fixed file, not just the console - so a run started
    # from any terminal (or by an agent, backgrounded) can be watched live
    # from any other terminal with `tail -f data/logs/research-agent.log`,
    # not just wherever the process happened to be launched from.
    log_path = Path("data/logs/research-agent.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    args = build_parser().parse_args()
    settings = Settings()
    if settings.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.command == "ingest":
        _cmd_ingest(args, settings)
    elif args.command == "index":
        _cmd_index(args, settings)
    elif args.command == "ask":
        _cmd_ask(args, settings)
    elif args.command == "scheduler":
        _cmd_scheduler(args, settings)


if __name__ == "__main__":
    main()
