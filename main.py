"""Entry point for the research assistant.

Run without arguments for an interactive session:

    python main.py

Or run a maintenance command directly:

    python main.py ingest arxiv --max-results 500
    python main.py index
    python main.py stats
    python main.py eval
    python main.py schedule
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time

from research_agent.config import Settings
from research_agent.models import SourceName
from research_agent.storage import connect, init_db

BANNER = """
Research Assistant
Ask a technical question and get an evidence-backed brief.
Type 'exit' or press Ctrl-C to quit.
"""


# --------------------------------------------------------------------------
# Output formatting
# --------------------------------------------------------------------------


def _wrap(text: str, indent: str = "  ") -> str:
    """Wrap text to a readable width for the terminal."""
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def print_out_of_scope(brief: dict) -> None:
    """Explain that the knowledge base has nothing on this subject.

    Shows the nearest passages found and how far short they fell, so the
    refusal is transparent rather than a blank wall.

    Args:
        brief: A brief with `out_of_scope` set.
    """
    print("\n" + "=" * 88)
    print("OUT OF SCOPE")
    print("=" * 88)
    print(_wrap(brief["executive_summary"]))

    if brief.get("nearest_matches"):
        print("\nClosest matches found (all below the relevance threshold):")
        for match in brief["nearest_matches"]:
            print(f"  {match['score']:>7.2f}  {match['title'][:64]}  ({match['source']})")

    print("\nTry asking about:")
    for suggestion in brief["follow_up_questions"]:
        print(_wrap(f"- {suggestion}"))
    print()


def print_brief(brief: dict) -> None:
    """Print a research brief in readable form.

    Args:
        brief: The brief returned by the agent.
    """
    if brief.get("out_of_scope"):
        print_out_of_scope(brief)
        return

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(_wrap(brief["executive_summary"]))

    print("\nKEY FINDINGS")
    print("-" * 88)
    for number, finding in enumerate(brief["key_findings"], start=1):
        print(f"\n{number}. [{finding['confidence']} confidence]")
        print(_wrap(finding["claim"], indent="   "))
        if finding["citations"]:
            for citation in finding["citations"]:
                print(f"      [{citation['n']}] {citation['title']}")
        else:
            print("      (no supporting source cited)")

    if brief["conflicts"]:
        print("\nCONFLICTING EVIDENCE")
        print("-" * 88)
        for conflict in brief["conflicts"]:
            print(_wrap(conflict["description"]))
            for citation in conflict["citations"]:
                print(f"      [{citation['n']}] {citation['title']}")

    print("\nSOURCES")
    print("-" * 88)
    for citation in brief["citations"]:
        date = citation["publication_date"] or "n.d."
        print(f"  [{citation['n']}] {citation['title']}")
        print(f"      {citation['source']}, {date} - {citation['url']}")

    print(f"\nOVERALL CONFIDENCE: {brief['confidence']}")

    print("\nKNOWLEDGE GAPS")
    print("-" * 88)
    for gap in brief["knowledge_gaps"]:
        print(_wrap(f"- {gap}"))

    print("\nSUGGESTED FOLLOW-UPS")
    print("-" * 88)
    for question in brief["follow_up_questions"]:
        print(_wrap(f"- {question}"))
    print()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_chat(settings: Settings) -> None:
    """Run the interactive question-and-answer session.

    Args:
        settings: Application configuration.
    """
    from research_agent.agent import answer_question

    conn = connect(settings.sqlite_path)
    try:
        indexed = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE last_indexed_version IS NOT NULL"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()

    print(BANNER)
    print(f"Knowledge base: {total} documents, {indexed} indexed for search.\n")

    if indexed == 0:
        print("Nothing is indexed yet. Run 'python main.py ingest all' then 'python main.py index'.\n")
        return

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        started = time.perf_counter()
        try:
            brief = answer_question(question, settings)
        except Exception as exc:
            print(f"\nCould not answer that: {exc}\n")
            continue

        print_brief(brief)
        print(f"({time.perf_counter() - started:.1f}s)\n")


def cmd_ingest(settings: Settings, args: argparse.Namespace) -> None:
    """Fetch documents from one source or all of them."""
    from research_agent.ingestion import ADAPTERS, ingest_source

    init_db(settings.sqlite_path)
    sources = list(ADAPTERS) if args.source == "all" else [SourceName(args.source)]
    trigger = "backfill" if args.backfill else "manual"

    for source in sources:
        # Only the arXiv adapter paginates, so only it takes these options.
        options = {}
        if source == SourceName.ARXIV:
            if args.max_results is not None:
                options["max_results"] = args.max_results
            if args.start_offset is not None:
                options["start_offset"] = args.start_offset
            if args.full_text_limit is not None:
                options["full_text_limit"] = args.full_text_limit

        print(f"Ingesting {source.value}...")
        result = ingest_source(source, settings, trigger=trigger, **options)
        print(result.model_dump_json(indent=2))


def cmd_index(settings: Settings, args: argparse.Namespace) -> None:
    """Embed and index everything that is new or has changed."""
    from research_agent.retrieval import run_indexing

    init_db(settings.sqlite_path)
    print("Indexing (this loads the embedding model and may take a few minutes)...")
    print(json.dumps(run_indexing(settings, limit=args.limit), indent=2))


def cmd_stats(settings: Settings) -> None:
    """Show what the knowledge base contains and when it was last updated."""
    from research_agent import storage

    init_db(settings.sqlite_path)
    conn = connect(settings.sqlite_path)
    try:
        print(json.dumps(storage.corpus_stats(conn), indent=2))
        print("\nSource freshness:")
        print(json.dumps(storage.source_freshness(conn), indent=2))
    finally:
        conn.close()


def cmd_eval(settings: Settings, args: argparse.Namespace) -> None:
    """Run the evaluation suite and write a report."""
    from pathlib import Path

    from research_agent.evaluation import run_evaluation

    result = run_evaluation(settings, include_agent=not args.skip_agent)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))

    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull results written to {output}")


def cmd_schedule(settings: Settings, args: argparse.Namespace) -> None:
    """Refresh every source on a repeating interval until interrupted."""
    from research_agent.scheduler import build_scheduler

    init_db(settings.sqlite_path)
    scheduler = build_scheduler(settings, interval_hours=args.interval_hours)
    scheduler.start()
    print(f"Refreshing all sources every {args.interval_hours}h. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("\nStopped.")


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Research assistant. Run with no arguments for an interactive session.",
    )
    commands = parser.add_subparsers(dest="command")

    ingest = commands.add_parser("ingest", help="Fetch documents from a source")
    ingest.add_argument("source", choices=[source.value for source in SourceName] + ["all"])
    ingest.add_argument("--backfill", action="store_true", help="Record this run as a historical backfill")
    ingest.add_argument("--max-results", type=int, help="arXiv only: how many papers to fetch")
    ingest.add_argument("--start-offset", type=int, help="arXiv only: skip this many results, to reach older papers")
    ingest.add_argument("--full-text-limit", type=int, help="arXiv only: how many papers also get their PDF")

    index = commands.add_parser("index", help="Embed and index new or changed documents")
    index.add_argument("--limit", type=int, help="Stop after this many documents")

    commands.add_parser("stats", help="Show corpus contents and source freshness")

    evaluate = commands.add_parser("eval", help="Run the evaluation suite")
    evaluate.add_argument("--skip-agent", action="store_true", help="Retrieval only; makes no model calls")
    evaluate.add_argument("--output", default="data/eval_report.json")

    schedule = commands.add_parser("schedule", help="Refresh sources on a repeating interval")
    schedule.add_argument("--interval-hours", type=int, default=24)

    return parser


def main() -> None:
    """Parse arguments and run the requested command."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    args = build_parser().parse_args()
    settings = Settings()

    if settings.hf_hub_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.command is None:
        cmd_chat(settings)
    elif args.command == "ingest":
        cmd_ingest(settings, args)
    elif args.command == "index":
        cmd_index(settings, args)
    elif args.command == "stats":
        cmd_stats(settings)
    elif args.command == "eval":
        cmd_eval(settings, args)
    elif args.command == "schedule":
        cmd_schedule(settings, args)


if __name__ == "__main__":
    sys.exit(main())
