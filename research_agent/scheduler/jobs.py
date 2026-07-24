from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from research_agent.config import Settings
from research_agent.ingestion import arxiv, github, rss, standards
from research_agent.ingestion.pipeline import IngestionResult, run_ingestion
from research_agent.models import SourceName

logger = logging.getLogger(__name__)

ADAPTERS = {
    SourceName.ARXIV: arxiv.fetch,
    SourceName.GITHUB: github.fetch,
    SourceName.RSS_BLOG: rss.fetch,
    SourceName.GOV_STANDARDS: standards.fetch,
}


def run_source(
    source: SourceName,
    settings: Settings,
    trigger: str = "scheduled",
    **fetch_kwargs,
) -> IngestionResult:
    """The one function every entrypoint calls - a cron job, a CLI manual
    refresh, and a CLI backfill are three different *callers* of this same
    code path (different trigger label, different fetch_kwargs such as
    arXiv's start_offset for backfill), never three separate
    implementations that could drift from each other."""
    adapter = ADAPTERS[source]
    documents = adapter(settings, **fetch_kwargs)
    result = run_ingestion(source, documents, str(settings.sqlite_path), trigger=trigger)
    logger.info("ingestion run source=%s trigger=%s result=%s", source.value, trigger, result.model_dump_json())
    return result


def build_scheduler(settings: Settings, interval_hours: int = 24) -> BackgroundScheduler:
    """One recurring job per source, all on the same configurable
    interval. Nothing here is source-specific: adding a fifth source later
    means adding one adapter module and one line in ADAPTERS, not touching
    the scheduler."""
    scheduler = BackgroundScheduler()
    for source in ADAPTERS:
        scheduler.add_job(
            run_source,
            trigger=IntervalTrigger(hours=interval_hours),
            args=[source, settings],
            kwargs={"trigger": "scheduled"},
            id=f"ingest-{source.value}",
            replace_existing=True,
        )
    return scheduler
