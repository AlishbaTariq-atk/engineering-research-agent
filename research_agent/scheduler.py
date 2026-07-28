"""Recurring ingestion.

Sources are refreshed on a fixed interval so the knowledge base stays
current without anyone running commands by hand. The jobs call the same
`ingest_source` used by manual runs, so scheduled and manual refreshes
behave identically.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from research_agent.config import Settings
from research_agent.ingestion import ADAPTERS, ingest_source

logger = logging.getLogger(__name__)


def _run_job(source, settings: Settings) -> None:
    """Refresh one source and log the outcome.

    Exceptions are caught so that one failing source does not stop the
    scheduler from running the others on their next tick.
    """
    try:
        result = ingest_source(source, settings, trigger="scheduled")
        logger.info("Refreshed %s: %s", source.value, result.model_dump_json())
    except Exception as exc:
        logger.error("Scheduled refresh of %s failed: %s", source.value, exc)


def build_scheduler(settings: Settings, interval_hours: int = 24) -> BackgroundScheduler:
    """Create a scheduler with one recurring job per source.

    Nothing here is source-specific, so a new source picked up from the
    adapter registry is scheduled automatically.

    Args:
        settings: Application configuration.
        interval_hours: How often each source is refreshed.

    Returns:
        A scheduler that has been configured but not started.
    """
    scheduler = BackgroundScheduler()
    for source in ADAPTERS:
        scheduler.add_job(
            _run_job,
            trigger=IntervalTrigger(hours=interval_hours),
            args=[source, settings],
            id=f"ingest-{source.value}",
            replace_existing=True,
        )
    return scheduler
