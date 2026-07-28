"""Shared test fixtures and helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_agent.models import Document, SourceCategory, SourceName, StorageMode
from research_agent.storage import init_db


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A temporary database with the real schema applied.

    Uses an actual SQLite file rather than a stand-in, so constraints and
    the pipeline's real SQL are exercised.

    Returns:
        Path to a fresh, empty database.
    """
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def make_document(**overrides) -> Document:
    """Build a valid Document, overriding only the fields a test cares about.

    Args:
        **overrides: Any Document field to replace.

    Returns:
        A Document ready to pass into the pipeline.
    """
    now = datetime.now(UTC)
    fields = {
        "doc_id": "test000000000000000000000",
        "source": SourceName.ARXIV,
        "source_id": "2401.00001",
        "category": SourceCategory.TECHNICAL_LITERATURE,
        "canonical_url": "https://arxiv.org/abs/2401.00001",
        "title": "A Test Paper",
        "abstract": "An abstract used for testing.",
        "full_text": None,
        "tags": ["cs.AI"],
        "publication_date": None,
        "ingested_at": now,
        "last_checked_at": now,
        "updated_at": now,
        "content_hash": "0" * 64,
        "version": 1,
        "storage_mode": StorageMode.ABSTRACT_ONLY,
        "source_metadata": {},
    }
    fields.update(overrides)
    return Document(**fields)
