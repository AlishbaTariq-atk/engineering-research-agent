from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_agent.models import Document, SourceCategory, SourceName, StorageMode
from research_agent.storage import init_db


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """A real (temp file) SQLite DB, not an in-memory mock - the schema's
    foreign keys/constraints and pipeline.py's actual SQL should be
    exercised for real, matching the rest of this project's preference for
    integration-style checks over mocked-out unit tests."""
    path = str(tmp_path / "test_kb.db")
    init_db(path)
    return path


def make_document(**overrides) -> Document:
    """A minimally valid Document with sensible defaults, so each test
    only specifies the fields it actually cares about."""
    now = datetime.now(UTC)
    defaults = dict(
        doc_id="test-doc-id-000000000000",
        source=SourceName.ARXIV,
        source_id="2401.00001",
        category=SourceCategory.TECHNICAL_LITERATURE,
        canonical_url="https://arxiv.org/abs/2401.00001",
        title="A Test Paper",
        abstract="This is a test abstract about testing things.",
        full_text=None,
        tags=["cs.AI"],
        publication_date=None,
        ingested_at=now,
        last_checked_at=now,
        updated_at=now,
        content_hash="0" * 64,
        version=1,
        storage_mode=StorageMode.ABSTRACT_ONLY,
        source_metadata={},
    )
    defaults.update(overrides)
    return Document(**defaults)
