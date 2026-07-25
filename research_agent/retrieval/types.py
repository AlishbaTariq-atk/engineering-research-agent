from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    title: str
    canonical_url: str
    source: str
    category: str
    publication_date: str | None
    score: float
    reranked: bool
