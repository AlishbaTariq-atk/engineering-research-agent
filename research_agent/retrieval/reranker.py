from __future__ import annotations

from sentence_transformers import CrossEncoder

from .types import RetrievedChunk

# A cross-encoder scores the (query, passage) pair jointly - slower than
# the bi-encoder similarity used for the first pass (exactly why it only
# runs over a small over-fetched candidate pool, never the whole index),
# but it can weigh query/passage term interactions that two independently
# computed embedding vectors can't capture. Whether this actually improves
# precision here (not just in general) is measured directly in the
# evaluation framework, not assumed.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            candidate.reranked = True
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
