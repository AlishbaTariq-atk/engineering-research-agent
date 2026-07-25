from __future__ import annotations

from sentence_transformers import SentenceTransformer

# BAAI/bge-small-en-v1.5 was trained asymmetrically for retrieval: queries
# are expected to carry this instruction prefix, passages/documents are
# not. Skipping this halves the model's effective retrieval quality
# relative to its own benchmarks - it's a model-specific detail easy to
# miss if you only skim "how to use sentence-transformers" and not this
# model's card.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Thin wrapper around sentence-transformers, not langchain-huggingface.
    We need exact control over the query-vs-document asymmetry above, and
    writing it directly is one line clearer than configuring it through an
    extra wrapper layer whose behavior would need explaining anyway."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            BGE_QUERY_INSTRUCTION + text, convert_to_numpy=True, show_progress_bar=False
        ).tolist()
