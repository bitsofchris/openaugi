"""Sentence-transformers embedding adapter (default, local, free).

Uses sentence-transformers library for local embedding.
Default model: all-MiniLM-L6-v2 (384 dims, fast, good quality).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding:
    """Local embedding via sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = model_name
        self._model = None
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._ensure_model()
        return self._dimensions  # type: ignore[return-value]

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install openaugi[local]"
                ) from e
            self._model = SentenceTransformer(self.name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded {self.name} (dims={self._dimensions})"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        self._ensure_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        self._ensure_model()
        embedding = self._model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
