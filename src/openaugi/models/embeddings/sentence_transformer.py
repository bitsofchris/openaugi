"""Sentence-transformers embedding adapter (default, local, free).

Uses sentence-transformers library for local embedding.
Default model: all-MiniLM-L6-v2 (384 dims, fast, good quality).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Known dimensions for common models (avoids loading model just to get dims)
_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
}


# Known max tokens for common models
_KNOWN_MAX_TOKENS: dict[str, int] = {
    "all-MiniLM-L6-v2": 256,
    "all-mpnet-base-v2": 384,
    "all-MiniLM-L12-v2": 256,
}


class SentenceTransformerEmbedding:
    """Local embedding via sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.name = model_name
        self.dimensions: int = _KNOWN_DIMS.get(model_name, 384)
        self.max_tokens: int = _KNOWN_MAX_TOKENS.get(model_name, 256)
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                    SentenceTransformer,
                )
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install openaugi[local]"
                ) from e
            self._model = SentenceTransformer(self.name)
            self.dimensions = self._model.get_sentence_embedding_dimension()
            self.max_tokens = self._model.max_seq_length
            logger.info(
                f"Loaded {self.name} (dims={self.dimensions}, max_tokens={self.max_tokens})"
            )

    def truncate(self, text: str) -> str:
        """Truncate text to fit within the model's token limit."""
        self._ensure_model()
        tokenizer = self._model.tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self.max_tokens:
            return text
        return tokenizer.decode(tokens[: self.max_tokens])

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
