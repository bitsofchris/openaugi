"""OpenAI embedding adapter.

Uses OpenAI API for embedding. Requires OPENAI_API_KEY env var.
Default model: text-embedding-3-small (1536 dims).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OpenAIEmbedding:
    """OpenAI API embedding."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.name = model_name
        self._client: Any = None
        self.dimensions: int = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }.get(model_name, 1536)

    def _ensure_client(self) -> None:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai not installed. Run: pip install openaugi[openai]"
                ) from e
            self._client = OpenAI()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenAI API."""
        self._ensure_client()
        response = self._client.embeddings.create(model=self.name, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        self._ensure_client()
        response = self._client.embeddings.create(model=self.name, input=[query])
        return response.data[0].embedding
