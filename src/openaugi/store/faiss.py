"""FAISS vector index wrapper.

Builds an in-memory FAISS IndexFlatIP (inner product = cosine on normalized vectors).
Same approach as v1's query_engine.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """In-memory FAISS index for semantic search over block embeddings."""

    def __init__(self):
        self._index: Any = None
        self._block_ids: list[str] = []
        self._dim: int = 0

    @property
    def size(self) -> int:
        return len(self._block_ids)

    def build(self, block_ids: list[str], embeddings: list[bytes], dim: int) -> None:
        """Build index from block IDs and embedding blobs.

        Args:
            block_ids: Parallel list of block IDs.
            embeddings: Parallel list of embedding bytes (float32).
            dim: Embedding dimension.
        """
        try:
            import faiss  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install openaugi[local]"
            ) from e

        if not block_ids:
            logger.warning("No embeddings to build index from")
            self._index = None
            self._block_ids = []
            return

        self._dim = dim
        self._block_ids = list(block_ids)

        # Convert blobs to numpy matrix
        vectors = []
        for blob in embeddings:
            vec = np.frombuffer(blob, dtype=np.float32)
            vectors.append(vec)

        matrix = np.array(vectors, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / (norms + 1e-10)

        # Build flat inner-product index
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(matrix)

        logger.info(f"FAISS index built: {len(block_ids)} vectors, dim={dim}")

    def search(
        self,
        query_embedding: list[float],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_embedding: Query vector (will be normalized).
            k: Number of results.

        Returns:
            List of (block_id, score) tuples, sorted by score descending.
        """
        if self._index is None or not self._block_ids:
            return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query)
        query = query / (norm + 1e-10)

        search_k = min(k, len(self._block_ids))
        distances, indices = self._index.search(query, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append((self._block_ids[idx], float(dist)))

        return results

    def save(self, path: str | Path) -> None:
        """Save FAISS index to disk."""
        if self._index is None:
            return
        try:
            import faiss  # type: ignore[import]

            faiss.write_index(self._index, str(path))
            logger.info(f"FAISS index saved to {path}")
        except ImportError:
            logger.warning("faiss-cpu not available, cannot save index")

    def load(self, path: str | Path, block_ids: list[str]) -> None:
        """Load FAISS index from disk."""
        try:
            import faiss  # type: ignore[import]

            self._index = faiss.read_index(str(path))
            self._block_ids = list(block_ids)
            self._dim = self._index.d
            logger.info(
                f"FAISS index loaded from {path}: "
                f"{self._index.ntotal} vectors, dim={self._dim}"
            )
        except ImportError as e:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install openaugi[local]"
            ) from e
