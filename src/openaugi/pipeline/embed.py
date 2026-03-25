"""Embedding pipeline step — batch embed blocks via EmbeddingModel.

Processes blocks where embedding IS NULL. Stores embeddings as BLOBs.
Builds FAISS index from all embedded blocks.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from openaugi.model.protocols import EmbeddingModel
from openaugi.store.faiss import FaissIndex
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_CHARS = 24_000  # ~8k tokens conservative


def _truncate(content: str) -> str:
    if len(content) > MAX_CHARS:
        return content[:MAX_CHARS]
    return content


def run_embed(
    store: SQLiteStore,
    model: EmbeddingModel,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed all blocks that need embeddings. Returns count embedded."""
    blocks = store.get_blocks_needing_embeddings(kind="entry")
    if not blocks:
        logger.info("All entry blocks already have embeddings")
        return 0

    total_to_embed = len(blocks)
    logger.info(f"Embedding {total_to_embed} blocks (model: {model.name})")
    start = time.time()
    total = 0

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i : i + batch_size]
        texts = [_truncate(b.content or "") for b in batch]
        block_ids = [b.id for b in batch]

        try:
            vectors = model.embed_texts(texts)
            embeddings = {}
            for bid, vec in zip(block_ids, vectors, strict=True):
                blob = np.array(vec, dtype=np.float32).tobytes()
                embeddings[bid] = blob
            store.update_embeddings(embeddings)
            total += len(embeddings)
        except Exception as e:
            logger.warning(f"Batch {i // batch_size + 1} failed: {e}")
            # Retry individually
            for bid, text in zip(block_ids, texts, strict=True):
                try:
                    vec = model.embed_texts([text])[0]
                    blob = np.array(vec, dtype=np.float32).tobytes()
                    store.update_embeddings({bid: blob})
                    total += 1
                except Exception as e2:
                    logger.warning(f"Block {bid} failed: {e2}")

        logger.info(f"Embedded {total} / {total_to_embed}")

    elapsed = time.time() - start
    logger.info(f"Done embedding {total} blocks in {elapsed:.1f}s")
    return total


def build_faiss_index(store: SQLiteStore, dim: int | None = None) -> FaissIndex:
    """Build FAISS index from all embedded entry blocks."""
    blocks = store.get_blocks_with_embeddings(kind="entry")
    if not blocks:
        logger.warning("No embedded blocks found")
        return FaissIndex()

    block_ids = [b.id for b in blocks]
    embeddings = [b.embedding for b in blocks if b.embedding]

    if dim is None and embeddings:
        dim = len(np.frombuffer(embeddings[0], dtype=np.float32))

    index = FaissIndex()
    index.build(block_ids, embeddings, dim or 384)
    return index
