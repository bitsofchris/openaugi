"""Embedding pipeline step — batch embed blocks via EmbeddingModel.

Processes blocks where embedding IS NULL. Stores embeddings as BLOBs
in the blocks table and in the sqlite-vec vec_blocks virtual table.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from openaugi.model.protocols import EmbeddingModel
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _is_bad_request(exc: Exception) -> bool:
    """Check if an exception is a 400-class error (permanent, not retryable)."""
    # openai.BadRequestError, httpx.HTTPStatusError with 4xx, etc.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and 400 <= status < 500:
        return True
    # Walk the exception chain
    if isinstance(exc.__cause__, Exception):
        return _is_bad_request(exc.__cause__)
    return False


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

    # Ensure vec_blocks table exists with the correct dimension before writing
    store.ensure_vec_table(model.dimensions)

    total_to_embed = len(blocks)
    logger.info(f"Embedding {total_to_embed} blocks (model: {model.name})")
    start = time.time()
    total = 0

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i : i + batch_size]
        texts = [model.truncate(b.content or "") for b in batch]
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
            if _is_bad_request(e):
                logger.warning(f"Batch {i // batch_size + 1} rejected (400): {e}")
                continue
            logger.warning(f"Batch {i // batch_size + 1} failed: {e}")
            # Retry individually only for non-400 errors (e.g. rate limits)
            for bid, text in zip(block_ids, texts, strict=True):
                try:
                    vec = model.embed_texts([text])[0]
                    blob = np.array(vec, dtype=np.float32).tobytes()
                    store.update_embeddings({bid: blob})
                    total += 1
                except Exception as e2:
                    if _is_bad_request(e2):
                        logger.warning(f"Block {bid} rejected (400): {e2}")
                    else:
                        logger.warning(f"Block {bid} failed: {e2}")

        logger.info(f"Embedded {total} / {total_to_embed}")

    elapsed = time.time() - start
    logger.info(f"Done embedding {total} blocks in {elapsed:.1f}s")
    return total
