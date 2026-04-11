"""Embedding pipeline step — batch embed blocks via EmbeddingModel.

Processes blocks where embedding IS NULL. Stores embeddings as BLOBs
in the blocks table and in the sqlite-vec vec_blocks virtual table.

Embedding strategy: clean content only — no title prepend.

We previously prepended the note title: "{title}\\n\\n{content}". This gave
short/ambiguous chunks more context for retrieval (e.g. a block from a "Trading
Journal" note would embed with that anchor). We removed it because it caused all
blocks from the same document to cluster together — the title signal dominated
over the block's actual semantic content, making topic clusters look like
document clusters. See docs/plans/embedding-strategy.md for the full analysis.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

from openaugi.model.block import Block
from openaugi.model.protocols import EmbeddingModel
from openaugi.store.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

BATCH_SIZE = 32

# ── Content cleaning ──────────────────────────────────────────────────────────

# Order matters: image embeds before generic markdown links, links before raw URLs.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](url)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) → keep text
_RAW_URL_RE = re.compile(r"https?://\S+")  # bare URLs
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}(.*?)\*{1,3}|_{1,3}(.*?)_{1,3}", re.DOTALL)
_CHECKBOX_RE = re.compile(r"^(\s*)-\s+\[[ xX?/\-]\]\s*", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^\s*---+\s*$", re.MULTILINE)


def _clean_for_embedding(text: str) -> str:
    """Strip markdown noise, leaving clean prose for the embedding model.

    Removes:
    - Image embeds  ![alt](url)
    - Raw URLs      https://...
    - Markdown link syntax  [text](url) → keeps the anchor text
    - Bold/italic markers  **x**, *x*, __x__, _x_
    - Checkbox markers  - [ ], - [x], etc. (keeps the task text)
    - Blockquote markers  > (keeps the quoted text)
    - Horizontal rules  ---

    Wikilinks [[...]] and inline #tags are left intact for now.
    """
    text = _IMAGE_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _RAW_URL_RE.sub("", text)
    text = _BOLD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _CHECKBOX_RE.sub(r"\1", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _HR_RE.sub("", text)
    # Collapse runs of blank lines left by the strips
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_embed_text(block: Block) -> str:
    """Build the text to embed for a block — clean content only."""
    return _clean_for_embedding(block.content or "")


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


def _run_embed_batch(
    store: SQLiteStore,
    model: EmbeddingModel,
    blocks: list[Block],
    text_fn: object,  # Callable[[Block], str]
    update_fn: object,  # Callable[[dict[str, bytes]], None]
    batch_size: int = BATCH_SIZE,
) -> int:
    """Shared batch-embed loop. text_fn builds the text; update_fn persists blobs."""
    total_to_embed = len(blocks)
    total = 0
    start = time.time()

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i : i + batch_size]
        texts = [model.truncate(text_fn(b)) for b in batch]  # type: ignore[operator]
        block_ids = [b.id for b in batch]

        try:
            vectors = model.embed_texts(texts)
            embeddings: dict[str, bytes] = {}
            for bid, vec in zip(block_ids, vectors, strict=True):
                embeddings[bid] = np.array(vec, dtype=np.float32).tobytes()
            update_fn(embeddings)  # type: ignore[operator]
            total += len(embeddings)
        except Exception as e:
            if _is_bad_request(e):
                logger.warning(f"Batch {i // batch_size + 1} rejected (400): {e}")
                continue
            logger.warning(f"Batch {i // batch_size + 1} failed: {e}")
            for bid, text in zip(block_ids, texts, strict=True):
                try:
                    vec = model.embed_texts([text])[0]
                    update_fn({bid: np.array(vec, dtype=np.float32).tobytes()})  # type: ignore[operator]
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


def run_embed(
    store: SQLiteStore,
    model: EmbeddingModel,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed all blocks that need title-prepended embeddings. Returns count embedded."""
    blocks = store.get_blocks_needing_embeddings(kind="data_block")
    if not blocks:
        logger.info("All data_block blocks already have embeddings")
        return 0

    # Ensure vec_blocks table exists with the correct dimension before writing
    store.ensure_vec_table(model.dimensions)

    logger.info(f"Embedding {len(blocks)} blocks (model: {model.name})")
    return _run_embed_batch(
        store, model, blocks, _build_embed_text, store.update_embeddings, batch_size
    )


def run_embed_content_only(
    store: SQLiteStore,
    model: EmbeddingModel,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Embed all blocks that need content-only embeddings (no title prefix).

    Writes to the content_only_embedding column. Used for clustering experiments —
    see docs/plans/embedding-strategy.md.
    """
    blocks = store.get_blocks_needing_content_only_embeddings(kind="data_block")
    if not blocks:
        logger.info("All data_block blocks already have content_only_embedding")
        return 0
    logger.info(f"Content-only embedding {len(blocks)} blocks (model: {model.name})")
    return _run_embed_batch(
        store,
        model,
        blocks,
        _clean_for_embedding,
        store.update_content_only_embeddings,
        batch_size,
    )
