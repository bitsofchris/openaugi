"""Model providers — factory functions for embedding + LLM.

Configuration via TOML or direct instantiation.
Default: sentence-transformers (local, free, no API key).
"""

from __future__ import annotations

import logging
from typing import Any

from openaugi.model.protocols import EmbeddingModel, LLMModel

logger = logging.getLogger(__name__)


def get_embedding_model(config: dict[str, Any] | None = None) -> EmbeddingModel:
    """Create an embedding model from config.

    Config shape:
        {"provider": "sentence-transformers", "model": "all-MiniLM-L6-v2"}
        {"provider": "openai", "model": "text-embedding-3-small"}

    Defaults to local sentence-transformers if no config.
    """
    config = config or {}
    provider = config.get("provider", "sentence-transformers")
    model_name = config.get("model")

    if provider == "sentence-transformers":
        from openaugi.models.embeddings.sentence_transformer import (
            SentenceTransformerEmbedding,
        )

        return SentenceTransformerEmbedding(model_name=model_name or "all-MiniLM-L6-v2")

    if provider == "openai":
        from openaugi.models.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model_name=model_name or "text-embedding-3-small")

    raise ValueError(f"Unknown embedding provider: {provider}")


def get_llm_model(config: dict[str, Any] | None = None) -> LLMModel | None:
    """Create an LLM model from config. Returns None if not configured.

    Config shape:
        {"provider": "openai", "model": "gpt-4o-mini"}
    """
    if not config:
        return None

    provider = config.get("provider")
    if not provider:
        return None

    if provider == "openai":
        from openaugi.models.llms.openai import OpenAILLM

        return OpenAILLM(model_name=config.get("model", "gpt-4o-mini"))

    raise ValueError(f"Unknown LLM provider: {provider}")
