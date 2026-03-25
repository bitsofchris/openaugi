"""Protocols for pluggable model providers.

Thin protocol layer so users can swap embedding/LLM providers
without touching engine code.

See docs/plans/m0.md § Model Abstraction.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding providers."""

    name: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. May use different instruction than batch."""
        ...


@runtime_checkable
class LLMModel(Protocol):
    """Protocol for LLM providers. Not invoked in M0."""

    name: str

    def complete(self, prompt: str, system: str = "") -> str:
        """Simple text completion."""
        ...

    def structured_output(
        self, prompt: str, response_model: type[BaseModel], system: str = ""
    ) -> BaseModel:
        """Return structured output matching a Pydantic model."""
        ...
