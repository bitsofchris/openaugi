"""OpenAI-compatible LLM adapter.

Works with OpenAI, OpenRouter, Ollama, or any OpenAI-compatible API.
Default model: gpt-5.4-nano (cheap, fast, good enough for classification).

Configure via TOML:
    [models.llm]
    provider = "openai"
    model = "gpt-5.4-nano"

Or override base URL for other providers:
    OPENAI_BASE_URL=https://openrouter.ai/api/v1
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class OpenAILLM:
    """OpenAI-compatible LLM provider."""

    name: str

    def __init__(self, model_name: str = "gpt-5.4-nano"):
        self.name = model_name
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()

    def complete(self, prompt: str, system: str = "") -> str:
        """Simple text completion."""
        self._ensure_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    def structured_output(
        self, prompt: str, response_model: type[BaseModel], system: str = ""
    ) -> BaseModel:
        """Return structured output matching a Pydantic model.

        Uses JSON mode + schema in prompt for broad compatibility.
        """
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n{schema_str}\n\n"
            f"Return ONLY valid JSON, no other text."
        )

        self._ensure_client()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": full_prompt})

        response = self._client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        return response_model.model_validate_json(raw)
