"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import openai

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM call failed (network, auth, rate limit, malformed reply)."""


class OpenAICompatibleLLM:
    """Talks to any OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        extra_query: dict[str, str] | None = None,
        timeout_s: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.extra_query = extra_query or {}
        self._client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-set",
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self.model, messages=messages, extra_query=self.extra_query or None
            )
            content = completion.choices[0].message.content
        except (openai.OpenAIError, httpx.HTTPError, IndexError, AttributeError) as exc:
            raise LLMError(str(exc)) from exc
        if not content or not content.strip():
            raise LLMError("empty completion")
        return content.strip()
