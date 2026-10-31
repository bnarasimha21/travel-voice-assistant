"""OpenAI-compatible chat completions client, streaming and non-streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import openai

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The LLM call failed (network, auth, rate limit, malformed reply)."""


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as the model produced it

    def parsed_arguments(self) -> dict[str, Any]:
        """Arguments as a dict; raises ValueError if the model sent bad JSON."""
        data = json.loads(self.arguments or "{}")
        if not isinstance(data, dict):
            raise ValueError("tool arguments must be a JSON object")
        return data

    def as_message_entry(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ToolCalls:
    """The model asked for these calls; the round ended with finish_reason=tool_calls."""

    calls: list[ToolCall] = field(default_factory=list)
    text: str = ""  # any text the model said before asking


@dataclass
class Done:
    finish_reason: str | None


LLMEvent = TextDelta | ToolCalls | Done


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
        """One non-streaming reply (used by the eval judge)."""
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

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMEvent]:
        """Stream one model round as TextDelta events, then ToolCalls (if any), then Done.

        Cancelling the consumer closes the HTTP response, so the server stops
        generating. Use `contextlib.aclosing` when breaking out early.
        """
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_query=self.extra_query or None,
                **kwargs,
            )
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise LLMError(str(exc)) from exc

        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish: str | None = None
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None and delta.content:
                    text.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in (delta.tool_calls if delta is not None else None) or []:
                    key = getattr(tc, "index", None)
                    if key is None:
                        # Some OpenAI-compatible servers omit `index`: a new id
                        # starts a new call, anything else continues the last one.
                        last = max(calls) if calls else None
                        new_call = last is None or (tc.id and calls[last]["id"] and tc.id != calls[last]["id"])
                        key = (last + 1 if last is not None else 0) if new_call else last
                    slot = calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        except (openai.OpenAIError, httpx.HTTPError) as exc:
            raise LLMError(str(exc)) from exc
        finally:
            await stream.close()

        if calls:
            parsed = [
                ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"])
                for i, c in sorted(calls.items())
            ]
            yield ToolCalls(parsed, "".join(text))
        elif not "".join(text).strip():
            raise LLMError("empty completion")
        yield Done(finish)


# Tool loop ---------------------------------------------------------------------


@dataclass
class ToolStarted:
    call: ToolCall


@dataclass
class ToolFinished:
    call: ToolCall
    result: Any  # tools.ToolResult
    messages: list[dict[str, Any]]  # complete assistant tool_calls message + tool results, once per round


AgentEvent = TextDelta | ToolStarted | ToolFinished


async def stream_with_tools(
    llm: Any,
    messages: list[dict[str, Any]],
    registry: Any = None,
    max_rounds: int = 4,
) -> AsyncIterator[AgentEvent]:
    """Stream a reply, running tool calls between model rounds.

    Each round streams text deltas. If the round ends with tool calls, every
    call is announced (ToolStarted), run concurrently with the registry's
    per-call timeout, and reported (ToolFinished; the last event of a round
    carries the assistant and tool messages to keep in history). Then the
    model is called again with the results. The final round is sent without
    tools so the model has to answer in words.
    """
    convo = list(messages)
    for round_index in range(max_rounds):
        tools = registry.openai_schemas() if registry and round_index < max_rounds - 1 else None
        requested: ToolCalls | None = None
        async with contextlib.aclosing(llm.stream(convo, tools)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    yield event
                elif isinstance(event, ToolCalls):
                    requested = event
        if requested is None:
            return
        assistant = {
            "role": "assistant",
            "content": requested.text or None,
            "tool_calls": [c.as_message_entry() for c in requested.calls],
        }
        for call in requested.calls:
            yield ToolStarted(call)
        results = await asyncio.gather(*(registry.run(c.name, c.arguments) for c in requested.calls))
        round_messages = [assistant] + [
            {"role": "tool", "tool_call_id": c.id, "content": r.content} for c, r in zip(requested.calls, results)
        ]
        convo.extend(round_messages)
        for i, (call, result) in enumerate(zip(requested.calls, results)):
            yield ToolFinished(call, result, round_messages if i == len(results) - 1 else [])
    raise LLMError(f"no answer after {max_rounds} model rounds")
