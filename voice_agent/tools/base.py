"""Tool registry: schemas for the model, and a safe way to run a call."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


class ToolError(Exception):
    """A tool could not answer (bad arguments, unknown place, provider down).

    The message is shown to the model, so keep it short and factual.
    """


@dataclass
class ToolResult:
    name: str
    ok: bool
    data: dict[str, Any]
    elapsed_ms: float = 0.0

    @property
    def content(self) -> str:
        """The `content` of the OpenAI `tool` message."""
        return json.dumps(self.data, ensure_ascii=False)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[dict[str, Any]]]
    filler: str = "One moment while I look that up."
    apology: str = "Sorry, I couldn't get that information just now."

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolRegistry:
    tools: list[Tool] = field(default_factory=list)
    timeout_s: float = 8.0

    def __bool__(self) -> bool:
        return bool(self.tools)

    def get(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Chat Completions `tools` (nested under "function")."""
        return [t.openai_schema() for t in self.tools]

    def realtime_schemas(self) -> list[dict[str, Any]]:
        """OpenAI Realtime session `tools` (flat: name and parameters at the top level)."""
        return [
            {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self.tools
        ]

    def gemini_declarations(self) -> list[dict[str, Any]]:
        """Gemini Live `functionDeclarations`."""
        return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in self.tools]

    async def run(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        """Run one call with the per-call timeout. Never raises: failures become `{"error": ...}`."""
        started = time.perf_counter()

        def done(ok: bool, data: dict[str, Any]) -> ToolResult:
            return ToolResult(name, ok, data, (time.perf_counter() - started) * 1000)

        tool = self.get(name)
        if tool is None:
            return done(False, {"error": f"unknown tool {name!r}"})
        try:
            args = json.loads(arguments or "{}") if isinstance(arguments, str) else dict(arguments)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            return done(False, {"error": f"invalid arguments: {exc}"})
        allowed = set(tool.parameters.get("properties", {}))
        missing = [p for p in tool.parameters.get("required", []) if p not in args]
        if missing:
            return done(False, {"error": f"missing arguments: {', '.join(missing)}"})
        args = {k: v for k, v in args.items() if k in allowed}
        try:
            data = await asyncio.wait_for(tool.handler(**args), timeout=self.timeout_s)
            return done(True, data)
        except asyncio.TimeoutError:
            log.warning("tool %s timed out after %.1f s", name, self.timeout_s)
            return done(False, {"error": f"{name} timed out"})
        except ToolError as exc:
            return done(False, {"error": str(exc)})
        except Exception as exc:  # a provider bug must not kill the turn
            log.exception("tool %s failed", name)
            return done(False, {"error": f"{name} failed: {type(exc).__name__}"})
