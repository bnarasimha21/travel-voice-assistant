"""A small local OpenAI-compatible chat completions server for offline runs.

It speaks enough of the API for this app: non-streaming and SSE streaming
replies, streamed `tool_calls`, artificial latency, and error statuses. Every
request body is recorded, together with whether the client read the whole
stream or hung up early (`outcomes`), so tests can assert that barge-in
really cancels the HTTP request.

The default "brain" is a scripted travel agent: it calls the travel tools
when the question mentions weather, flights or hotels (and tools were
offered), turns tool results into a spoken-style sentence, and otherwise
echoes what the user said so tests can tell which utterance was answered.
Tests can pass their own `brain` to script any behaviour.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


@dataclass
class Plan:
    """What the stub answers to one request."""

    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    status: int = 200
    first_token_delay_s: float = 0.0
    token_delay_s: float = 0.0


Brain = Callable[[dict[str, Any]], Plan]

CITIES = ["san diego", "san francisco", "los angeles", "palm springs", "santa barbara", "anaheim", "new york", "las vegas"]
AIRPORTS = {"lax", "sfo", "san", "jfk", "las", "psp", "sba", "sna", "oak", "sjc"}
CITY_TO_AIRPORT = {"san francisco": "SFO", "san diego": "SAN", "los angeles": "LAX", "new york": "JFK", "las vegas": "LAS", "palm springs": "PSP"}
DAY_WORDS = ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _day(text: str) -> str:
    for word in DAY_WORDS:
        if word in text:
            return word
    return "today"


def _places(text: str) -> list[str]:
    """Cities and airport codes in the order they are mentioned."""
    found: list[tuple[int, str]] = []
    for city in CITIES:
        for m in re.finditer(re.escape(city), text):
            found.append((m.start(), city))
    for m in re.finditer(r"\b([a-z]{3})\b", text):
        if m.group(1) in AIRPORTS and not any(m.start() >= s and m.start() < s + len(c) for s, c in found):
            found.append((m.start(), m.group(1)))
    return [p for _, p in sorted(found)]


def _airport(place: str) -> str:
    return CITY_TO_AIRPORT.get(place, place.upper())


def _describe_tool_result(name: str, content: str) -> str:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"error": content}
    if "error" in data:
        return "Would you like me to try something else, like a different date?"
    if name == "get_weather":
        return (
            f"On {data['day']} in {data['location']} expect a high of {round(data['high_f'])} and a low of "
            f"{round(data['low_f'])} degrees, {data['conditions']}."
        )
    if name == "search_flights":
        flights = data.get("flights") or []
        if not flights:
            return f"I couldn't find flights from {data.get('origin')} to {data.get('destination')} that day."
        f = flights[0]
        return (
            f"The best option is {f['airline']} flight {f['flight_number']}, leaving {f['origin']} at {f['depart_time']} "
            f"and landing at {f['arrive_time']} for {f['price']} {f['currency']}."
        )
    if name == "search_hotels":
        hotels = data.get("hotels") or []
        if not hotels:
            return f"I couldn't find hotels in {data.get('city')} for those dates."
        h = hotels[0]
        return f"{h['name']} has a room for {h['price_per_night']} {h['currency']} a night, rated {h['rating']} stars."
    return "Here is what I found."


def travel_brain(body: dict[str, Any]) -> Plan:
    messages = body.get("messages") or []
    tool_names = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    if messages and messages[-1].get("role") == "tool":
        names = {}
        for m in messages:
            for call in m.get("tool_calls") or []:
                names[call["id"]] = call["function"]["name"]
        trailing = []
        for m in reversed(messages):
            if m.get("role") != "tool":
                break
            trailing.append(m)
        parts = [_describe_tool_result(names.get(m.get("tool_call_id"), ""), str(m.get("content"))) for m in reversed(trailing)]
        return Plan(text=" ".join(parts))

    text = _last_user(messages).lower()
    if "weather" in text and "get_weather" in tool_names:
        places = _places(text)
        return Plan(tool_calls=[("get_weather", {"location": (places[0] if places else "los angeles").title(), "date": _day(text)})])
    if ("flight" in text or "fly" in text) and "search_flights" in tool_names:
        places = _places(text)
        origin = _airport(places[0]) if places else "LAX"
        dest = _airport(places[1]) if len(places) > 1 else "SFO"
        return Plan(tool_calls=[("search_flights", {"origin": origin, "destination": dest, "date": _day(text)})])
    if "hotel" in text and "search_hotels" in tool_names:
        places = _places(text)
        return Plan(tool_calls=[("search_hotels", {"city": (places[0] if places else "san diego").title(), "check_in": _day(text), "nights": 2})])
    return Plan(text=f"Happy to help with that. You asked: {_last_user(messages).strip()}")


def _words(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


class StubLLMServer:
    """Run with `with StubLLMServer() as stub:` and point the client at `stub.base_url`."""

    def __init__(self, brain: Brain | None = None):
        self.brain: Brain = brain or travel_brain
        self.requests: list[dict[str, Any]] = []
        self.outcomes: list[str] = []
        self._outcome_events: list[threading.Event] = []
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    # Lifecycle -------------------------------------------------------------

    def start(self) -> "StubLLMServer":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        app = Starlette(
            routes=[
                Route("/v1/chat/completions", self._handle, methods=["POST"]),
                Route("/chat/completions", self._handle, methods=["POST"]),
            ]
        )
        config = uvicorn.Config(app, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, daemon=True)
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("stub LLM server did not start")
            time.sleep(0.01)
        self.base_url = f"http://127.0.0.1:{port}/v1"
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "StubLLMServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def wait_for_outcome(self, index: int, timeout: float = 5.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if index < len(self._outcome_events) and self._outcome_events[index].wait(timeout=0.05):
                return self.outcomes[index]
            time.sleep(0.01)
        raise TimeoutError(f"request {index} has no outcome yet")

    # Request handling ------------------------------------------------------

    def _record(self, body: dict[str, Any]) -> int:
        self.requests.append(body)
        self.outcomes.append("pending")
        self._outcome_events.append(threading.Event())
        return len(self.requests) - 1

    def _finish(self, index: int, outcome: str) -> None:
        if self.outcomes[index] == "pending":
            self.outcomes[index] = outcome
            self._outcome_events[index].set()

    async def _handle(self, request: Request):
        body = await request.json()
        index = self._record(body)
        plan = self.brain(body)
        if plan.status != 200:
            self._finish(index, "error")
            return JSONResponse({"error": {"message": "stub error", "type": "server_error"}}, status_code=plan.status)
        if plan.first_token_delay_s and not body.get("stream"):
            await asyncio.sleep(plan.first_token_delay_s)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = body.get("model", "stub")
        calls = [
            {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for n, a in plan.tool_calls
        ]
        if not body.get("stream"):
            self._finish(index, "completed")
            message: dict[str, Any] = {"role": "assistant", "content": plan.text or None}
            if calls:
                message["tool_calls"] = calls
            return JSONResponse(
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                }
            )
        return StreamingResponse(self._stream(request, index, cid, model, plan, calls), media_type="text/event-stream")

    async def _stream(self, request: Request, index: int, cid: str, model: str, plan: Plan, calls: list[dict[str, Any]]):
        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        try:
            yield chunk({"role": "assistant", "content": ""})
            if plan.first_token_delay_s:
                await asyncio.sleep(plan.first_token_delay_s)
            for i, word in enumerate(_words(plan.text)):
                if await request.is_disconnected():
                    self._finish(index, "cancelled")
                    return
                if i and plan.token_delay_s:
                    await asyncio.sleep(plan.token_delay_s)
                yield chunk({"content": word})
            for i, call in enumerate(calls):
                args = call["function"]["arguments"]
                half = len(args) // 2
                yield chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": args[:half]}}]})
                yield chunk({"tool_calls": [{"index": i, "function": {"arguments": args[half:]}}]})
            yield chunk({}, "tool_calls" if calls else "stop")
            yield "data: [DONE]\n\n"
            self._finish(index, "completed")
        finally:
            self._finish(index, "cancelled")
