"""Local stand-ins for the hosted realtime services, for offline tests and evals.

Three WebSocket servers that speak enough of each provider's protocol for
this app, run in the caller's event loop on 127.0.0.1 with a random port:

- `FakeOpenAIRealtime`: OpenAI Realtime (GA event names)
- `FakeGeminiLive`: Gemini Live `BidiGenerateContent`
- `FakeDeepgramFlux`: Deepgram Flux `/v2/listen` (`TurnInfo` events)

They "hear" the tone speech of `FakeTTS` (decoded with `FakeCodec`), decide
turns with an energy VAD in audio time (a pause of `eot_ms`; the OpenAI and
Flux fakes, like their semantic turn detectors, also wait through a pause
after an unfinished sentence, up to `MAX_SILENCE_MS`), and the two
speech-to-speech fakes answer with the stub LLM's scripted travel brain,
including the tool round trip, as tone audio at 24 kHz. A caller talking over
a reply cancels it, as the real services do. Every message received, the
request path and the request headers are recorded so tests can check exactly
what the adapters sent. Each server checks its credential like the real one
(Bearer token, `Token` header, `key=` query) and answers HTTP 401 otherwise.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import numpy as np

from ..fakes import FakeCodec
from .stub_llm import Brain, travel_brain

SPEECH_RMS = 0.02
WINDOW_MS = 20
DEFAULT_EOT_MS = 500
MAX_SILENCE_MS = 2000  # a "semantic" fake ends even an unfinished turn after this much silence
REPLY_RATE = 24000
CHUNK_MS = 100


def _pcm16_b64(audio: np.ndarray) -> str:
    return base64.b64encode((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()).decode("ascii")


def _from_pcm16(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0


def _chat_tools(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provider tool declarations in the nested Chat Completions shape the brain reads."""
    return [{"type": "function", "function": {"name": d.get("name"), "parameters": d.get("parameters", {})}} for d in declarations]


class EnergyTurns:
    """Speech start and end of turn, measured in audio time, not wall time."""

    def __init__(
        self,
        sample_rate: int,
        eot_ms: int = DEFAULT_EOT_MS,
        speech_rms: float = SPEECH_RMS,
        complete: Callable[[np.ndarray], bool] | None = None,
        max_silence_ms: int = MAX_SILENCE_MS,
    ):
        self.sample_rate = sample_rate
        self.speech_rms = speech_rms
        self.window = sample_rate * WINDOW_MS // 1000
        self.eot_windows = max(1, eot_ms // WINDOW_MS)
        # complete(turn audio) -> False keeps the turn open after the pause, up to max_silence_ms
        self.complete = complete
        self.max_windows = max(self.eot_windows, max_silence_ms // WINDOW_MS)
        self._waiting = False
        self._buf = np.zeros(0, dtype=np.float32)
        self._turn: list[np.ndarray] = []
        self._silent = 0
        self.in_speech = False

    def feed(self, audio: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        buf = np.concatenate([self._buf, audio])
        events: list[tuple[str, np.ndarray | None]] = []
        n = buf.size // self.window
        for i in range(n):
            w = buf[i * self.window : (i + 1) * self.window]
            voiced = float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) >= self.speech_rms
            if voiced and not self.in_speech:
                self.in_speech = True
                self._turn = []
                events.append(("start", None))
            if self.in_speech:
                self._turn.append(w)
                self._silent = 0 if voiced else self._silent + 1
                if voiced:
                    self._waiting = False
                elif self._silent == self.eot_windows and self.complete is not None:
                    self._waiting = not self.complete(np.concatenate(self._turn))
                if self._silent >= (self.max_windows if self._waiting else self.eot_windows):
                    self.in_speech = False
                    self._silent = 0
                    self._waiting = False
                    events.append(("end", np.concatenate(self._turn)))
                    self._turn = []
        self._buf = buf[n * self.window :].copy()
        return events


@dataclass
class Connection:
    """One client session as the fake server saw it."""

    path: str
    headers: dict[str, str]
    received: list[Any] = field(default_factory=list)
    sent: list[dict[str, Any]] = field(default_factory=list)
    conversation: list[dict[str, Any]] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)

    @property
    def query(self) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlsplit(self.path).query).items()}

    def kinds(self) -> list[str]:
        return [m.get("type", next(iter(m), "")) for m in self.received if isinstance(m, dict)]


class _FakeServer:
    path_prefix = "/"
    semantic_turns = False  # wait through pauses after unfinished sentences

    def __init__(
        self,
        brain: Brain | None = None,
        eot_ms: int = DEFAULT_EOT_MS,
        reply_speedup: float = 4.0,
        speech_rms: float = SPEECH_RMS,
        codec: FakeCodec | None = None,
    ):
        self.brain = brain or travel_brain
        self.eot_ms = eot_ms
        self.speech_rms = speech_rms  # raise above the noise floor for noisy callers
        self.reply_speedup = reply_speedup  # how much faster than real time reply audio is sent
        self.codec = codec or FakeCodec()
        self.connections: list[Connection] = []
        self.rejected = 0
        self._server = None
        self._live: set[Any] = set()
        self.port = 0

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}{self.path_prefix}"

    async def start(self) -> "_FakeServer":
        from websockets.asyncio.server import serve

        self._server = await serve(self._serve, "127.0.0.1", 0, process_request=self._check, max_size=None)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 5)
            self._server = None

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    async def drop_all(self) -> None:
        """Close every open session from the server side, like a network drop."""
        for ws in list(self._live):
            with contextlib.suppress(Exception):
                await ws.close(1011, "fake server dropped the session")

    def authorised(self, path: str, headers: dict[str, str]) -> bool:
        raise NotImplementedError

    def _check(self, connection, request):
        headers = {k.lower(): v for k, v in request.headers.raw_items()}
        if not self.authorised(request.path, headers):
            self.rejected += 1
            return connection.respond(401, "unauthorised\n")
        return None

    async def _serve(self, ws) -> None:
        conn = Connection(ws.request.path, {k.lower(): v for k, v in ws.request.headers.raw_items()})
        self.connections.append(conn)
        self._live.add(ws)
        try:
            await self.session(ws, conn)
        except Exception:  # client went away
            pass
        finally:
            self._live.discard(ws)

    async def session(self, ws, conn: Connection) -> None:
        raise NotImplementedError

    async def _send(self, ws, conn: Connection, message: dict[str, Any]) -> None:
        conn.sent.append(message)
        await ws.send(json.dumps(message))

    def turns(self, sample_rate: int, semantic: bool | None = None) -> EnergyTurns:
        complete = None
        if self.semantic_turns if semantic is None else semantic:
            def complete(audio: np.ndarray) -> bool:
                return self.codec.decode(audio, sample_rate).strip().endswith((".", "?", "!"))
        return EnergyTurns(sample_rate, self.eot_ms, self.speech_rms, complete=complete)

    async def _pace(self, samples: int) -> None:
        await asyncio.sleep(samples / REPLY_RATE / self.reply_speedup if self.reply_speedup > 0 else 0)


class FakeOpenAIRealtime(_FakeServer):
    """OpenAI Realtime over WebSocket, server VAD on, `interrupt_response` behaviour.
    `semantic_vad` sessions wait through pauses after unfinished sentences;
    `server_vad` sessions end on any pause."""

    path_prefix = "/v1/realtime"

    def __init__(self, api_key: str = "test-openai-key", **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = api_key

    def authorised(self, path: str, headers: dict[str, str]) -> bool:
        return headers.get("authorization") == f"Bearer {self.api_key}" and "model=" in path

    async def session(self, ws, conn: Connection) -> None:
        state: dict[str, Any] = {"tools": [], "task": None, "items": {}, "rid": ""}
        turns = self.turns(24000)
        await self._send(ws, conn, {"type": "session.created", "session": {"id": f"sess_{uuid.uuid4().hex[:8]}"}})
        try:
            async for raw in ws:
                m = json.loads(raw)
                kind = m.get("type")
                conn.received.append(m if kind != "input_audio_buffer.append" else {"type": kind, "bytes": len(m.get("audio", ""))})
                if kind == "session.update":
                    session = m.get("session") or {}
                    state["tools"] = _chat_tools(session.get("tools") or [])
                    detection = ((session.get("audio") or {}).get("input") or {}).get("turn_detection") or {}
                    if detection.get("type"):
                        turns.complete = self.turns(24000, semantic=detection["type"] == "semantic_vad").complete
                    if session.get("instructions"):
                        system = {"role": "system", "content": session["instructions"]}
                        if conn.conversation and conn.conversation[0].get("role") == "system":
                            conn.conversation[0] = system
                        else:
                            conn.conversation.insert(0, system)
                    await self._send(ws, conn, {"type": "session.updated", "session": session})
                elif kind == "input_audio_buffer.append":
                    for event, audio in turns.feed(_from_pcm16(base64.b64decode(m.get("audio", "")))):
                        if event == "start":
                            await self._send(ws, conn, {"type": "input_audio_buffer.speech_started"})
                            await self._cancel(ws, conn, state)
                        else:
                            await self._end_of_turn(ws, conn, state, audio)
                elif kind == "conversation.item.create":
                    item = m.get("item") or {}
                    if item.get("type") == "function_call_output":
                        conn.conversation.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output", "")})
                elif kind == "response.create":
                    self._start_response(ws, conn, state)
                elif kind == "response.cancel":
                    if not await self._cancel(ws, conn, state):
                        await self._send(ws, conn, {"type": "error", "error": {"type": "invalid_request_error", "code": "response_cancel_not_active", "message": "no active response"}})
                elif kind == "conversation.item.truncate":
                    self._truncate(conn, state, m)
                    await self._send(ws, conn, {"type": "conversation.item.truncated", "item_id": m.get("item_id")})
        finally:
            task = state["task"]
            if task is not None:
                task.cancel()

    async def _end_of_turn(self, ws, conn: Connection, state: dict[str, Any], audio: np.ndarray) -> None:
        await self._send(ws, conn, {"type": "input_audio_buffer.speech_stopped"})
        await self._send(ws, conn, {"type": "input_audio_buffer.committed", "item_id": f"item_{uuid.uuid4().hex[:8]}"})
        text = self.codec.decode(audio, 24000)
        conn.transcripts.append(text)
        conn.conversation.append({"role": "user", "content": text})
        for word in text.split(" "):
            await self._send(ws, conn, {"type": "conversation.item.input_audio_transcription.delta", "delta": word + " "})
        await self._send(ws, conn, {"type": "conversation.item.input_audio_transcription.completed", "transcript": text})
        self._start_response(ws, conn, state)

    def _start_response(self, ws, conn: Connection, state: dict[str, Any]) -> None:
        state["rid"] = f"resp_{uuid.uuid4().hex[:8]}"
        state["task"] = asyncio.create_task(self._respond(ws, conn, state, state["rid"]))

    async def _cancel(self, ws, conn: Connection, state: dict[str, Any]) -> bool:
        task = state["task"]
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        await self._send(ws, conn, {"type": "response.done", "response": {"id": state["rid"], "status": "cancelled", "output": []}})
        return True

    def _truncate(self, conn: Connection, state: dict[str, Any], m: dict[str, Any]) -> None:
        entry = state["items"].get(m.get("item_id"))
        if entry is None:
            return
        index, text, total_ms = entry
        words = text.split()
        keep = int(len(words) * min(1.0, max(0, m.get("audio_end_ms", 0)) / max(total_ms, 1)))
        conn.conversation[index] = {"role": "assistant", "content": " ".join(words[:keep])}

    async def _respond(self, ws, conn: Connection, state: dict[str, Any], rid: str) -> None:
        plan = self.brain({"messages": conn.conversation, "tools": state["tools"]})
        await self._send(ws, conn, {"type": "response.created", "response": {"id": rid, "status": "in_progress"}})
        if plan.tool_calls:
            output = []
            calls = []
            for name, args in plan.tool_calls:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                arguments = json.dumps(args)
                output.append({"type": "function_call", "id": f"item_{uuid.uuid4().hex[:8]}", "call_id": call_id, "name": name, "arguments": arguments})
                calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
            conn.conversation.append({"role": "assistant", "content": None, "tool_calls": calls})
            await self._send(ws, conn, {"type": "response.done", "response": {"id": rid, "status": "completed", "output": output}})
            return
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        audio = self.codec.encode(plan.text, REPLY_RATE)
        words = plan.text.split(" ")
        conn.conversation.append({"role": "assistant", "content": plan.text})
        state["items"][item_id] = (len(conn.conversation) - 1, plan.text, audio.size * 1000 / REPLY_RATE)
        chunk = REPLY_RATE * CHUNK_MS // 1000
        n_chunks = max(1, -(-audio.size // chunk))
        sent_words = 0
        for i in range(n_chunks):
            piece = audio[i * chunk : (i + 1) * chunk]
            upto = round(len(words) * (i + 1) / n_chunks)
            if upto > sent_words:
                delta = " ".join(words[sent_words:upto]) + (" " if upto < len(words) else "")
                sent_words = upto
                await self._send(ws, conn, {"type": "response.output_audio_transcript.delta", "response_id": rid, "item_id": item_id, "delta": delta})
            await self._send(ws, conn, {"type": "response.output_audio.delta", "response_id": rid, "item_id": item_id, "content_index": 0, "delta": _pcm16_b64(piece)})
            await self._pace(piece.size)
        await self._send(ws, conn, {"type": "response.output_audio.done", "response_id": rid, "item_id": item_id})
        message = {"type": "message", "id": item_id, "role": "assistant", "content": [{"type": "output_audio", "transcript": plan.text}]}
        await self._send(ws, conn, {"type": "response.done", "response": {"id": rid, "status": "completed", "output": [message]}})


class FakeGeminiLive(_FakeServer):
    """Gemini Live `BidiGenerateContent`: setup handshake, realtimeInput, toolCall."""

    path_prefix = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    def __init__(self, api_key: str = "test-gemini-key", **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = api_key

    def authorised(self, path: str, headers: dict[str, str]) -> bool:
        return parse_qs(urlsplit(path).query).get("key", [""])[-1] == self.api_key

    async def session(self, ws, conn: Connection) -> None:
        first = json.loads(await ws.recv())
        conn.received.append(first)
        setup = first.get("setup")
        if not isinstance(setup, dict) or not str(setup.get("model", "")).startswith("models/"):
            await ws.close(1007, "the first message must be setup with a models/ name")
            return
        state: dict[str, Any] = {"task": None}
        tools = []
        for group in setup.get("tools") or []:
            tools.extend(group.get("functionDeclarations") or [])
        state["tools"] = _chat_tools(tools)
        instructions = " ".join(p.get("text", "") for p in (setup.get("systemInstruction") or {}).get("parts") or [])
        conn.conversation.append({"role": "system", "content": instructions})
        await self._send(ws, conn, {"setupComplete": {}})
        turns = self.turns(16000)
        try:
            async for raw in ws:
                m = json.loads(raw)
                if "realtimeInput" in m:
                    chunk = (m["realtimeInput"].get("audio") or {})
                    conn.received.append({"realtimeInput": {"mimeType": chunk.get("mimeType"), "bytes": len(chunk.get("data", ""))}})
                    for event, audio in turns.feed(_from_pcm16(base64.b64decode(chunk.get("data", "")))):
                        if event == "start":
                            task = state["task"]
                            if task is not None and not task.done():
                                task.cancel()
                                with contextlib.suppress(asyncio.CancelledError, Exception):
                                    await task
                                await self._send(ws, conn, {"serverContent": {"interrupted": True}})
                        else:
                            text = self.codec.decode(audio, 16000)
                            conn.transcripts.append(text)
                            conn.conversation.append({"role": "user", "content": text})
                            await self._send(ws, conn, {"serverContent": {"inputTranscription": {"text": text}}})
                            state["task"] = asyncio.create_task(self._respond(ws, conn, state))
                    continue
                conn.received.append(m)
                if "toolResponse" in m:
                    for r in m["toolResponse"].get("functionResponses") or []:
                        conn.conversation.append({"role": "tool", "tool_call_id": r.get("id"), "content": json.dumps(r.get("response"))})
                    state["task"] = asyncio.create_task(self._respond(ws, conn, state))
        finally:
            task = state["task"]
            if task is not None:
                task.cancel()

    async def _respond(self, ws, conn: Connection, state: dict[str, Any]) -> None:
        plan = self.brain({"messages": conn.conversation, "tools": state["tools"]})
        if plan.tool_calls:
            calls, chat_calls = [], []
            for name, args in plan.tool_calls:
                call_id = f"fc_{uuid.uuid4().hex[:8]}"
                calls.append({"id": call_id, "name": name, "args": args})
                chat_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
            conn.conversation.append({"role": "assistant", "content": None, "tool_calls": chat_calls})
            await self._send(ws, conn, {"toolCall": {"functionCalls": calls}})
            return
        audio = self.codec.encode(plan.text, REPLY_RATE)
        conn.conversation.append({"role": "assistant", "content": plan.text})
        words = plan.text.split(" ")
        chunk = REPLY_RATE * CHUNK_MS // 1000
        n_chunks = max(1, -(-audio.size // chunk))
        sent_words = 0
        for i in range(n_chunks):
            piece = audio[i * chunk : (i + 1) * chunk]
            part = {"inlineData": {"mimeType": f"audio/pcm;rate={REPLY_RATE}", "data": _pcm16_b64(piece)}}
            await self._send(ws, conn, {"serverContent": {"modelTurn": {"parts": [part]}}})
            upto = round(len(words) * (i + 1) / n_chunks)
            if upto > sent_words:
                text = " ".join(words[sent_words:upto]) + (" " if upto < len(words) else "")
                sent_words = upto
                await self._send(ws, conn, {"serverContent": {"outputTranscription": {"text": text}}})
            await self._pace(piece.size)
        await self._send(ws, conn, {"serverContent": {"generationComplete": True}})
        await self._send(ws, conn, {"serverContent": {"turnComplete": True}})


class FakeDeepgramFlux(_FakeServer):
    """Deepgram Flux `/v2/listen`: linear16 in, `TurnInfo` events out."""

    path_prefix = "/v2/listen"
    semantic_turns = True  # Flux decides end of turn from the words as well as the pause

    def __init__(self, api_key: str = "test-deepgram-key", **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = api_key

    def authorised(self, path: str, headers: dict[str, str]) -> bool:
        return headers.get("authorization") == f"Token {self.api_key}"

    async def session(self, ws, conn: Connection) -> None:
        q = conn.query
        if q.get("encoding") != "linear16" or not q.get("model") or not q.get("sample_rate"):
            await self._send(ws, conn, {"type": "Error", "code": "INVALID_QUERY", "description": "model, encoding=linear16 and sample_rate are required"})
            await ws.close(1008, "bad query")
            return
        rate = int(q["sample_rate"])
        turns = self.turns(rate)
        index = 0
        await self._send(ws, conn, {"type": "Connected", "request_id": uuid.uuid4().hex, "sequence_id": 0})
        async for raw in ws:
            if isinstance(raw, bytes):
                conn.received.append({"audio_bytes": len(raw)})
                for event, audio in turns.feed(_from_pcm16(raw)):
                    if event == "start":
                        await self._send(ws, conn, {"type": "TurnInfo", "event": "StartOfTurn", "turn_index": index, "transcript": "", "end_of_turn_confidence": 0.0})
                    else:
                        text = self.codec.decode(audio, rate)
                        conn.transcripts.append(text)
                        await self._send(ws, conn, {"type": "TurnInfo", "event": "EndOfTurn", "turn_index": index, "transcript": text, "end_of_turn_confidence": 0.9})
                        index += 1
                continue
            m = json.loads(raw)
            conn.received.append(m)
            if m.get("type") == "CloseStream":
                await ws.close()
                return
