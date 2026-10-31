"""OpenAI Realtime API adapter (GA interface, `session.type = "realtime"`).

Reference: https://platform.openai.com/docs/guides/realtime-websocket and
https://platform.openai.com/docs/api-reference/realtime-client-events

Audio is 24 kHz 16-bit PCM in both directions, base64 in JSON events. The
server runs turn detection (`server_vad` or `semantic_vad`), cancels its own
response when the caller starts talking, and reports function calls in
`response.done`. On barge-in the adapter truncates the assistant item to
what was actually played, so the model's memory of the conversation matches
what the caller heard.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Callable, Sequence

import numpy as np

from ..config import ConfigError
from .base import (
    AdapterError,
    AssistantText,
    AudioDelta,
    RealtimeEvent,
    RealtimeToolCall,
    ResponseDone,
    UserItemCommitted,
    UserSpeechStarted,
    UserSpeechStopped,
    UserTranscript,
    b64_pcm16,
    dumps,
    pcm16_b64,
)

log = logging.getLogger(__name__)

# Errors the server sends when a cancel or truncate races the end of a response.
BENIGN_ERRORS = {"response_cancel_not_active", "item_truncate_invalid_item_id"}


class OpenAIRealtimeAdapter:
    name = "openai"
    input_sample_rate = 24000
    output_sample_rate = 24000
    reports_speech_stopped = True

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        voice: str,
        transcribe_model: str = "",
        turn_detection: str = "semantic_vad",
        connect: Callable[..., Any] | None = None,
    ):
        missing = [n for n, v in (("OPENAI_API_KEY", api_key), ("OPENAI_REALTIME_URL", url), ("OPENAI_REALTIME_MODEL", model), ("OPENAI_REALTIME_VOICE", voice)) if not v]
        if missing:
            raise ConfigError(f"PIPELINE=s2s with S2S_PROVIDER=openai needs {', '.join(missing)} (see .env.example)")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.transcribe_model = transcribe_model
        self.turn_detection = turn_detection
        self._connect = connect
        self._ws = None
        self._items: dict[str, str] = {}  # response id -> assistant audio item id

    def session_config(self, instructions: str, tools: Any) -> dict[str, Any]:
        audio_in: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
            "turn_detection": {"type": self.turn_detection},
        }
        if self.transcribe_model:
            audio_in["transcription"] = {"model": self.transcribe_model}
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_in,
                "output": {"format": {"type": "audio/pcm", "rate": self.output_sample_rate}, "voice": self.voice},
            },
        }
        if tools:
            session["tools"] = tools.realtime_schemas()
            session["tool_choice"] = "auto"
        return session

    async def connect(self, instructions: str, tools: Any) -> None:
        connect = self._connect
        if connect is None:
            import websockets

            connect = websockets.connect
        sep = "&" if "?" in self.url else "?"
        self._ws = await connect(
            f"{self.url}{sep}model={self.model}",
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )
        await self._ws.send(dumps({"type": "session.update", "session": self.session_config(instructions, tools)}))

    async def send_audio(self, audio: np.ndarray) -> None:
        if self._ws is not None and audio.size:
            await self._ws.send(dumps({"type": "input_audio_buffer.append", "audio": pcm16_b64(audio)}))

    async def send_tool_results(self, results: list[tuple[RealtimeToolCall, str]]) -> None:
        if self._ws is None:
            return
        for call, output in results:
            item = {"type": "function_call_output", "call_id": call.id, "output": output}
            await self._ws.send(dumps({"type": "conversation.item.create", "item": item}))
        await self._ws.send(dumps({"type": "response.create"}))

    async def interrupt(self, response_id: str, played_ms: int, unheard: Sequence[str] = (), active: str = "") -> None:
        """Stop generating and make the model's context match what the caller heard.

        `response_id` was playing when the caller cut in, `played_ms` of it was
        heard; `unheard` responses (a later tool round) were received but never
        played. `active` is the interrupted reply's latest response, the one
        that may still be generating. The cancel names it: an unnamed cancel
        sent after a burst of events could stop the reply to the caller's new
        words instead.
        """
        if self._ws is None:
            return
        await self._ws.send(dumps({"type": "response.cancel", "response_id": active or response_id}))
        item_id = self._items.get(response_id)
        if item_id:
            await self._ws.send(
                dumps({"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": max(0, played_ms)})
            )
        for later in unheard:
            later_item = self._items.get(later)
            if later_item and later_item != item_id:
                await self._ws.send(dumps({"type": "conversation.item.delete", "item_id": later_item}))

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                for event in self._translate(message):
                    yield event
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # connection closed by the server or the network
            yield AdapterError(f"OpenAI Realtime connection lost: {exc}", fatal=True)
            return
        yield AdapterError("OpenAI Realtime closed the session", fatal=True)

    def _translate(self, m: dict[str, Any]) -> list[RealtimeEvent]:
        kind = m.get("type", "")
        if kind == "response.output_audio.delta":
            rid = str(m.get("response_id", ""))
            if m.get("item_id"):
                self._items[rid] = str(m["item_id"])
            return [AudioDelta(b64_pcm16(m.get("delta", "")), rid)]
        if kind == "response.output_audio_transcript.delta":
            return [AssistantText(str(m.get("delta", "")), str(m.get("response_id", "")))]
        if kind == "input_audio_buffer.speech_started":
            return [UserSpeechStarted()]
        if kind == "input_audio_buffer.speech_stopped":
            return [UserSpeechStopped()]
        if kind == "input_audio_buffer.committed":
            return [UserItemCommitted(str(m.get("item_id", "")))] if m.get("item_id") else []
        if kind == "conversation.item.input_audio_transcription.delta":
            return [UserTranscript(str(m.get("delta", "")), final=False, item_id=str(m.get("item_id", "")))]
        if kind == "conversation.item.input_audio_transcription.completed":
            return [UserTranscript(str(m.get("transcript", "")).strip(), final=True, item_id=str(m.get("item_id", "")))]
        if kind == "conversation.item.input_audio_transcription.failed":
            # No words will come for this item: settle it as empty so nothing waits on it.
            err = m.get("error") or {}
            return [
                UserTranscript("", final=True, item_id=str(m.get("item_id", ""))),
                AdapterError(f"OpenAI Realtime could not transcribe the caller: {err.get('message') or err}"),
            ]
        if kind == "response.done":
            response = m.get("response") or {}
            calls = [
                RealtimeToolCall(str(o.get("call_id", "")), str(o.get("name", "")), str(o.get("arguments") or "{}"))
                for o in response.get("output") or []
                if o.get("type") == "function_call"
            ]
            cancelled = response.get("status") == "cancelled"
            return [ResponseDone(str(response.get("id", "")), [] if cancelled else calls, cancelled)]
        if kind == "error":
            err = m.get("error") or {}
            if err.get("code") in BENIGN_ERRORS:
                return []
            return [AdapterError(f"OpenAI Realtime error: {err.get('message') or err}")]
        return []

    async def aclose(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
