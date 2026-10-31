"""Gemini Live API adapter (BidiGenerateContent over WebSocket).

Reference: https://ai.google.dev/api/live and
https://ai.google.dev/gemini-api/docs/live-tools

The caller's audio goes up as 16 kHz 16-bit PCM in `realtimeInput`; the
model answers with 24 kHz PCM in `serverContent.modelTurn`. Turn detection
runs on Google's side: when the caller talks over a reply the server stops
generating and sends `interrupted`. Function calls arrive as `toolCall` and
are answered with `toolResponse`. Gemini does not report the moment it
decided the caller was done, so `turn_end` is not recorded for this provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Callable

import numpy as np

from ..config import ConfigError
from .base import (
    AdapterError,
    AssistantText,
    AudioDelta,
    RealtimeEvent,
    RealtimeToolCall,
    ResponseDone,
    UserSpeechStarted,
    UserTranscript,
    b64_pcm16,
    dumps,
    pcm16_b64,
)

log = logging.getLogger(__name__)
SETUP_TIMEOUT_S = 15.0


class GeminiLiveAdapter:
    name = "gemini"
    input_sample_rate = 16000
    output_sample_rate = 24000
    reports_speech_stopped = False

    def __init__(self, url: str, api_key: str, model: str, voice: str, connect: Callable[..., Any] | None = None):
        missing = [n for n, v in (("GEMINI_API_KEY", api_key), ("GEMINI_LIVE_URL", url), ("GEMINI_LIVE_MODEL", model), ("GEMINI_LIVE_VOICE", voice)) if not v]
        if missing:
            raise ConfigError(f"PIPELINE=s2s with S2S_PROVIDER=gemini needs {', '.join(missing)} (see .env.example)")
        self.url = url
        self.api_key = api_key
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.voice = voice
        self._connect = connect
        self._ws = None
        self._turn = 0  # Gemini has no response ids; count model turns instead

    def setup_message(self, instructions: str, tools: Any) -> dict[str, Any]:
        setup: dict[str, Any] = {
            "model": self.model,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}},
            },
            "systemInstruction": {"parts": [{"text": instructions}]},
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        if tools:
            setup["tools"] = [{"functionDeclarations": tools.gemini_declarations()}]
        return {"setup": setup}

    async def connect(self, instructions: str, tools: Any) -> None:
        connect = self._connect
        if connect is None:
            import websockets

            connect = websockets.connect
        sep = "&" if "?" in self.url else "?"
        self._ws = await connect(f"{self.url}{sep}key={self.api_key}", max_size=None)
        await self._ws.send(dumps(self.setup_message(instructions, tools)))
        raw = await asyncio.wait_for(self._ws.recv(), timeout=SETUP_TIMEOUT_S)
        message = json.loads(raw)
        if "setupComplete" not in message:
            await self.aclose()
            raise ConnectionError(f"Gemini Live setup failed: {str(message)[:300]}")

    async def send_audio(self, audio: np.ndarray) -> None:
        if self._ws is not None and audio.size:
            chunk = {"data": pcm16_b64(audio), "mimeType": f"audio/pcm;rate={self.input_sample_rate}"}
            await self._ws.send(dumps({"realtimeInput": {"audio": chunk}}))

    async def send_tool_results(self, results: list[tuple[RealtimeToolCall, str]]) -> None:
        if self._ws is None:
            return
        responses = []
        for call, output in results:
            try:
                data = json.loads(output)
            except ValueError:
                data = {"result": output}
            responses.append({"id": call.id, "name": call.name, "response": data if isinstance(data, dict) else {"result": data}})
        await self._ws.send(dumps({"toolResponse": {"functionResponses": responses}}))

    async def interrupt(self, response_id: str, played_ms: int) -> None:
        """Nothing to send: Gemini stops on its own when it hears the caller."""

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
        except Exception as exc:
            yield AdapterError(f"Gemini Live connection lost: {exc}", fatal=True)
            return
        yield AdapterError("Gemini Live closed the session", fatal=True)

    def _translate(self, m: dict[str, Any]) -> list[RealtimeEvent]:
        rid = f"turn-{self._turn}"
        out: list[RealtimeEvent] = []
        if "toolCall" in m:
            calls = [
                RealtimeToolCall(str(c.get("id", "")), str(c.get("name", "")), json.dumps(c.get("args") or {}))
                for c in (m["toolCall"].get("functionCalls") or [])
            ]
            out.append(ResponseDone(rid, calls))
        content = m.get("serverContent")
        if content:
            if content.get("interrupted"):
                self._turn += 1
                out.append(UserSpeechStarted())
            if "inputTranscription" in content:
                out.append(UserTranscript(str(content["inputTranscription"].get("text", "")), final=False))
            for part in (content.get("modelTurn") or {}).get("parts") or []:
                data = (part.get("inlineData") or {}).get("data")
                if data:
                    out.append(AudioDelta(b64_pcm16(data), rid))
            if "outputTranscription" in content:
                out.append(AssistantText(str(content["outputTranscription"].get("text", "")), rid))
            if content.get("turnComplete"):
                self._turn += 1
                out.append(ResponseDone(rid))
        if "goAway" in m:
            out.append(AdapterError("Gemini Live is ending the session (goAway)", fatal=True))
        if "error" in m:
            out.append(AdapterError(f"Gemini Live error: {m['error']}"))
        return out

    async def aclose(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
