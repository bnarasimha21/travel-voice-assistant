"""Deepgram Flux: hosted streaming speech to text with its own end-of-turn model.

With `STT_BACKEND=deepgram-flux` the local VAD turn taker is replaced by a
`FluxTurnSource` per connection. Microphone audio streams to Flux in 80 ms
chunks (the size Deepgram recommends) as 16 kHz linear16, and Flux decides
when the turn is over: its `EndOfTurn` event carries the final transcript, so
no local Whisper runs at all.

Protocol (https://developers.deepgram.com/docs/flux/quickstart):
`wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16&sample_rate=16000`
with `Authorization: Token <key>`. The server sends `TurnInfo` messages whose
`event` is `StartOfTurn`, `Update`, `EagerEndOfTurn`, `TurnResumed` or
`EndOfTurn`, each with the turn's `transcript`. `{"type": "CloseStream"}`
ends the stream.

A local VAD still runs on the same audio, only to timestamp when you stopped
speaking, so the latency records use the same anchor as the local pipeline.
That measurement needs audio at real time (the browser always is; the eval
runner forces `--realtime` for hosted configurations).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Callable
from urllib.parse import urlencode

import numpy as np

from .audio import STT_SAMPLE_RATE, float_to_int16
from .config import ConfigError
from .turn import PREROLL_MS, SpeechStarted, TurnEnded
from .vad import VAD, WINDOW_SAMPLES

log = logging.getLogger(__name__)

CHUNK_SAMPLES = STT_SAMPLE_RATE * 80 // 1000  # 80 ms
RECONNECT_AFTER_S = 2.0


class _Name:
    def __init__(self, name: str):
        self.name = name


class FluxTurnSource:
    """Same surface as `TurnTaker` (`feed`, `in_speech`, `speech_ms`, `reset`,
    `detector`) plus a listener, because Flux decides turns asynchronously:
    `EndOfTurn` arrives from the network, not from a `feed` call."""

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str,
        vad: VAD,
        vad_threshold: float = 0.5,
        eot_threshold: float = 0.7,
        eot_timeout_ms: int = 5000,
        connect: Callable[..., Any] | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.eot_threshold = eot_threshold
        self.eot_timeout_ms = eot_timeout_ms
        self.detector = _Name("flux")
        self.vad = vad
        self.vad_threshold = vad_threshold
        self._connect = connect
        self._listener: Callable[[list], None] | None = None
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._connecting: asyncio.Task | None = None
        self._failed_at = 0.0
        self._send_buf = np.zeros(0, dtype=np.float32)
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._preroll: list[np.ndarray] = []
        self._turn_audio: list[np.ndarray] = []
        self._fed_samples = 0
        self._last_voiced_sample = 0
        self._last_voiced_at = 0.0
        self._turn_start_sample = 0
        self._closed = False
        self.in_speech = False
        self.speech_ms = 0.0

    # Connection --------------------------------------------------------------

    def set_listener(self, listener: Callable[[list], None]) -> None:
        self._listener = listener

    def stream_url(self) -> str:
        query = {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": STT_SAMPLE_RATE,
            "eot_threshold": self.eot_threshold,
            "eot_timeout_ms": self.eot_timeout_ms,
        }
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}{urlencode(query)}"

    async def _open(self) -> None:
        connect = self._connect
        if connect is None:
            import websockets

            connect = websockets.connect
        try:
            self._ws = await connect(self.stream_url(), additional_headers={"Authorization": f"Token {self.api_key}"})
        except Exception as exc:  # network, auth (HTTP 401) or DNS failure
            self._failed_at = time.monotonic()
            log.error("Deepgram Flux connection failed: %s", exc)
            return
        self._reader = asyncio.create_task(self._read(self._ws))

    async def _ensure_connected(self) -> bool:
        if self._ws is not None:
            return True
        if self._closed or time.monotonic() - self._failed_at < RECONNECT_AFTER_S:
            return False
        if self._connecting is None or self._connecting.done():
            self._connecting = asyncio.create_task(self._open())
        await self._connecting
        if self._closed and self._ws is not None:
            await self._drop(self._ws)  # closed while connecting
        return self._ws is not None

    async def _drop(self, ws, close_stream: bool = False) -> None:
        """Forget a connection: stop its reader and close the socket. A turn in
        progress is lost with it; the next connection starts a fresh stream."""
        if self._ws is ws:
            self._ws = None
        reader, self._reader = self._reader, None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        if close_stream:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "CloseStream"}))
        with contextlib.suppress(Exception):
            await ws.close()
        self.reset()

    async def aclose(self) -> None:
        self._closed = True
        connecting = self._connecting
        if connecting is not None and not connecting.done():
            connecting.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await connecting
        if self._ws is not None:
            await self._drop(self._ws, close_stream=True)

    # Audio in ----------------------------------------------------------------

    async def feed(self, audio16k: np.ndarray) -> list[SpeechStarted | TurnEnded]:
        self._track_speech(audio16k)
        if self.in_speech:
            self._turn_audio.append(audio16k)
            self.speech_ms += audio16k.size * 1000 / STT_SAMPLE_RATE
        else:
            self._preroll.append(audio16k)
            while sum(p.size for p in self._preroll) > PREROLL_MS * STT_SAMPLE_RATE // 1000:
                self._preroll.pop(0)
        self._send_buf = np.concatenate([self._send_buf, audio16k])
        if self._send_buf.size < CHUNK_SAMPLES:
            return []
        n = self._send_buf.size // CHUNK_SAMPLES * CHUNK_SAMPLES
        chunk, self._send_buf = self._send_buf[:n], self._send_buf[n:]
        if await self._ensure_connected():
            try:
                await self._ws.send(float_to_int16(chunk).tobytes())
            except Exception as exc:
                log.warning("Deepgram Flux send failed, reconnecting: %s", exc)
                self._failed_at = time.monotonic()
                if self._ws is not None:
                    await self._drop(self._ws)
        return []  # turn events arrive through the listener

    def _track_speech(self, audio16k: np.ndarray) -> None:
        """Local VAD, used only to timestamp the end of speech for the latency records."""
        buf = np.concatenate([self._vad_buf, audio16k])
        n = buf.size // WINDOW_SAMPLES
        for i in range(n):
            window = buf[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
            if self.vad.speech_probability(window) >= self.vad_threshold:
                self._last_voiced_sample = self._fed_samples + (i + 1) * WINDOW_SAMPLES
                self._last_voiced_at = time.perf_counter()
        self._vad_buf = buf[n * WINDOW_SAMPLES :].copy()
        self._fed_samples += audio16k.size

    def reset(self) -> None:
        """Forget the turn in progress (the connection stays open)."""
        self._turn_audio.clear()
        self._preroll.clear()
        self.in_speech = False
        self.speech_ms = 0.0

    # Turn events out ---------------------------------------------------------

    async def _read(self, ws) -> None:
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                events = self._handle(message)
                if events and self._listener is not None:
                    self._listener(events)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Deepgram Flux stream ended: %s", exc)
        finally:
            if self._ws is ws:
                self._ws = None
                self._failed_at = time.monotonic()
                self.reset()  # the turn in progress went with the stream

    def _handle(self, message: dict[str, Any]) -> list[SpeechStarted | TurnEnded]:
        if message.get("type") == "Error":
            log.error("Deepgram Flux error: %s", message.get("description") or message)
            return []
        if message.get("type") != "TurnInfo":
            return []
        event = message.get("event")
        if event == "StartOfTurn":
            if not self.in_speech:
                self.in_speech = True
                self._turn_audio = list(self._preroll)
                self._preroll.clear()
                self.speech_ms = 0.0
            return [SpeechStarted(at=time.perf_counter())]
        if event != "EndOfTurn":
            return []
        transcript = str(message.get("transcript") or "").strip()
        audio = np.concatenate(self._turn_audio) if self._turn_audio else np.zeros(0, dtype=np.float32)
        speech_ms = self.speech_ms
        self.reset()
        if not transcript:
            return []
        # Audio time between the last voiced window and now: what Flux took to decide.
        delay_ms = max(0.0, (self._fed_samples - self._last_voiced_sample) * 1000 / STT_SAMPLE_RATE)
        return [
            TurnEnded(
                audio=audio,
                speech_ms=speech_ms,
                speech_end_at=self._last_voiced_at,
                end_of_turn_delay_ms=delay_ms,
                transcript=transcript,
            )
        ]


class DeepgramFluxSTT:
    """STT backend marker for the handler: turns and transcripts come from
    `open_turns()`, so `transcribe` is never needed."""

    name = "deepgram-flux"
    streaming = True

    def __init__(self, url: str, api_key: str, model: str, eot_threshold: float = 0.7, eot_timeout_ms: int = 5000):
        if not api_key:
            raise ConfigError("STT_BACKEND=deepgram-flux needs DEEPGRAM_API_KEY. Set it in your environment or .env.")
        if not url or not model:
            raise ConfigError("STT_BACKEND=deepgram-flux needs DEEPGRAM_URL and DEEPGRAM_MODEL (see .env.example)")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.eot_threshold = eot_threshold
        self.eot_timeout_ms = eot_timeout_ms

    def open_turns(self, vad: VAD, vad_threshold: float = 0.5) -> FluxTurnSource:
        return FluxTurnSource(
            self.url,
            self.api_key,
            self.model,
            vad=vad,
            vad_threshold=vad_threshold,
            eot_threshold=self.eot_threshold,
            eot_timeout_ms=self.eot_timeout_ms,
        )

    async def transcribe(self, audio16k: np.ndarray) -> str:
        raise RuntimeError("deepgram-flux transcribes while streaming; turns carry their transcript")
