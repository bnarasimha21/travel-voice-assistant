"""The FastRTC stream handler: one instance per browser connection.

Audio arrives from the browser over WebRTC in `receive`, goes through VAD
turn taking, and each finished turn runs STT -> LLM -> TTS. The reply audio
is handed back to FastRTC in `emit`, which sends it to that browser. Nothing
is played on the server.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from fastrtc.utils import current_context

from .audio import STT_SAMPLE_RATE, resample, to_mono_float32
from .llm import LLMError
from .prompts import LLM_FALLBACK_LINE, system_prompt
from .session import Session, SessionStore
from .turn import TurnDetector, TurnEnded, TurnTaker
from .vad import VAD

log = logging.getLogger(__name__)

FRAME_MS = 20
PLAYBACK_LEAD_S = 0.06


@dataclass
class Backends:
    """Shared, already-loaded models and clients. Built once in `app.main()`."""

    stt: Any
    tts: Any
    llm: Any
    store: SessionStore
    make_vad: Callable[[], VAD]
    make_turn_detector: Callable[[], TurnDetector]
    vad_threshold: float = 0.5
    min_speech_ms: float = 250
    max_turn_ms: float = 30000


@dataclass
class _Audio:
    samples: np.ndarray


@dataclass
class _UI:
    outputs: AdditionalOutputs


class VoiceHandler(AsyncStreamHandler):
    def __init__(self, backends: Backends, session_id: str | None = None, pace_output: bool = True):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=backends.tts.sample_rate,
            input_sample_rate=STT_SAMPLE_RATE,
        )
        self.backends = backends
        self.pace_output = pace_output
        self._session_id = session_id
        self.session: Session | None = None
        self.turns = TurnTaker(
            vad=backends.make_vad(),
            detector=backends.make_turn_detector(),
            threshold=backends.vad_threshold,
            min_speech_ms=backends.min_speech_ms,
            max_turn_ms=backends.max_turn_ms,
        )
        self._out: asyncio.Queue[_Audio | _UI] = asyncio.Queue()
        self._reply_task: asyncio.Task | None = None
        self._play_end = 0.0

    # FastRTC interface -------------------------------------------------------

    def copy(self) -> "VoiceHandler":
        return VoiceHandler(self.backends, pace_output=self.pace_output)

    async def start_up(self) -> None:
        self.bind_session()

    def shutdown(self) -> None:
        self._cancel_reply()
        if self.session is not None:
            self.backends.store.drop(self.session.id)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        self.bind_session()
        sample_rate, data = frame
        audio = to_mono_float32(data)
        if sample_rate != STT_SAMPLE_RATE:
            audio = resample(audio, sample_rate, STT_SAMPLE_RATE)
        if self.is_replying:
            # No barge-in yet: ignore the microphone while the agent answers.
            self.turns.reset()
            return
        for event in await self.turns.feed(audio):
            if isinstance(event, TurnEnded):
                self._start_reply(event)

    async def emit(self):
        try:
            item = await asyncio.wait_for(self._out.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None
        if isinstance(item, _UI):
            return item.outputs
        await self._pace(item.samples.size)
        return (self.output_sample_rate, item.samples.reshape(1, -1))

    # Session -----------------------------------------------------------------

    def bind_session(self) -> Session:
        if self.session is None:
            sid = self._session_id
            if sid is None:
                ctx = current_context.get()
                sid = ctx.webrtc_id if ctx is not None else uuid.uuid4().hex
            self.session = self.backends.store.get(sid)
            self.session.on_reset(self._on_session_reset)
        return self.session

    def _on_session_reset(self) -> None:
        self._cancel_reply()
        self._drain()
        self.turns.reset()

    # Reply -------------------------------------------------------------------

    @property
    def is_replying(self) -> bool:
        busy = self._reply_task is not None and not self._reply_task.done()
        return busy or any(isinstance(i, _Audio) for i in list(self._out._queue))  # type: ignore[attr-defined]

    def _start_reply(self, turn: TurnEnded) -> None:
        self._reply_task = asyncio.create_task(self._reply(turn))

    async def _reply(self, turn: TurnEnded) -> None:
        session = self.bind_session()
        transcript = turn.transcript
        if transcript is None:
            transcript = (await self.backends.stt.transcribe(turn.audio)).strip()
        if not transcript:
            return
        messages = session.llm_messages(system_prompt(), transcript)
        try:
            answer = await self.backends.llm.complete(messages)
        except LLMError as exc:
            log.warning("LLM call failed for session %s: %s", session.id, exc)
            await self._speak(LLM_FALLBACK_LINE)
            return
        session.commit_turn(transcript, answer)
        self._out.put_nowait(_UI(AdditionalOutputs(session.chat_messages())))
        await self._speak(answer)

    async def _speak(self, text: str) -> None:
        audio = await self.backends.tts.synthesize(text)
        frame = int(self.output_sample_rate * FRAME_MS / 1000)
        for start in range(0, audio.size, frame):
            self._out.put_nowait(_Audio(audio[start : start + frame].astype(np.float32)))

    def _cancel_reply(self) -> None:
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
        self._reply_task = None

    def _drain(self) -> None:
        while not self._out.empty():
            self._out.get_nowait()
        if self._clear_queue is not None:
            self._clear_queue()  # also drop frames FastRTC already buffered for sending

    # Playback pacing ---------------------------------------------------------

    async def _pace(self, n_samples: int) -> None:
        """Hand frames to FastRTC at real time plus a small lead, so the server
        always knows how much of the reply the visitor has actually heard."""
        now = time.perf_counter()
        if self._play_end < now:
            self._play_end = now  # playback was idle or starved: restart the clock
        if self.pace_output:
            wait = self._play_end - PLAYBACK_LEAD_S - now
            if wait > 0:
                await asyncio.sleep(wait)
        self._play_end += n_samples / self.output_sample_rate

    def unplayed_seconds(self) -> float:
        """Audio already handed to the transport that has not been heard yet."""
        return max(0.0, self._play_end - time.perf_counter()) if self.pace_output else 0.0
