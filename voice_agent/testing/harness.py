"""Drive a `VoiceHandler` the way FastRTC does, without a browser.

`HandlerDriver` binds the handler to a WebRTC id through FastRTC's own
context variable, calls `start_up`, pushes microphone frames into `receive`
and runs a background consumer calling `emit`, exactly like FastRTC's audio
track. Used by the integration tests and by the eval runner.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastrtc import AdditionalOutputs
from fastrtc.utils import Context, current_context

from ..audio import float_to_int16, frames, silence


@dataclass
class EmittedAudio:
    at: float
    sample_rate: int
    samples: np.ndarray


@dataclass
class HandlerDriver:
    handler: Any
    session_id: str
    input_rate: int = 16000
    frame_ms: int = 20
    audio: list[EmittedAudio] = field(default_factory=list)
    ui: list[AdditionalOutputs] = field(default_factory=list)
    _consumer: asyncio.Task | None = None
    _closed: bool = False

    async def start(self) -> "HandlerDriver":
        token = current_context.set(Context(webrtc_id=self.session_id))
        try:
            await self.handler.start_up()
        finally:
            current_context.reset(token)
        self._consumer = asyncio.create_task(self._consume())
        return self

    async def _consume(self) -> None:
        while not self._closed:
            item = await self.handler.emit()
            if item is None:
                continue
            if isinstance(item, AdditionalOutputs):
                self.ui.append(item)
                continue
            sample_rate, samples = item[0], item[1]
            self.audio.append(EmittedAudio(time.perf_counter(), sample_rate, np.asarray(samples).reshape(-1)))

    async def feed(self, audio: np.ndarray, realtime: bool = False) -> None:
        """Send float32 mono audio at `input_rate` as int16 frames, like the browser track."""
        step = self.frame_ms / 1000
        start = time.perf_counter()
        for i, frame in enumerate(frames(audio, self.input_rate, self.frame_ms)):
            await self.handler.receive((self.input_rate, float_to_int16(frame).reshape(1, -1)))
            if realtime:
                delay = start + (i + 1) * step - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)

    async def say(self, audio: np.ndarray, trailing_silence_s: float = 1.2, realtime: bool = False) -> None:
        await self.feed(np.concatenate([silence(0.2, self.input_rate), audio, silence(trailing_silence_s, self.input_rate)]), realtime)

    async def wait_until(self, predicate, timeout: float = 15.0, what: str = "condition") -> None:
        deadline = time.perf_counter() + timeout
        while not predicate():
            if time.perf_counter() > deadline:
                raise TimeoutError(f"timed out waiting for {what}")
            await asyncio.sleep(0.01)

    async def wait_idle(self, timeout: float = 15.0) -> None:
        """Wait until the current reply has been fully produced and played out."""
        await asyncio.sleep(0.05)
        await self.wait_until(lambda: not self.handler.is_replying, timeout, "reply to finish")

    def audio_samples(self) -> np.ndarray:
        if not self.audio:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([a.samples for a in self.audio])

    async def close(self) -> None:
        self._closed = True
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except (asyncio.CancelledError, Exception):
                pass
        self.handler.shutdown()
