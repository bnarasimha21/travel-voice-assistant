"""Turn taking: VAD windows in, "the user finished a turn" events out.

`TurnTaker` runs the VAD over 16 kHz audio and keeps the current utterance
(with a little pre-roll so the first syllable is not clipped). When the user
pauses, it asks a `TurnDetector` whether the turn is over.

Everything is counted in audio time, not wall-clock time, so feeding audio
faster than real time (tests, evals) gives the same decisions.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .vad import VAD, WINDOW_SAMPLES

SAMPLE_RATE = 16000
WINDOW_MS = WINDOW_SAMPLES * 1000 / SAMPLE_RATE
PREROLL_MS = 300


class TurnDetector(ABC):
    """Decides, during a pause, whether the user's turn is complete."""

    name: str = "base"

    def on_speech(self) -> None:
        """Speech resumed: forget anything decided about the previous pause."""

    @abstractmethod
    async def should_end(self, turn_audio: np.ndarray, silence_ms: float) -> bool:
        """Called on every silent window while a turn is open."""


class SilenceTurnDetector(TurnDetector):
    """End the turn after a fixed pause."""

    name = "silence"

    def __init__(self, silence_ms: int = 700):
        self.silence_ms = silence_ms

    async def should_end(self, turn_audio: np.ndarray, silence_ms: float) -> bool:
        return silence_ms >= self.silence_ms


@dataclass
class SpeechStarted:
    at: float


@dataclass
class TurnEnded:
    audio: np.ndarray
    speech_ms: float
    speech_end_at: float  # wall clock (perf_counter) when the last voiced window was processed
    end_of_turn_delay_ms: float  # audio time from end of speech to the decision
    transcript: str | None = None  # set by streaming STT backends that do their own turn taking


@dataclass
class TurnTaker:
    vad: VAD
    detector: TurnDetector
    threshold: float = 0.5
    min_speech_ms: float = 250
    max_turn_ms: float = 30000

    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _preroll: list[np.ndarray] = field(default_factory=list)
    _turn: list[np.ndarray] = field(default_factory=list)
    in_speech: bool = False
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    _turn_ms: float = 0.0
    _last_voiced_at: float = 0.0

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._turn.clear()
        self.in_speech = False
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self._turn_ms = 0.0
        self.detector.on_speech()

    def current_audio(self) -> np.ndarray:
        return np.concatenate(self._turn) if self._turn else np.zeros(0, dtype=np.float32)

    async def feed(self, audio16k: np.ndarray) -> list[SpeechStarted | TurnEnded]:
        events: list[SpeechStarted | TurnEnded] = []
        buf = np.concatenate([self._pending, audio16k]) if self._pending.size else audio16k
        n_windows = buf.size // WINDOW_SAMPLES
        self._pending = buf[n_windows * WINDOW_SAMPLES :].copy()
        for i in range(n_windows):
            window = buf[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
            event = await self._window(window)
            if event is not None:
                events.append(event)
        return events

    async def _window(self, window: np.ndarray) -> SpeechStarted | TurnEnded | None:
        voiced = self.vad.speech_probability(window) >= self.threshold
        if not self.in_speech:
            if voiced:
                self.in_speech = True
                self._turn = list(self._preroll) + [window]
                self._preroll.clear()
                self.speech_ms = WINDOW_MS
                self.silence_ms = 0.0
                self._turn_ms = WINDOW_MS * len(self._turn)
                self._last_voiced_at = time.perf_counter()
                self.detector.on_speech()
                return SpeechStarted(at=self._last_voiced_at)
            self._preroll.append(window)
            while len(self._preroll) * WINDOW_MS > PREROLL_MS:
                self._preroll.pop(0)
            return None

        self._turn.append(window)
        self._turn_ms += WINDOW_MS
        if voiced:
            if self.silence_ms > 0:
                self.detector.on_speech()
            self.speech_ms += WINDOW_MS
            self.silence_ms = 0.0
            self._last_voiced_at = time.perf_counter()
            if self._turn_ms >= self.max_turn_ms:
                return self._finish()
            return None

        self.silence_ms += WINDOW_MS
        if self.speech_ms < self.min_speech_ms and self.silence_ms >= 200:
            # A click or cough, not a turn.
            self._discard()
            return None
        if self._turn_ms >= self.max_turn_ms or await self.detector.should_end(
            self.current_audio(), self.silence_ms
        ):
            return self._finish()
        return None

    def _discard(self) -> None:
        self._turn.clear()
        self.in_speech = False
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self._turn_ms = 0.0
        self.detector.on_speech()

    def _finish(self) -> TurnEnded:
        event = TurnEnded(
            audio=self.current_audio(),
            speech_ms=self.speech_ms,
            speech_end_at=self._last_voiced_at,
            end_of_turn_delay_ms=self.silence_ms,
        )
        self._discard()
        return event
