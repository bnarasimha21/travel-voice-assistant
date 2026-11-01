"""Text to speech backends. `synthesize` returns mono float32 audio at `sample_rate`."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import REPO_ROOT, ConfigError


class TTS(Protocol):
    name: str
    sample_rate: int

    async def synthesize(self, text: str) -> np.ndarray: ...


def resolve_model_file(path: str, env_name: str) -> Path:
    if not path:
        raise ConfigError(f"{env_name} is not set. See .env.example.")
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise ConfigError(
            f"{env_name} points to {p}, which does not exist. Download the Kokoro files "
            "(kokoro-v1.0.onnx or kokoro-v1.0.int8.onnx, and voices-v1.0.bin) from "
            "https://github.com/thewh1teagle/kokoro-onnx/releases and set "
            "TTS_MODEL_PATH / TTS_VOICES_PATH."
        )
    return p


class KokoroTTS:
    """Kokoro 82M through ONNX Runtime (24 kHz output)."""

    name = "kokoro"
    sample_rate = 24000

    def __init__(self, model_path: str, voices_path: str, voice: str, speed: float = 1.0, lang: str = "en-us"):
        from kokoro_onnx import Kokoro

        model = resolve_model_file(model_path, "TTS_MODEL_PATH")
        voices = resolve_model_file(voices_path, "TTS_VOICES_PATH")
        if not voice:
            raise ConfigError("TTS_VOICE is not set. See .env.example.")
        self._kokoro = Kokoro(str(model), str(voices))
        if voice not in self._kokoro.get_voices():
            raise ConfigError(f"TTS_VOICE={voice!r} is not in {voices.name}")
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._lock = threading.Lock()

    def _run(self, text: str, voice: str, speed: float) -> np.ndarray:
        with self._lock:
            samples, sr = self._kokoro.create(text, voice=voice, speed=speed, lang=self.lang)
        if sr != self.sample_rate:  # pragma: no cover - Kokoro is fixed at 24 kHz
            from .audio import resample

            samples = resample(samples, sr, self.sample_rate)
        return np.asarray(samples, dtype=np.float32)

    async def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        return await asyncio.to_thread(self._run, text, voice or self.voice, speed or self.speed)

    def voices(self) -> list[str]:
        return list(self._kokoro.get_voices())
