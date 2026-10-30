"""Deterministic offline stand-ins for the speech models.

`FakeTTS` "speaks" text as a tone sequence (one pitch per character) and
`FakeSTT` decodes those tones back to text, so tests and the stub eval mode
push real audio through VAD, turn detection, resampling and the transport
without downloading any model. Unknown characters (digits outside the
alphabet, symbols) are dropped, much like a real ASR loses formatting.

They are selected like any other backend: `STT_BACKEND=fake`,
`TTS_BACKEND=fake`, `SMART_TURN_BACKEND=fake`.
"""

from __future__ import annotations

import numpy as np

from .audio import STT_SAMPLE_RATE, resample

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,?!'-:"
BASE_HZ = 400.0
STEP_HZ = 70.0
CHAR_MS = 30
AMPLITUDE = 0.5


def _freq(index: int) -> float:
    return BASE_HZ + STEP_HZ * index


class FakeCodec:
    """Text <-> tone audio. Encoding is continuous-phase so the decoder sees clean peaks."""

    def __init__(self, char_ms: int = CHAR_MS):
        self.char_ms = char_ms

    def normalise(self, text: str) -> str:
        return "".join(ch for ch in text.lower() if ch in ALPHABET)

    def encode(self, text: str, sample_rate: int, speed: float = 1.0) -> np.ndarray:
        text = self.normalise(text)
        n = int(sample_rate * self.char_ms / 1000 / speed)
        if not text:
            return np.zeros(0, dtype=np.float32)
        out = np.empty(n * len(text), dtype=np.float32)
        phase = 0.0
        t = np.arange(n, dtype=np.float64)
        for i, ch in enumerate(text):
            w = 2 * np.pi * _freq(ALPHABET.index(ch)) / sample_rate
            out[i * n : (i + 1) * n] = AMPLITUDE * np.sin(phase + w * t)
            phase = (phase + w * n) % (2 * np.pi)
        return out

    def decode(self, audio: np.ndarray, sample_rate: int, speed: float = 1.0) -> str:
        """Decode tone audio. Robust to resampling, leading/inner silence and moderate noise."""
        if sample_rate != STT_SAMPLE_RATE:
            audio = resample(audio, sample_rate, STT_SAMPLE_RATE)
            sample_rate = STT_SAMPLE_RATE
        if audio.size == 0:
            return ""
        n = int(sample_rate * self.char_ms / 1000 / speed)
        guard = max(1, n // 6)
        win = max(8, sample_rate // 200)  # 5 ms energy window for onset search
        sq = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
        threshold = (AMPLITUDE / np.sqrt(2)) * 0.55  # roughly half the tone RMS

        def rms(a: int, b: int) -> float:
            b = min(b, audio.size)
            if b <= a:
                return 0.0
            return float(np.sqrt((sq[b] - sq[a]) / (b - a)))

        chars: list[str] = []
        pos = 0
        nfft = 4096
        while pos + win <= audio.size:
            # Find the next onset: first 5 ms window that is clearly voiced.
            energy = np.sqrt(np.maximum(sq[pos + win :] - sq[pos : audio.size - win + 1], 0.0) / win)
            hits = np.nonzero(energy > threshold)[0]
            if hits.size == 0:
                break
            onset = pos + int(hits[0])
            if chars and chars[-1] != " " and onset - pos > n:
                # A real gap inside a decoded run reads as a word break.
                chars.append(" ")
            k = onset
            while k + n - guard <= audio.size and rms(k + guard, k + n - guard) > threshold:
                seg = audio[k + guard : k + n - guard]
                spectrum = np.abs(np.fft.rfft(seg * np.hanning(seg.size), nfft))
                peak_hz = np.argmax(spectrum) * sample_rate / nfft
                idx = int(round((peak_hz - BASE_HZ) / STEP_HZ))
                if 0 <= idx < len(ALPHABET):
                    chars.append(ALPHABET[idx])
                k += n
            pos = max(k, onset + win)
        return " ".join("".join(chars).split()).replace(" .", ".").replace(" ?", "?")


_CODEC = FakeCodec()


class FakeTTS:
    """TTS backend producing tone-encoded speech at 24 kHz."""

    name = "fake"

    def __init__(self, sample_rate: int = 24000, speed: float = 1.0, codec: FakeCodec | None = None):
        self.sample_rate = sample_rate
        self.speed = speed
        self.codec = codec or _CODEC

    async def synthesize(self, text: str) -> np.ndarray:
        return self.codec.encode(text, self.sample_rate, self.speed)


class FakeSTT:
    """STT backend that decodes FakeTTS audio."""

    name = "fake"

    def __init__(self, codec: FakeCodec | None = None):
        self.codec = codec or _CODEC

    async def transcribe(self, audio16k: np.ndarray) -> str:
        return self.codec.decode(audio16k, STT_SAMPLE_RATE)


class FakeEndpointModel:
    """Stand-in for the Smart Turn classifier: a turn is complete when the decoded
    words end in sentence punctuation, and incomplete otherwise (e.g. "fly to... um")."""

    name = "fake"

    def __init__(self, codec: FakeCodec | None = None):
        self.codec = codec or _CODEC

    def predict(self, audio16k: np.ndarray) -> float:
        text = self.codec.decode(audio16k, STT_SAMPLE_RATE).strip()
        return 0.95 if text.endswith((".", "?", "!")) else 0.05
