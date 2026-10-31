"""Small audio helpers shared by the pipeline, tests and evals."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterator

import numpy as np

STT_SAMPLE_RATE = 16000


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 or float audio of shape (n,), (1, n), (ch, n) or (n, ch) to mono float32 in [-1, 1]."""
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        # FastRTC delivers (channels, samples); some callers pass (samples, channels).
        channel_axis = 0 if arr.shape[0] <= arr.shape[1] else 1
        arr = arr.mean(axis=channel_axis)
    elif arr.ndim > 2:
        raise ValueError(f"unsupported audio shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample mono float32 audio. Uses soxr when available, linear interpolation otherwise."""
    if sr_from == sr_to or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    try:
        import soxr

        return soxr.resample(audio, sr_from, sr_to).astype(np.float32, copy=False)
    except ImportError:  # pragma: no cover - soxr ships with the requirements
        n_out = int(round(audio.size * sr_to / sr_from))
        x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


class StreamResampler:
    """Resamples a live stream frame by frame without clicks at frame edges.

    `resample` on each 20 ms frame on its own filters every frame as if the
    signal started and stopped there; a stream resampler carries the filter
    state across frames. Rebuilt if the input rate changes.
    """

    def __init__(self, sr_to: int):
        self.sr_to = sr_to
        self._sr_from: int | None = None
        self._stream = None

    def __call__(self, audio: np.ndarray, sr_from: int) -> np.ndarray:
        if sr_from == self.sr_to:
            return audio.astype(np.float32, copy=False)
        if sr_from != self._sr_from:
            self._sr_from = sr_from
            try:
                import soxr

                self._stream = soxr.ResampleStream(sr_from, self.sr_to, 1, dtype="float32")
            except ImportError:  # pragma: no cover - soxr ships with the requirements
                self._stream = None
        if self._stream is None:
            return resample(audio, sr_from, self.sr_to)
        return self._stream.resample_chunk(audio.astype(np.float32, copy=False))


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def frames(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> Iterator[np.ndarray]:
    """Split audio into fixed frames (the last one zero padded), like a WebRTC track delivers."""
    size = int(sample_rate * frame_ms / 1000)
    for start in range(0, audio.size, size):
        chunk = audio[start : start + size]
        if chunk.size < size:
            chunk = np.pad(chunk, (0, size - chunk.size))
        yield chunk


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Mix white noise into the whole clip (silences included) at the given SNR relative to the speech RMS."""
    rng = np.random.default_rng(seed)
    voiced = audio[np.abs(audio) > 1e-4]
    signal_rms = float(np.sqrt(np.mean(voiced**2))) if voiced.size else 0.1
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    noisy = audio + rng.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_int16(audio).tobytes())


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        if wf.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).T
    return sr, to_mono_float32(data)
