"""Speech to text backends. All take 16 kHz mono float32 audio for one user turn."""

from __future__ import annotations

import asyncio
import threading
from typing import Protocol

import numpy as np


class STT(Protocol):
    name: str

    async def transcribe(self, audio16k: np.ndarray) -> str: ...


def _normalise_peak(audio: np.ndarray) -> np.ndarray | None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return None  # silent clip: nothing to transcribe (and avoids a divide by zero)
    return (audio / peak).astype(np.float32)


class TransformersWhisperSTT:
    """Whisper through Hugging Face `transformers` (the original backend)."""

    name = "transformers"

    def __init__(self, model: str, device: str = "auto", language: str = "en"):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        dtype = torch.float16 if use_cuda else torch.float32
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if use_cuda else -1,
        )
        self._generate_kwargs = {"language": language, "task": "transcribe"} if language else {}
        # One pipeline instance is shared by all sessions; serialise calls into it.
        self._lock = threading.Lock()

    def _run(self, audio: np.ndarray) -> str:
        with self._lock:
            out = self._pipe(
                {"sampling_rate": 16000, "raw": audio}, generate_kwargs=self._generate_kwargs
            )
        return str(out["text"]).strip()

    async def transcribe(self, audio16k: np.ndarray) -> str:
        audio = _normalise_peak(audio16k)
        if audio is None:
            return ""
        return await asyncio.to_thread(self._run, audio)


class FasterWhisperSTT:
    """Whisper through CTranslate2 (`faster-whisper`): int8 on the CPU, much less
    latency than `transformers` for the same model size.

    `model` is a size name (`tiny.en`, `base.en`, `small`, ...) or a Hugging Face
    repo such as `Systran/faster-whisper-tiny`; it downloads on first use.
    """

    name = "faster-whisper"

    def __init__(self, model: str, device: str = "auto", language: str = "en", compute_type: str = "default"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the environment
            from .config import ConfigError

            raise ConfigError("STT_BACKEND=faster-whisper needs `pip install faster-whisper`") from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language or None
        self._lock = threading.Lock()

    def _run(self, audio: np.ndarray) -> str:
        with self._lock:
            segments, _info = self._model.transcribe(
                audio, language=self._language, beam_size=1, vad_filter=False, condition_on_previous_text=False
            )
            return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, audio16k: np.ndarray) -> str:
        audio = _normalise_peak(audio16k)
        if audio is None:
            return ""
        return await asyncio.to_thread(self._run, audio)
