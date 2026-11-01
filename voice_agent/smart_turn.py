"""Pipecat Smart Turn v3: an audio end-of-turn classifier run through ONNX Runtime.

The model (BSD-2-Clause, https://github.com/pipecat-ai/smart-turn) is a
Whisper Tiny encoder with a small head. It takes the last 8 seconds of 16 kHz
mono audio and returns the probability that the speaker has finished their
turn. It hears intonation and filler words ("to... um"), which a pause
threshold cannot. Download `smart-turn-v3.2-cpu.onnx` (about 8 MB) from
https://huggingface.co/pipecat-ai/smart-turn-v3 into `models/`.

Preprocessing follows the reference `inference.py`: keep the last 8 s, pad
with zeros at the start, Whisper log-mel features (80 x 800), one forward
pass, probability above 0.5 means "complete".
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .config import ConfigError, REPO_ROOT

SAMPLE_RATE = 16000
WINDOW_S = 8


def prepare_audio(audio16k: np.ndarray, seconds: int = WINDOW_S) -> np.ndarray:
    """Last `seconds` of audio, zero padded at the start to exactly that length."""
    audio = np.asarray(audio16k, dtype=np.float32).reshape(-1)
    n = seconds * SAMPLE_RATE
    if audio.size >= n:
        return audio[-n:]
    return np.pad(audio, (n - audio.size, 0))


class SmartTurnOnnxModel:
    name = "onnx"

    def __init__(self, model_path: str):
        if not model_path:
            raise ConfigError("SMART_TURN_MODEL_PATH is not set. See .env.example.")
        path = Path(model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise ConfigError(
                f"SMART_TURN_MODEL_PATH points to {path}, which does not exist. Download "
                "smart-turn-v3.2-cpu.onnx from https://huggingface.co/pipecat-ai/smart-turn-v3 "
                "or set TURN_DETECTOR=silence."
            )
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self._features = WhisperFeatureExtractor(chunk_length=WINDOW_S)
        self._lock = threading.Lock()  # the feature extractor is not documented as thread safe

    def predict(self, audio16k: np.ndarray) -> float:
        """Probability (0..1) that the turn is complete."""
        audio = prepare_audio(audio16k)
        with self._lock:
            feats = self._features(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=WINDOW_S * SAMPLE_RATE,
                truncation=True,
                do_normalize=True,
            ).input_features
            out = self._session.run(None, {"input_features": feats.astype(np.float32)})
        return float(np.asarray(out[0]).reshape(-1)[0])
