"""Voice activity detection on 16 kHz mono audio, in fixed 32 ms windows.

`SileroVAD` (default) runs the Silero ONNX model that FastRTC ships a loader
for (about 2 MB, fetched from the Hugging Face hub on first use).
`EnergyVAD` is a level threshold with no model; it is what the offline tests
use and is fine for clean, close-mic audio.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

WINDOW_SAMPLES = 512  # 32 ms at 16 kHz, the window size Silero expects


class VAD(Protocol):
    name: str

    def speech_probability(self, window: np.ndarray) -> float:
        """Probability that a WINDOW_SAMPLES window of 16 kHz audio is speech."""


class EnergyVAD:
    name = "energy"

    def __init__(self, threshold_db: float = -40.0):
        self.threshold_db = threshold_db

    def speech_probability(self, window: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) if window.size else 0.0
        db = 20 * np.log10(max(rms, 1e-9))
        return 1.0 if db >= self.threshold_db else 0.0


class _SileroModelHolder:
    _lock = threading.Lock()
    _model = None

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._model is None:
                from fastrtc.pause_detection.silero import SileroVADModel

                cls._model = SileroVADModel()
            return cls._model


class SileroVAD:
    """Streaming Silero VAD. The ONNX session is shared; the recurrent state is per instance."""

    name = "silero"

    def __init__(self):
        self._model = _SileroModelHolder.get()
        self._state = self._model.get_initial_state(batch_size=1)

    def speech_probability(self, window: np.ndarray) -> float:
        x = window.astype(np.float32).reshape(1, -1)
        out, self._state = self._model(x, self._state, 16000)
        return float(np.asarray(out).reshape(-1)[0])


def warm_up_silero() -> None:
    """Load the shared Silero model at startup so the first caller does not pay for it."""
    _SileroModelHolder.get()
