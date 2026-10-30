from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice_agent.backends import build_backends
from voice_agent.config import REPO_ROOT, load_settings
from voice_agent.handler import VoiceHandler
from voice_agent.testing.harness import HandlerDriver
from voice_agent.testing.stub_llm import StubLLMServer

FAKE_ENV = {
    "STT_BACKEND": "fake",
    "TTS_BACKEND": "fake",
    "VAD_BACKEND": "energy",
    "LLM_MODEL": "stub-model",
    "LLM_API_KEY": "test",
    "WEATHER_PROVIDER": "fixture",
    "FLIGHTS_PROVIDER": "fixture",
    "HOTELS_PROVIDER": "fixture",
}


def fake_settings(stub: StubLLMServer, **overrides):
    """Settings for fully offline runs: fake speech models, energy VAD, local stub LLM."""
    env = dict(FAKE_ENV, LLM_BASE_URL=stub.base_url)
    return load_settings(env=env, env_files=(), **overrides)


@pytest.fixture
def stub():
    with StubLLMServer() as server:
        yield server


@pytest.fixture
def make_backends(stub):
    def make(**overrides):
        return build_backends(fake_settings(stub, **overrides))

    return make


@pytest.fixture
async def drivers():
    """Factory for started HandlerDrivers; all are closed at teardown."""
    created: list[HandlerDriver] = []

    async def make(backends, session_id: str, input_rate: int = 16000, pace_output: bool = False, **kwargs) -> HandlerDriver:
        handler = VoiceHandler(backends, pace_output=pace_output, **kwargs).copy()
        driver = HandlerDriver(handler, session_id=session_id, input_rate=input_rate)
        await driver.start()
        created.append(driver)
        return driver

    yield make
    for d in created:
        await d.close()


MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))


def real_model_paths() -> dict[str, Path]:
    return {
        "kokoro": MODELS_DIR / "kokoro-v1.0.int8.onnx",
        "voices": MODELS_DIR / "voices-v1.0.bin",
        "smart_turn": MODELS_DIR / "smart-turn-v3.2-cpu.onnx",
    }


def whisper_tiny_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("openai/whisper-tiny", "config.json"), str)
    except Exception:
        return False


def require_real_models(*names: str) -> dict[str, Path]:
    paths = real_model_paths()
    missing = [str(paths[n]) for n in names if not paths[n].exists()]
    if missing:
        pytest.skip(f"real model files missing: {', '.join(missing)}")
    return paths
