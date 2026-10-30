"""Build the configured STT, LLM, TTS, VAD and turn detector from Settings."""

from __future__ import annotations

from .config import ConfigError, Settings
from .fakes import FakeSTT, FakeTTS
from .handler import Backends
from .llm import OpenAICompatibleLLM
from .session import SessionStore
from .turn import SilenceTurnDetector, TurnDetector
from .vad import EnergyVAD, SileroVAD, warm_up_silero


def build_stt(settings: Settings):
    backend = settings.stt_backend
    if backend == "fake":
        return FakeSTT()
    if backend == "transformers":
        settings.require("stt_model")
        from .stt import TransformersWhisperSTT

        return TransformersWhisperSTT(settings.stt_model, settings.stt_device, settings.stt_language)
    raise ConfigError(f"Unknown STT_BACKEND={backend!r} (expected transformers or fake)")


def build_tts(settings: Settings):
    backend = settings.tts_backend
    if backend == "fake":
        return FakeTTS(speed=settings.tts_speed)
    if backend == "kokoro":
        from .tts import KokoroTTS

        return KokoroTTS(
            settings.tts_model_path,
            settings.tts_voices_path,
            settings.tts_voice,
            settings.tts_speed,
            settings.tts_lang,
        )
    raise ConfigError(f"Unknown TTS_BACKEND={backend!r} (expected kokoro or fake)")


def build_llm(settings: Settings, http_client=None) -> OpenAICompatibleLLM:
    settings.require("llm_base_url", "llm_model")
    return OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        extra_query=settings.llm_query,
        timeout_s=settings.llm_timeout_s,
        http_client=http_client,
    )


def vad_factory(settings: Settings):
    if settings.vad_backend == "energy":
        return lambda: EnergyVAD(settings.energy_vad_threshold_db)
    if settings.vad_backend == "silero":
        warm_up_silero()
        return SileroVAD
    raise ConfigError(f"Unknown VAD_BACKEND={settings.vad_backend!r} (expected silero or energy)")


def turn_detector_factory(settings: Settings):
    def make() -> TurnDetector:
        return SilenceTurnDetector(settings.silence_ms)

    return make


def build_backends(settings: Settings, *, stt=None, tts=None, llm=None, store: SessionStore | None = None) -> Backends:
    """Load every model once. Explicit stt/tts/llm arguments replace the configured ones."""
    make_vad = vad_factory(settings)
    return Backends(
        stt=stt or build_stt(settings),
        tts=tts or build_tts(settings),
        llm=llm or build_llm(settings),
        store=store or SessionStore(),
        make_vad=make_vad,
        make_turn_detector=turn_detector_factory(settings),
        vad_threshold=settings.vad_threshold,
        min_speech_ms=settings.min_speech_ms,
        max_turn_ms=settings.max_turn_s * 1000,
    )
