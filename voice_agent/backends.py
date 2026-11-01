"""Build the configured STT, LLM, TTS, VAD and turn detector from Settings."""

from __future__ import annotations

from .config import ConfigError, Settings
from .fakes import FakeSTT, FakeTTS
from .handler import Backends
from .realtime.handler import S2SBackends
from .llm import OpenAICompatibleLLM
from .metrics import TurnLog
from .session import SessionStore
from .turn import SilenceTurnDetector, SmartTurnDetector, TurnDetector
from .vad import EnergyVAD, SileroVAD, warm_up_silero


def build_stt(settings: Settings):
    backend = settings.stt_backend
    if backend == "fake":
        return FakeSTT()
    if backend == "transformers":
        settings.require("stt_model")
        from .stt import TransformersWhisperSTT

        return TransformersWhisperSTT(settings.stt_model, settings.stt_device, settings.stt_language)
    if backend == "faster-whisper":
        settings.require("stt_model")
        from .stt import FasterWhisperSTT

        device = "cpu" if settings.stt_device in ("", "auto", "mps") else settings.stt_device
        return FasterWhisperSTT(settings.stt_model, device, settings.stt_language, settings.stt_compute_type or "default")
    if backend == "deepgram-flux":
        from .deepgram import DeepgramFluxSTT

        return DeepgramFluxSTT(
            settings.deepgram_url,
            settings.deepgram_api_key,
            settings.deepgram_model,
            eot_threshold=settings.deepgram_eot_threshold,
            eot_timeout_ms=settings.deepgram_eot_timeout_ms,
        )
    raise ConfigError(
        f"Unknown STT_BACKEND={backend!r} (expected transformers, faster-whisper, deepgram-flux or fake)"
    )


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


def build_endpoint_model(settings: Settings):
    backend = settings.smart_turn_backend
    if backend == "fake":
        from .fakes import FakeEndpointModel

        return FakeEndpointModel()
    if backend == "onnx":
        from .smart_turn import SmartTurnOnnxModel

        return SmartTurnOnnxModel(settings.smart_turn_model_path)
    raise ConfigError(f"Unknown SMART_TURN_BACKEND={backend!r} (expected onnx or fake)")


def turn_detector_factory(settings: Settings, endpoint_model=None):
    """A new detector per connection; the Smart Turn model itself is loaded once and shared."""
    kind = settings.turn_detector
    if kind == "silence":
        return lambda: SilenceTurnDetector(settings.silence_ms)
    if kind == "smart":
        model = endpoint_model or build_endpoint_model(settings)
        return lambda: SmartTurnDetector(
            model,
            pause_ms=settings.smart_turn_pause_ms,
            max_silence_ms=settings.smart_turn_max_silence_ms,
            threshold=settings.smart_turn_threshold,
        )
    raise ConfigError(f"Unknown TURN_DETECTOR={kind!r} (expected silence or smart)")


def build_backends(
    settings: Settings, *, stt=None, tts=None, llm=None, store: SessionStore | None = None, tools=None
) -> Backends:
    """Load every model once. Explicit stt/tts/llm/tools arguments replace the configured ones."""
    from .tools import build_registry

    make_vad = vad_factory(settings)
    stt = stt or build_stt(settings)
    make_turn_source = None
    if getattr(stt, "streaming", False):
        # Deepgram Flux decides turns itself; the local VAD only timestamps speech end.
        make_turn_source = lambda: stt.open_turns(make_vad(), settings.vad_threshold)  # noqa: E731
    return Backends(
        stt=stt,
        tts=tts or build_tts(settings),
        llm=llm or build_llm(settings),
        store=store or SessionStore(),
        make_vad=make_vad,
        make_turn_detector=(lambda: None) if make_turn_source else turn_detector_factory(settings),
        make_turn_source=make_turn_source,
        vad_threshold=settings.vad_threshold,
        min_speech_ms=settings.min_speech_ms,
        max_turn_ms=settings.max_turn_s * 1000,
        barge_in=settings.barge_in,
        barge_in_min_speech_ms=settings.barge_in_min_speech_ms,
        tools=tools if tools is not None else build_registry(settings),
        max_tool_rounds=settings.max_tool_rounds,
        filler_after_ms=settings.filler_after_ms,
        turn_log=TurnLog(settings.metrics_log_path) if settings.metrics_log_path else None,
    )


def build_adapter_factory(settings: Settings):
    """A factory for one realtime session per browser connection."""
    provider = settings.s2s_provider
    if provider == "openai":
        from .realtime.openai_realtime import OpenAIRealtimeAdapter

        detection = settings.openai_realtime_turn_detection or "server_vad"
        if detection not in ("semantic_vad", "server_vad"):
            raise ConfigError(
                f"OPENAI_REALTIME_TURN_DETECTION={detection!r} (expected semantic_vad or server_vad)"
            )

        def make():
            return OpenAIRealtimeAdapter(
                settings.openai_realtime_url,
                settings.openai_api_key,
                settings.openai_realtime_model,
                settings.openai_realtime_voice,
                transcribe_model=settings.openai_realtime_transcribe_model,
                turn_detection=detection,
            )

    elif provider == "gemini":
        from .realtime.gemini_live import GeminiLiveAdapter

        def make():
            return GeminiLiveAdapter(
                settings.gemini_live_url, settings.gemini_api_key, settings.gemini_live_model, settings.gemini_live_voice
            )

    else:
        raise ConfigError(f"Unknown S2S_PROVIDER={provider!r} (expected openai or gemini)")
    make()  # fail at startup, not on the first call, if a key or URL is missing
    return make


def build_s2s_backends(settings: Settings, *, store: SessionStore | None = None, tools=None) -> S2SBackends:
    from .tools import build_registry

    make_adapter = build_adapter_factory(settings)
    return S2SBackends(
        make_adapter=make_adapter,
        store=store or SessionStore(),
        make_vad=vad_factory(settings),
        tools=tools if tools is not None else build_registry(settings),
        vad_threshold=settings.vad_threshold,
        turn_log=TurnLog(settings.metrics_log_path) if settings.metrics_log_path else None,
        provider=settings.s2s_provider,
        output_sample_rate=make_adapter().output_sample_rate,
        barge_in=settings.barge_in,
        barge_in_min_speech_ms=settings.barge_in_min_speech_ms,
    )
