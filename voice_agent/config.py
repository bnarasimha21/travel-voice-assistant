"""Runtime configuration, read from the environment.

Values are resolved in this order: real environment variables, then `.env`,
then `.env.example`. The example file carries every default (URLs, model
names, voices), so nothing deployment specific is hardcoded in Python and the
app starts with the example values plus an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_query(value: str) -> dict[str, str]:
    """Parse `a=1&b=2` into a dict (used for extra LLM query params)."""
    out: dict[str, str] = {}
    for part in value.split("&"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ConfigError(f"LLM_EXTRA_QUERY entry {part!r} is not key=value")
        key, val = part.split("=", 1)
        out[key.strip()] = val.strip()
    return out


@dataclass(frozen=True)
class Settings:
    """All tunables. Field names map to upper-case env vars (`llm_model` -> `LLM_MODEL`)."""

    # Which pipeline serves the browser: cascade (STT, LLM, TTS here) or s2s
    # (a speech-to-speech realtime model, see voice_agent/realtime/).
    pipeline: str = "cascade"

    # LLM: any OpenAI-compatible chat completions endpoint.
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_extra_query: str = ""
    llm_timeout_s: float = 30.0

    # Travel tools.
    tools: str = "weather,flights,hotels"
    tool_timeout_s: float = 8.0
    max_tool_rounds: int = 4
    filler_after_ms: int = 300
    weather_provider: str = "open-meteo"
    flights_provider: str = "fixture"
    hotels_provider: str = "fixture"
    open_meteo_geocoding_url: str = ""
    open_meteo_forecast_url: str = ""
    duffel_api_key: str = ""
    duffel_base_url: str = ""

    # Speech to text.
    stt_backend: str = ""
    stt_model: str = ""
    stt_device: str = "auto"
    stt_language: str = "en"
    stt_compute_type: str = "default"  # faster-whisper: int8, float16, ...

    # Deepgram Flux (STT_BACKEND=deepgram-flux): streaming STT with end-of-turn detection.
    deepgram_url: str = ""
    deepgram_model: str = ""
    deepgram_api_key: str = ""
    deepgram_eot_threshold: float = 0.7
    deepgram_eot_timeout_ms: int = 5000

    # Speech to speech (PIPELINE=s2s).
    s2s_provider: str = ""
    openai_api_key: str = ""
    openai_realtime_url: str = ""
    openai_realtime_model: str = ""
    openai_realtime_voice: str = ""
    openai_realtime_transcribe_model: str = ""
    openai_realtime_turn_detection: str = ""
    gemini_api_key: str = ""
    gemini_live_url: str = ""
    gemini_live_model: str = ""
    gemini_live_voice: str = ""

    # Text to speech.
    tts_backend: str = ""
    tts_model_path: str = ""
    tts_voices_path: str = ""
    tts_voice: str = ""
    tts_speed: float = 1.0
    tts_lang: str = "en-us"

    # Voice activity detection and turn taking.
    vad_backend: str = "silero"
    vad_threshold: float = 0.5
    energy_vad_threshold_db: float = -40.0
    silence_ms: int = 700
    min_speech_ms: int = 250
    max_turn_s: float = 30.0
    barge_in: bool = True
    # Turn detector: silence (pause threshold) or smart (Smart Turn model at a short pause).
    turn_detector: str = "silence"
    smart_turn_backend: str = "onnx"
    smart_turn_model_path: str = ""
    smart_turn_pause_ms: int = 200
    smart_turn_max_silence_ms: int = 2000
    smart_turn_threshold: float = 0.5
    barge_in_min_speech_ms: int = 300

    # Per-turn latency log (JSON lines). Relative paths are under the repo; none disables it.
    metrics_log: str = "logs/turns.jsonl"

    # Web server.
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    turn_server: str = ""

    @property
    def metrics_log_path(self) -> Path | None:
        value = self.metrics_log.strip()
        if value.lower() in ("", "none", "off"):
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def llm_query(self) -> dict[str, str]:
        return _as_query(self.llm_extra_query) if self.llm_extra_query else {}

    def require(self, *names: str) -> None:
        missing = [n.upper() for n in names if not str(getattr(self, n)).strip()]
        if missing:
            raise ConfigError(
                "Missing configuration: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill these in."
            )


def _coerce(field_type: Any, raw: str) -> Any:
    type_name = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", "")
    if type_name == "bool":
        return _as_bool(raw)
    if type_name == "int":
        return int(raw)
    if type_name == "float":
        return float(raw)
    return raw


def load_settings(
    env: Mapping[str, str] | None = None,
    env_files: tuple[Path, ...] | None = None,
    **overrides: Any,
) -> Settings:
    """Build Settings from env files, the process environment and overrides.

    `env` replaces `os.environ` (tests pass a dict); `env_files` defaults to
    the repo's `.env` and `.env.example`. Keyword overrides win over all.
    """
    if env_files is None:
        env_files = (REPO_ROOT / ".env", REPO_ROOT / ".env.example")
    merged: dict[str, str] = {}
    # Lowest priority first: later files are overridden by earlier ones.
    for path in reversed(env_files):
        if path.exists():
            merged.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    merged.update(dict(os.environ if env is None else env))

    kwargs: dict[str, Any] = {}
    for f in fields(Settings):
        raw = merged.get(f.name.upper())
        if raw is None or raw == "":
            continue
        try:
            kwargs[f.name] = _coerce(f.type, raw)
        except ValueError as exc:
            raise ConfigError(f"{f.name.upper()}={raw!r}: {exc}") from exc
    kwargs.update(overrides)
    return Settings(**kwargs)
