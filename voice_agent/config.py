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

    # Which pipeline serves the browser.
    pipeline: str = "cascade"

    # LLM: any OpenAI-compatible chat completions endpoint.
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_extra_query: str = ""
    llm_timeout_s: float = 30.0

    # Speech to text.
    stt_backend: str = ""
    stt_model: str = ""
    stt_device: str = "auto"
    stt_language: str = "en"

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

    # Web server.
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    turn_server: str = ""

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
