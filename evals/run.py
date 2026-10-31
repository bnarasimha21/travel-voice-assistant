"""Replay scripted spoken conversations through the real voice pipeline.

    python -m evals.run --llm stub              # offline: Whisper tiny, Kokoro, Silero, stub LLM
    python -m evals.run --llm live              # the configured LLM endpoint (needs LLM_API_KEY)
    python -m evals.run --backends fake         # no model files at all (fake speech models)
    python -m evals.run --matrix                # every configuration in evals/matrix.yaml

Each scenario in `evals/scenarios/*.yaml` is a short conversation. Caller
audio is synthesized with Kokoro in the scenario's voice and speed (noise is
mixed in when the scenario asks for it), or read from a recorded clip, and
pushed into a real `VoiceHandler` frame by frame, the way FastRTC delivers a
browser microphone. The run checks, per turn:

- transcript word error rate against the scripted line (`max_wer`)
- which tools the model called (`expect_tools`; omit the key to skip)
- facts the answer must contain (`facts`: each entry is a phrase or a list
  of alternatives, matched case-insensitively)
- interruption handling (`interrupt_after_s`: the next line is spoken that
  many seconds into this reply; the reply must stop and be stored as
  interrupted)
- latency: the scenario's p95 time to first audio must be within
  `ttfa_budget_ms` (times `--budget-scale`)
- turn taking: ending a turn before the caller finished a line is an early
  cutoff (a line may contain `[pause 1.0]` to provoke one). If the caller
  resumes before any of the reply is heard, the handler merges both parts and
  the cutoff is "recovered": counted, never a failure. An audible cutoff (a
  reply to half a line was played) fails the scenario unless the detector is
  listed in `allow_early_cutoff`. Rates are reported per detector
  (`--turn-detector silence|smart|both`)

It prints a table and exits with status 1 if any scenario fails.

Synthetic callers are cleaner and more cooperative than real people. For local
runs, put real recordings in `evals/recordings/` (gitignored) and reference
them with `audio:` on a turn; `say` is then the reference transcript. Turns
whose recording is missing make the scenario skip.

Tool data comes from the offline sample providers in every mode, so answers
are checkable and runs do not depend on outside APIs.

`--matrix` runs the same scenarios once per configuration listed in
`evals/matrix.yaml` (local Whisper models, turn detectors, Deepgram Flux, and
the speech-to-speech providers) and prints a comparison table. Rows whose
service key is not set are shown as skipped. With `--backends fake` the hosted
services are replaced by local fakes, so the whole matrix runs offline.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from voice_agent.audio import STT_SAMPLE_RATE, add_noise, read_wav, resample
from voice_agent.backends import build_backends, build_llm, build_s2s_backends, build_stt, build_tts
from voice_agent.config import ConfigError, Settings, load_settings
from voice_agent.fakes import FakeTTS
from voice_agent.handler import VoiceHandler
from voice_agent.metrics import STAGES, percentile
from voice_agent.realtime.handler import S2SHandler
from voice_agent.session import SessionStore
from voice_agent.testing.fake_realtime import FakeDeepgramFlux, FakeGeminiLive, FakeOpenAIRealtime
from voice_agent.testing.harness import HandlerDriver
from voice_agent.testing.stub_llm import Brain, Plan, StubLLMServer, travel_brain

EVALS_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = EVALS_DIR / "scenarios"
RESULTS_DIR = EVALS_DIR / "results"
MATRIX_FILE = EVALS_DIR / "matrix.yaml"
PAUSE_MARK = re.compile(r"\[pause\s+([0-9.]+)\]")
TRAILING_SILENCE_S = 2.5  # longer than the smart detector's max-silence fallback
INPUT_RATE = 48000  # browsers send 48 kHz; the handler resamples like in production
FAKE_SERVER_SPEECH_RMS = 0.1  # the fake services' VAD gate, above 15 dB SNR noise like the -20 dB local gate


# Scenarios -------------------------------------------------------------------


@dataclass
class ScriptedTurn:
    say: str
    audio: str | None = None
    expect_tools: list[str] | None = None
    facts: list[list[str]] = field(default_factory=list)
    interrupt_after_s: float | None = None

    @property
    def text(self) -> str:
        """The line without pause marks: the reference transcript."""
        return " ".join(PAUSE_MARK.sub(" ", self.say).split())

    def parts(self) -> list[tuple[str, float]]:
        """(text, pause after it in seconds) pieces of the line."""
        pieces = PAUSE_MARK.split(self.say)
        out = []
        for i in range(0, len(pieces), 2):
            pause = float(pieces[i + 1]) if i + 1 < len(pieces) else 0.0
            out.append((pieces[i].strip(), pause))
        return [(t, p) for t, p in out if t or p]

    @property
    def pauses(self) -> int:
        return len(PAUSE_MARK.findall(self.say))


@dataclass
class Scenario:
    name: str
    path: Path
    turns: list[ScriptedTurn]
    description: str = ""
    voice: str = "af_heart"
    speed: float = 1.0
    noise_snr_db: float | None = None
    ttfa_budget_ms: float = 2000.0
    max_wer: float = 0.25
    allow_early_cutoff: list[str] = field(default_factory=list)


def _facts(raw: Any) -> list[list[str]]:
    out = []
    for fact in raw or []:
        alternatives = fact if isinstance(fact, list) else [fact]
        out.append([str(a).lower() for a in alternatives])
    return out


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    turns_raw = data.get("turns") or []
    if not turns_raw:
        raise ValueError(f"{path.name}: no turns")
    turns = []
    for i, t in enumerate(turns_raw):
        if not t.get("say"):
            raise ValueError(f"{path.name}: turn {i + 1} needs `say` (the line, or the reference transcript of `audio`)")
        tools = t.get("expect_tools")
        turns.append(
            ScriptedTurn(
                say=str(t["say"]),
                audio=t.get("audio"),
                expect_tools=None if tools is None else [str(x) for x in tools],
                facts=_facts(t.get("facts")),
                interrupt_after_s=t.get("interrupt_after_s"),
            )
        )
    for i, t in enumerate(turns):
        if t.audio and t.pauses:
            raise ValueError(f"{path.name}: turn {i + 1} has both `audio` and pause marks")
    if turns[-1].interrupt_after_s is not None:
        raise ValueError(f"{path.name}: the last turn cannot be interrupted (nothing follows it)")
    caller = data.get("caller") or {}
    return Scenario(
        name=str(data.get("name") or path.stem),
        path=path,
        turns=turns,
        description=str(data.get("description") or "").strip(),
        voice=str(caller.get("voice", "af_heart")),
        speed=float(caller.get("speed", 1.0)),
        noise_snr_db=caller.get("noise_snr_db"),
        ttfa_budget_ms=float(data.get("ttfa_budget_ms", 2000)),
        max_wer=float(data.get("max_wer", 0.25)),
        allow_early_cutoff=[str(d) for d in data.get("allow_early_cutoff") or []],
    )


def load_scenarios(directory: Path = SCENARIOS_DIR, only: list[str] | None = None) -> list[Scenario]:
    scenarios = [load_scenario(p) for p in sorted(directory.glob("*.yaml"))]
    if only:
        unknown = set(only) - {s.name for s in scenarios}
        if unknown:
            raise ValueError(f"unknown scenario(s): {', '.join(sorted(unknown))}")
        scenarios = [s for s in scenarios if s.name in only]
    return scenarios


# Scoring ---------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalise_words(text: str) -> list[str]:
    text = text.lower().replace("'", "")
    words = re.findall(r"[a-z0-9]+", text)
    return [_NUMBER_WORDS.get(w, w) for w in words]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """(substitutions + deletions + insertions) / reference words, on normalised words."""
    ref, hyp = normalise_words(reference), normalise_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def missing_facts(answer: str, facts: list[list[str]]) -> list[str]:
    text = answer.lower()
    return [" / ".join(alts) for alts in facts if not any(a in text for a in alts)]


# Running ---------------------------------------------------------------------


@dataclass
class Options:
    llm: str = "stub"  # stub | live
    backends: str = "real"  # real | fake
    stt_model: str | None = None  # None: STT_MODEL from evals/matrix.yaml `base`
    repeats: int = 1
    realtime: bool = False
    budget_scale: float = 1.0
    stub_delay_ms: float = 0.0
    turn_detector: str | None = None  # silence | smart | both; None uses TURN_DETECTOR
    matrix: bool = False  # run every configuration in matrix_file
    matrix_file: Path = MATRIX_FILE
    judge: bool = False
    scenarios: list[str] | None = None
    scenarios_dir: Path = SCENARIOS_DIR
    env: dict[str, str] | None = None  # replaces os.environ (tests)
    env_files: tuple[Path, ...] | None = None
    brain: Brain | None = None  # stub LLM behaviour; default is the scripted travel agent
    save: bool = True  # write evals/results/<time>.json


@dataclass
class TurnResult:
    say: str
    heard: str
    answer: str
    wer: float
    tools: list[str]
    outcome: str
    ttfa_ms: float | None
    stages_ms: dict[str, float | None]
    failures: list[str]
    judge: dict[str, Any] | None = None


@dataclass
class ScenarioResult:
    name: str
    status: str  # pass | fail | skip
    turns: list[TurnResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    ttfa_p50_ms: float | None = None
    ttfa_p95_ms: float | None = None
    budget_ms: float = 0.0
    detector: str = ""  # who decided turns: silence, smart, flux, or the s2s provider
    config: str = ""  # the configuration name (the detector name outside --matrix)
    scripted_turns: int = 0
    early_cutoffs: int = 0
    audible_cutoffs: int = 0

    @property
    def label(self) -> str:
        tag = self.config or self.detector
        return f"{self.name} [{tag}]" if tag else self.name

    @property
    def max_wer(self) -> float | None:
        return max((t.wer for t in self.turns), default=None)


@dataclass
class EvalReport:
    options: dict[str, Any]
    scenarios: list[ScenarioResult]
    skipped_configs: dict[str, str] = field(default_factory=dict)  # name -> reason

    @property
    def passed(self) -> bool:
        return all(s.status != "fail" for s in self.scenarios) and any(s.status == "pass" for s in self.scenarios)

    def all_turns(self) -> list[TurnResult]:
        return [t for s in self.scenarios for t in s.turns]

    def summary(self) -> dict[str, Any]:
        turns = self.all_turns()
        ttfa = [t.ttfa_ms for t in turns if t.ttfa_ms is not None]
        stages = {}
        for stage in STAGES:
            values = [t.stages_ms.get(stage) for t in turns if t.stages_ms.get(stage) is not None]
            stages[stage] = round(statistics.median(values), 1) if values else None
        configs: dict[str, dict[str, Any]] = {}
        for name in dict.fromkeys(s.config for s in self.scenarios):
            runs = [s for s in self.scenarios if s.config == name and s.status != "skip"]
            scripted = sum(s.scripted_turns for s in runs)
            cutoffs = sum(s.early_cutoffs for s in runs)
            audible = sum(s.audible_cutoffs for s in runs)
            det_ttfa = [t.ttfa_ms for s in runs for t in s.turns if t.ttfa_ms is not None]
            configs[name] = {
                "detector": runs[0].detector if runs else "",
                "scenarios": len(runs),
                "passed": sum(s.status == "pass" for s in runs),
                "pass_rate": round(sum(s.status == "pass" for s in runs) / len(runs), 3) if runs else None,
                "scripted_turns": scripted,
                "early_cutoffs": cutoffs,
                "early_cutoff_rate": round(cutoffs / scripted, 3) if scripted else None,
                "audible_cutoffs": audible,
                "audible_cutoff_rate": round(audible / scripted, 3) if scripted else None,
                "ttfa_p50_ms": percentile(det_ttfa, 50),
                "ttfa_p95_ms": percentile(det_ttfa, 95),
            }
        return {
            "turns": len(ttfa),
            "ttfa_p50_ms": percentile(ttfa, 50),
            "ttfa_p95_ms": percentile(ttfa, 95),
            "median_stages_ms": stages,
            "configs": configs,
            "skipped_configs": dict(self.skipped_configs),
        }

    def to_json(self) -> dict[str, Any]:
        def turn(t: TurnResult) -> dict[str, Any]:
            return t.__dict__

        return {
            "options": self.options,
            "passed": self.passed,
            "summary": self.summary(),
            "scenarios": [
                {
                    "name": s.name,
                    "config": s.config,
                    "detector": s.detector,
                    "early_cutoffs": s.early_cutoffs,
                    "audible_cutoffs": s.audible_cutoffs,
                    "status": s.status,
                    "failures": s.failures,
                    "ttfa_p50_ms": s.ttfa_p50_ms,
                    "ttfa_p95_ms": s.ttfa_p95_ms,
                    "budget_ms": s.budget_ms,
                    "turns": [turn(t) for t in s.turns],
                }
                for s in self.scenarios
            ],
        }


class Caller:
    """Turns a scripted line into microphone audio at INPUT_RATE."""

    def __init__(self, tts: Any, fake: bool):
        self.tts = tts
        self.fake = fake

    async def line(self, scenario: Scenario, turn: ScriptedTurn, seed: int) -> np.ndarray:
        if turn.audio:
            rate, audio = read_wav(EVALS_DIR / turn.audio)
            audio = resample(audio, rate, INPUT_RATE)
        else:
            pieces = []
            for text, pause in turn.parts():
                if text:
                    pieces.append(await self._speak(scenario, text))
                pieces.append(np.zeros(int(pause * INPUT_RATE), dtype=np.float32))
            audio = np.concatenate(pieces)
        return audio

    async def _speak(self, scenario: Scenario, text: str) -> np.ndarray:
        if self.fake:
            # The fake recogniser decodes fixed-length tones, so voice and speed do not apply.
            return await FakeTTS(sample_rate=INPUT_RATE).synthesize(text)
        audio = await self.tts.synthesize(text, voice=scenario.voice, speed=scenario.speed)
        return resample(audio, self.tts.sample_rate, INPUT_RATE)


@dataclass
class RunConfig:
    """One column of the comparison: settings layered over the base configuration."""

    name: str
    env: dict[str, str] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    realtime: bool = False
    fake_server: str | None = None


# fake_server name -> (server class, URL variable, key variable)
FAKE_SERVERS = {
    "deepgram": (FakeDeepgramFlux, "DEEPGRAM_URL", "DEEPGRAM_API_KEY"),
    "openai": (FakeOpenAIRealtime, "OPENAI_REALTIME_URL", "OPENAI_API_KEY"),
    "gemini": (FakeGeminiLive, "GEMINI_LIVE_URL", "GEMINI_API_KEY"),
}


def load_matrix(path: Path, backends: str) -> tuple[dict[str, str], list[RunConfig]]:
    """The `base` settings and the `matrix` configurations for `--backends real|fake`."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = data.get(backends)
    if not isinstance(section, dict):
        raise ConfigError(f"{path.name} has no `{backends}` section")
    base = {str(k): str(v) for k, v in (section.get("base") or {}).items()}
    configs: list[RunConfig] = []
    for raw in section.get("matrix") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ConfigError(f"{path.name}: every `{backends}.matrix` entry needs a name")
        server = raw.get("fake_server")
        if server is not None and server not in FAKE_SERVERS:
            raise ConfigError(f"{path.name}: {raw['name']}: fake_server must be one of {', '.join(FAKE_SERVERS)}")
        configs.append(
            RunConfig(
                name=str(raw["name"]),
                env={str(k): str(v) for k, v in (raw.get("env") or {}).items()},
                requires=[str(r) for r in raw.get("requires") or []],
                realtime=bool(raw.get("realtime", False)),
                fake_server=server,
            )
        )
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ConfigError(f"{path.name}: configuration names in `{backends}.matrix` must be unique")
    return base, configs


def _settings(
    options: Options,
    stub: StubLLMServer | None,
    config: RunConfig | None = None,
    server_env: dict[str, str] | None = None,
) -> Settings:
    """Config files and environment, then the matrix `base`, the configuration, and the
    eval's fixed choices (sample tool data, no turn log, the stub LLM) on top."""
    if options.backends not in ("real", "fake"):
        raise ConfigError(f"--backends must be real or fake, not {options.backends!r}")
    if options.llm not in ("stub", "live"):
        raise ConfigError(f"--llm must be stub or live, not {options.llm!r}")
    base, _ = load_matrix(options.matrix_file, options.backends)
    env = dict(os.environ if options.env is None else options.env)
    env.update(base)
    if config is not None:
        env.update(config.env)
    env.update(server_env or {})
    overrides: dict[str, Any] = {
        "weather_provider": "fixture",
        "flights_provider": "fixture",
        "hotels_provider": "fixture",
        "metrics_log": "none",
    }
    if options.stt_model and options.backends == "real" and not (config and "STT_MODEL" in config.env):
        overrides["stt_model"] = options.stt_model
    if options.llm == "stub":
        assert stub is not None
        overrides.update(llm_base_url=stub.base_url, llm_model="stub-model", llm_api_key="stub", llm_extra_query="")
    settings = load_settings(env=env, env_files=options.env_files, **overrides)
    if options.llm == "live" and settings.pipeline != "s2s":
        settings.require("llm_base_url", "llm_model", "llm_api_key")
    return settings


def _run_configs(options: Options, settings: Settings) -> list[RunConfig]:
    if options.matrix:
        if options.turn_detector:
            raise ConfigError("--matrix sets the turn detector per configuration; drop --turn-detector")
        _, configs = load_matrix(options.matrix_file, options.backends)
        if not configs:
            raise ConfigError(f"{options.matrix_file.name} lists no `{options.backends}.matrix` configurations")
        return configs
    td = options.turn_detector
    if td == "both":
        detectors = ["silence", "smart"]
    elif td in (None, ""):
        detectors = [settings.turn_detector]
    elif td in ("silence", "smart"):
        detectors = [td]
    else:
        raise ConfigError(f"--turn-detector must be silence, smart or both, not {td!r}")
    return [RunConfig(d, {"TURN_DETECTOR": d}) for d in detectors]


def _detector(settings: Settings) -> str:
    """Who decides when the caller finished, for cutoff reporting and `allow_early_cutoff`."""
    if settings.pipeline == "s2s":
        return settings.s2s_provider
    if settings.stt_backend == "deepgram-flux":
        return "flux"
    return settings.turn_detector


MakeHandler = Callable[[SessionStore, str, bool], Any]


async def _handler_factory(settings: Settings, tts: Any, llm: Any, stt_cache: dict[tuple, Any], fake: bool) -> MakeHandler:
    """Build a configuration's backends once; each scenario run gets a handler with its own store."""
    if settings.pipeline == "s2s":
        s2s = build_s2s_backends(settings)

        def make_s2s(store: SessionStore, session_id: str, pace: bool) -> S2SHandler:
            return S2SHandler(replace(s2s, store=store), session_id=session_id, pace_output=pace)

        return make_s2s
    key = (settings.stt_backend, settings.stt_model, settings.stt_device, settings.stt_compute_type)
    stt = stt_cache.get(key)
    if stt is None:
        stt = build_stt(settings)
        if not getattr(stt, "streaming", False):
            if not fake:
                # Load the weights now so the first scenario is not charged for it.
                await stt.transcribe(np.zeros(STT_SAMPLE_RATE // 2, dtype=np.float32))
            stt_cache[key] = stt
    backends = build_backends(settings, stt=stt, tts=tts, llm=llm)
    await backends.warm_fillers()

    def make_cascade(store: SessionStore, session_id: str, pace: bool) -> VoiceHandler:
        return VoiceHandler(replace(backends, store=store), session_id=session_id, pace_output=pace)

    return make_cascade


def delayed(brain: Brain, delay_ms: float) -> Brain:
    """Wrap a stub brain so every reply waits `delay_ms` before its first token."""

    def wrapped(body: dict[str, Any]) -> Plan:
        plan = brain(body)
        plan.first_token_delay_s += delay_ms / 1000
        return plan

    return wrapped


async def _wait(predicate: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


async def _stop(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def run_scenario(scenario: Scenario, shared: dict[str, Any], options: Options, seed: int = 0) -> ScenarioResult:
    detector = shared["detector"]
    result = ScenarioResult(
        name=scenario.name,
        status="pass",
        budget_ms=scenario.ttfa_budget_ms * options.budget_scale,
        detector=detector,
        config=shared["config"],
        scripted_turns=len(scenario.turns),
    )
    missing = [t.audio for t in scenario.turns if t.audio and not (EVALS_DIR / t.audio).exists()]
    if missing:
        result.status = "skip"
        result.failures.append("recording not found: " + ", ".join(missing))
        return result

    store = SessionStore()
    interrupts = any(t.interrupt_after_s is not None for t in scenario.turns)
    # Barge-in needs real-time playback so the handler knows how much was heard.
    handler = shared["make_handler"](store, f"eval-{scenario.name}-{seed}", interrupts)
    realtime = options.realtime or shared["realtime"]
    driver = HandlerDriver(handler, session_id=handler._session_id, input_rate=INPUT_RATE)
    await driver.start()
    caller: Caller = shared["caller"]
    try:
        for turn in scenario.turns:
            audio = await caller.line(scenario, turn, seed)
            lead = np.zeros(int(0.2 * INPUT_RATE), dtype=np.float32)
            tail = np.zeros(int(TRAILING_SILENCE_S * INPUT_RATE), dtype=np.float32)
            clip = np.concatenate([lead, audio, tail])
            if scenario.noise_snr_db is not None:
                clip = add_noise(clip, float(scenario.noise_snr_db), seed=seed)
            before = len(handler.turn_records)
            # The trailing silence keeps streaming while the reply plays (a live
            # microphone never stops); the caller's next line replaces it.
            feeding = asyncio.create_task(driver.feed(clip, realtime=realtime))
            if turn.interrupt_after_s is None:
                await feeding
            try:
                await _wait(
                    lambda: handler._current is not None or len(handler.turn_records) > before,  # noqa: B023
                    10 + (clip.size / INPUT_RATE if realtime else 0),
                    "the turn to end",
                )
            except TimeoutError:
                await _stop(feeding)
                result.failures.append(f"no turn ended after {turn.text!r}")
                continue
            if turn.interrupt_after_s is not None:
                try:
                    await _wait(
                        lambda: handler._current is not None and handler._current.first_audio_at is not None,
                        60,
                        "reply audio to start",
                    )
                    await asyncio.sleep(turn.interrupt_after_s)
                finally:
                    await _stop(feeding)
                if handler._current is None:
                    result.failures.append(f"reply to {turn.text!r} ended before the interruption point")
            else:
                await driver.wait_idle(timeout=120)
        session = store.peek(handler._session_id)
        history = list(session.turns) if session else []  # read before close drops the session
    finally:
        await driver.close()

    replies = [r for r in handler.turn_records if r["outcome"] != "empty"]
    # Every reply beyond one per scripted line means a turn ended mid-line.
    result.early_cutoffs = max(0, len(replies) - len(scenario.turns))
    # A cut-off reply nobody heard is merged into the next one ("merged"), so
    # dropping those leaves one record per scripted line again.
    records = [r for r in replies if r["outcome"] != "merged"]
    result.audible_cutoffs = max(0, len(records) - len(scenario.turns))
    tolerated = result.early_cutoffs > 0 and detector in scenario.allow_early_cutoff
    outcomes = ", ".join(r["outcome"] for r in handler.turn_records) or "none"
    if result.audible_cutoffs and not tolerated:
        result.failures.append(
            f"{result.audible_cutoffs} audible early cutoff(s) with the {detector} detector ({outcomes})"
        )
    if len(records) != len(scenario.turns) or len(history) != len(scenario.turns):
        if tolerated:
            return result  # the cut split the conversation; only the cutoff is reported
        result.failures.append(
            f"expected {len(scenario.turns)} turns, got {len(replies)} replies ({outcomes}) "
            f"and {len(history)} history entries"
        )
    for i, (turn, record, entry) in enumerate(zip(scenario.turns, records, history), 1):
        fails: list[str] = []
        wer = word_error_rate(turn.text, entry.user)
        if wer > scenario.max_wer:
            fails.append(f"WER {wer:.2f} > {scenario.max_wer:.2f} (heard {entry.user!r})")
        if turn.expect_tools is not None and sorted(set(record["tools"])) != sorted(set(turn.expect_tools)):
            fails.append(f"tools {record['tools']} != expected {turn.expect_tools}")
        if turn.interrupt_after_s is not None:
            if record["outcome"] != "interrupted" or not entry.interrupted:
                fails.append(f"expected the reply to be interrupted, outcome was {record['outcome']}")
        else:
            if record["outcome"] != "completed":
                fails.append(f"reply outcome {record['outcome']}")
            gaps = missing_facts(entry.assistant, turn.facts)
            if gaps:
                fails.append(f"answer lacks {gaps}: {entry.assistant!r}")
        if record["ttfa_ms"] is None:
            fails.append("no reply audio was sent")
        result.turns.append(
            TurnResult(
                say=turn.text,
                heard=entry.user,
                answer=entry.assistant,
                wer=round(wer, 3),
                tools=list(record["tools"]),
                outcome=record["outcome"],
                ttfa_ms=record["ttfa_ms"],
                stages_ms=dict(record["stages_ms"]),
                failures=fails,
            )
        )
        result.failures.extend(f"turn {i}: {f}" for f in fails)
    if result.failures:
        result.status = "fail"
    return result


def _apply_budget(result: ScenarioResult) -> None:
    ttfa = [t.ttfa_ms for t in result.turns if t.ttfa_ms is not None]
    result.ttfa_p50_ms = percentile(ttfa, 50)
    result.ttfa_p95_ms = percentile(ttfa, 95)
    if result.status != "skip" and result.ttfa_p95_ms is not None and result.ttfa_p95_ms > result.budget_ms:
        result.failures.append(f"TTFA p95 {result.ttfa_p95_ms:.0f} ms exceeds budget {result.budget_ms:.0f} ms")
        result.status = "fail"


async def _run_config(
    config: RunConfig,
    options: Options,
    stub: StubLLMServer | None,
    scenarios: list[Scenario],
    caller: Caller,
    cache: dict[str, Any],
) -> list[ScenarioResult]:
    server = None
    server_env: dict[str, str] = {}
    if config.fake_server:
        cls, url_var, key_var = FAKE_SERVERS[config.fake_server]
        brain = options.brain or travel_brain
        if options.stub_delay_ms:
            brain = delayed(brain, options.stub_delay_ms)
        server = await cls(brain=brain, speech_rms=FAKE_SERVER_SPEECH_RMS).start()
        server_env = {url_var: server.url, key_var: server.api_key}
    try:
        settings = _settings(options, stub, config, server_env)
        missing = [name for name in config.requires if not getattr(settings, name.lower(), "")]
        if missing:
            raise _Skip("needs " + ", ".join(missing))
        if "tts" not in cache:
            cache["tts"] = build_tts(settings)
            if options.backends == "real":
                await cache["tts"].synthesize("Warming up.")
        if "llm" not in cache and settings.pipeline != "s2s":
            cache["llm"] = build_llm(settings)
        make_handler = await _handler_factory(
            settings, cache["tts"], cache.get("llm"), cache.setdefault("stt", {}), options.backends == "fake"
        )
        shared = {
            "make_handler": make_handler,
            "caller": caller,
            "detector": _detector(settings),
            "config": config.name,
            "realtime": config.realtime,
        }
        results: list[ScenarioResult] = []
        for scenario in scenarios:
            merged: ScenarioResult | None = None
            for seed in range(options.repeats):
                one = await run_scenario(scenario, shared, options, seed)
                if merged is None:
                    merged = one
                else:
                    merged.turns.extend(one.turns)
                    merged.failures.extend(f"repeat {seed + 1}: {f}" for f in one.failures)
                    merged.early_cutoffs += one.early_cutoffs
                    merged.audible_cutoffs += one.audible_cutoffs
                    merged.scripted_turns += one.scripted_turns
                    if one.status == "fail":
                        merged.status = "fail"
                if one.status == "skip":
                    break
            assert merged is not None
            _apply_budget(merged)
            results.append(merged)
        return results
    finally:
        if server is not None:
            await server.stop()


class _Skip(Exception):
    """A configuration that cannot run here (a missing key or model setting)."""


async def run_eval(options: Options) -> EvalReport:
    if options.judge and options.llm != "live":
        raise ConfigError("--judge needs --llm live (the grader is an LLM call)")
    scenarios = load_scenarios(options.scenarios_dir, options.scenarios)
    stub = None
    if options.llm == "stub":
        brain = options.brain or travel_brain
        if options.stub_delay_ms:
            brain = delayed(brain, options.stub_delay_ms)
        stub = StubLLMServer(brain=brain).start()
    results: list[ScenarioResult] = []
    skipped: dict[str, str] = {}
    try:
        settings = _settings(options, stub)
        configs = _run_configs(options, settings)
        fake = options.backends == "fake"
        cache: dict[str, Any] = {"tts": build_tts(settings)}
        if not fake:
            await cache["tts"].synthesize("Warming up.")
        caller = Caller(cache["tts"], fake)
        for config in configs:
            try:
                results.extend(await _run_config(config, options, stub, scenarios, caller, cache))
            except _Skip as exc:
                skipped[config.name] = str(exc)
            except ConfigError as exc:
                if not options.matrix:
                    raise
                skipped[config.name] = f"configuration error: {exc}"
        if options.judge:
            from .judge import judge_report

            await judge_report(results, settings)
    finally:
        if stub is not None:
            stub.stop()
    opts = {
        "llm": options.llm,
        "backends": options.backends,
        "matrix": options.matrix,
        "stt_model": (options.stt_model or settings.stt_model) if options.backends == "real" else "fake",
        "repeats": options.repeats,
        "realtime": options.realtime,
        "budget_scale": options.budget_scale,
        "stub_delay_ms": options.stub_delay_ms,
        "configs": [c.name for c in configs],
    }
    return EvalReport(options=opts, scenarios=results, skipped_configs=skipped)


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def format_report(report: EvalReport) -> str:
    rows = [("scenario", "status", "turns", "cutoffs (audible)", "max WER", "TTFA p50", "TTFA p95", "budget")]
    for s in report.scenarios:
        wer = "-" if s.max_wer is None else f"{s.max_wer:.2f}"
        rows.append(
            (s.label, s.status.upper(), str(len(s.turns)), f"{s.early_cutoffs} ({s.audible_cutoffs})", wer, _ms(s.ttfa_p50_ms), _ms(s.ttfa_p95_ms), _ms(s.budget_ms))
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    summary = report.summary()
    lines.append("")
    lines.append(
        f"All turns: TTFA p50 {_ms(summary['ttfa_p50_ms'])} ms, p95 {_ms(summary['ttfa_p95_ms'])} ms over {summary['turns']} turns"
    )
    for name, d in summary["configs"].items():
        rate = "-" if d["early_cutoff_rate"] is None else f"{d['early_cutoff_rate']:.0%}"
        lines.append(
            f"{name} ({d['detector']} decides turns): early cutoffs {d['early_cutoffs']} of {d['scripted_turns']} turns ({rate}, "
            f"{d['audible_cutoffs']} audible), "
            f"TTFA p50 {_ms(d['ttfa_p50_ms'])} ms, p95 {_ms(d['ttfa_p95_ms'])} ms"
        )
    stages = summary["median_stages_ms"]
    lines.append("Median stage offsets (ms after end of speech): " + ", ".join(f"{k} {_ms(v)}" for k, v in stages.items()))
    for s in report.scenarios:
        for f in s.failures:
            lines.append(f"{s.label}: {f}")
        for t in s.turns:
            if t.judge:
                lines.append(f"{s.label}: judge {t.judge}")
    if report.options.get("matrix"):
        lines.append("")
        lines.append(format_matrix(report))
    lines.append("")
    lines.append("PASS" if report.passed else "FAIL")
    return "\n".join(lines)


def format_matrix(report: EvalReport) -> str:
    """One row per configuration: the comparison table for the README."""
    summary = report.summary()
    rows = [("config", "passed", "pass rate", "TTFA p50 ms", "TTFA p95 ms", "cutoffs (audible)")]
    for name, d in summary["configs"].items():
        rate = "-" if d["pass_rate"] is None else f"{d['pass_rate']:.0%}"
        rows.append(
            (name, f"{d['passed']}/{d['scenarios']}", rate, _ms(d["ttfa_p50_ms"]), _ms(d["ttfa_p95_ms"]), f"{d['early_cutoffs']} ({d['audible_cutoffs']})")
        )
    for name, reason in report.skipped_configs.items():
        rows.append((name, "skipped", "-", "-", "-", reason))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    out.insert(1, "| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> Options:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm", choices=["stub", "live"], default="stub")
    parser.add_argument("--backends", choices=["real", "fake"], default="real", help="real: Whisper, Kokoro, Silero")
    parser.add_argument("--stt-model", help="Whisper model id for --backends real (default: STT_MODEL in evals/matrix.yaml)")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="run only this scenario (repeatable)")
    parser.add_argument("--repeats", type=int, default=1, help="run every scenario this many times")
    parser.add_argument("--realtime", action="store_true", help="feed caller audio at real time (slower, closest to a browser)")
    parser.add_argument("--budget-scale", type=float, default=1.0, help="multiply every TTFA budget (slow CI machines)")
    parser.add_argument(
        "--turn-detector", choices=["silence", "smart", "both"], help="default: TURN_DETECTOR from the config"
    )
    parser.add_argument("--matrix", action="store_true", help="run every configuration in evals/matrix.yaml")
    parser.add_argument("--stub-delay-ms", type=float, default=0.0, help="extra first-token delay for the stub LLM")
    parser.add_argument("--judge", action="store_true", help="grade answers with the LLM (live mode only)")
    parser.add_argument("--no-save", action="store_true", help="do not write evals/results/<time>.json")
    args = parser.parse_args(argv)
    options = Options(
        llm=args.llm,
        backends=args.backends,
        stt_model=args.stt_model,
        repeats=max(1, args.repeats),
        realtime=args.realtime,
        budget_scale=args.budget_scale,
        stub_delay_ms=args.stub_delay_ms,
        turn_detector=args.turn_detector,
        matrix=args.matrix,
        judge=args.judge,
        scenarios=args.scenarios,
        save=not args.no_save,
    )
    return options


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        report = asyncio.run(run_eval(options))
    except (ConfigError, ValueError) as exc:
        print(f"eval error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    if options.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        kind = "matrix" if options.matrix else options.backends
        out = RESULTS_DIR / f"{stamp}-{options.llm}-{kind}.json"
        out.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        print(f"results: {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
