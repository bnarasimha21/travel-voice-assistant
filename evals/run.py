"""Replay scripted spoken conversations through the real voice pipeline.

    python -m evals.run --llm stub              # offline: Whisper tiny, Kokoro, Silero, stub LLM
    python -m evals.run --llm live              # the configured LLM endpoint (needs LLM_API_KEY)
    python -m evals.run --backends fake         # no model files at all (fake speech models)

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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from voice_agent.audio import STT_SAMPLE_RATE, add_noise, read_wav, resample
from voice_agent.backends import build_backends, build_llm, build_stt, build_tts
from voice_agent.config import ConfigError, Settings, load_settings
from voice_agent.fakes import FakeTTS
from voice_agent.handler import Backends, VoiceHandler
from voice_agent.metrics import STAGES, percentile
from voice_agent.session import SessionStore
from voice_agent.testing.harness import HandlerDriver
from voice_agent.testing.stub_llm import Brain, Plan, StubLLMServer, travel_brain

EVALS_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = EVALS_DIR / "scenarios"
RESULTS_DIR = EVALS_DIR / "results"
DEFAULT_STT_MODEL = "openai/whisper-tiny"
PAUSE_MARK = re.compile(r"\[pause\s+([0-9.]+)\]")
TRAILING_SILENCE_S = 2.5  # longer than the smart detector's max-silence fallback
INPUT_RATE = 48000  # browsers send 48 kHz; the handler resamples like in production
FAKE_ENERGY_THRESHOLD_DB = -20.0  # fake speech is a loud tone; keep 15 dB SNR noise below the gate


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
    stt_model: str = DEFAULT_STT_MODEL
    repeats: int = 1
    realtime: bool = False
    budget_scale: float = 1.0
    stub_delay_ms: float = 0.0
    turn_detector: str | None = None  # silence | smart | both; None uses TURN_DETECTOR
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
    detector: str = ""
    scripted_turns: int = 0
    early_cutoffs: int = 0
    audible_cutoffs: int = 0

    @property
    def label(self) -> str:
        return f"{self.name} [{self.detector}]" if self.detector else self.name

    @property
    def max_wer(self) -> float | None:
        return max((t.wer for t in self.turns), default=None)


@dataclass
class EvalReport:
    options: dict[str, Any]
    scenarios: list[ScenarioResult]

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
        detectors: dict[str, dict[str, Any]] = {}
        for det in dict.fromkeys(s.detector for s in self.scenarios):
            runs = [s for s in self.scenarios if s.detector == det and s.status != "skip"]
            scripted = sum(s.scripted_turns for s in runs)
            cutoffs = sum(s.early_cutoffs for s in runs)
            audible = sum(s.audible_cutoffs for s in runs)
            det_ttfa = [t.ttfa_ms for s in runs for t in s.turns if t.ttfa_ms is not None]
            detectors[det] = {
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
            "detectors": detectors,
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


def _settings(options: Options, stub: StubLLMServer | None, detector: str | None = None) -> Settings:
    overrides: dict[str, Any] = {
        "weather_provider": "fixture",
        "flights_provider": "fixture",
        "hotels_provider": "fixture",
        "metrics_log": "none",
    }
    if options.backends == "fake":
        overrides.update(
            stt_backend="fake",
            tts_backend="fake",
            vad_backend="energy",
            energy_vad_threshold_db=FAKE_ENERGY_THRESHOLD_DB,
            smart_turn_backend="fake",
        )
    elif options.backends == "real":
        overrides.update(stt_backend="transformers", stt_model=options.stt_model)
    else:
        raise ConfigError(f"--backends must be real or fake, not {options.backends!r}")
    if options.llm == "stub":
        assert stub is not None
        overrides.update(llm_base_url=stub.base_url, llm_model="stub-model", llm_api_key="stub", llm_extra_query="")
    elif options.llm != "live":
        raise ConfigError(f"--llm must be stub or live, not {options.llm!r}")
    if detector:
        overrides["turn_detector"] = detector
    settings = load_settings(env=options.env, env_files=options.env_files, **overrides)
    if options.llm == "live":
        settings.require("llm_base_url", "llm_model", "llm_api_key")
    return settings


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


async def run_scenario(scenario: Scenario, shared: dict[str, Any], options: Options, seed: int = 0) -> ScenarioResult:
    detector = shared["detector"]
    result = ScenarioResult(
        name=scenario.name,
        status="pass",
        budget_ms=scenario.ttfa_budget_ms * options.budget_scale,
        detector=detector,
        scripted_turns=len(scenario.turns),
    )
    missing = [t.audio for t in scenario.turns if t.audio and not (EVALS_DIR / t.audio).exists()]
    if missing:
        result.status = "skip"
        result.failures.append("recording not found: " + ", ".join(missing))
        return result

    base: Backends = shared["backends"]
    backends = Backends(**{**base.__dict__, "store": SessionStore()})
    interrupts = any(t.interrupt_after_s is not None for t in scenario.turns)
    # Barge-in needs real-time playback so the handler knows how much was heard.
    handler = VoiceHandler(backends, session_id=f"eval-{scenario.name}-{seed}", pace_output=interrupts)
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
            await driver.feed(clip, realtime=options.realtime)
            try:
                await _wait(
                    lambda: handler._current is not None or len(handler.turn_records) > before,
                    10,
                    "the turn to end",
                )
            except TimeoutError:
                result.failures.append(f"no turn ended after {turn.text!r}")
                continue
            if turn.interrupt_after_s is not None:
                await _wait(
                    lambda: handler._current is not None and handler._current.first_audio_at is not None,
                    60,
                    "reply audio to start",
                )
                await asyncio.sleep(turn.interrupt_after_s)
                if handler._current is None:
                    result.failures.append(f"reply to {turn.text!r} ended before the interruption point")
            else:
                await driver.wait_idle(timeout=120)
        session = backends.store.peek(handler._session_id)
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
    try:
        settings = _settings(options, stub)
        if options.turn_detector == "both":
            detectors = ["silence", "smart"]
        elif options.turn_detector in (None, ""):
            detectors = [settings.turn_detector]
        elif options.turn_detector in ("silence", "smart"):
            detectors = [options.turn_detector]
        else:
            raise ConfigError(f"--turn-detector must be silence, smart or both, not {options.turn_detector!r}")
        fake = options.backends == "fake"
        tts = build_tts(settings)
        stt, llm = build_stt(settings), build_llm(settings)
        if not fake:
            # Load Whisper weights and warm ONNX sessions so the first scenario is not charged for it.
            await stt.transcribe(np.zeros(STT_SAMPLE_RATE // 2, dtype=np.float32))
            await tts.synthesize("Warming up.")
        caller = Caller(tts, fake)
        results: list[ScenarioResult] = []
        for detector in detectors:
            det_settings = _settings(options, stub, detector)
            backends = build_backends(det_settings, stt=stt, tts=tts, llm=llm)
            await backends.warm_fillers()
            shared = {"backends": backends, "caller": caller, "detector": detector}
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
        if options.judge:
            from .judge import judge_report

            await judge_report(results, settings)
    finally:
        if stub is not None:
            stub.stop()
    opts = {
        "llm": options.llm,
        "backends": options.backends,
        "stt_model": options.stt_model if options.backends == "real" else "fake",
        "repeats": options.repeats,
        "realtime": options.realtime,
        "budget_scale": options.budget_scale,
        "stub_delay_ms": options.stub_delay_ms,
        "turn_detectors": detectors,
    }
    return EvalReport(options=opts, scenarios=results)


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
    for det, d in summary["detectors"].items():
        rate = "-" if d["early_cutoff_rate"] is None else f"{d['early_cutoff_rate']:.0%}"
        lines.append(
            f"Turn detector {det}: early cutoffs {d['early_cutoffs']} of {d['scripted_turns']} turns ({rate}, "
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
    lines.append("")
    lines.append("PASS" if report.passed else "FAIL")
    return "\n".join(lines)


def parse_args(argv: list[str] | None) -> Options:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm", choices=["stub", "live"], default="stub")
    parser.add_argument("--backends", choices=["real", "fake"], default="real", help="real: Whisper, Kokoro, Silero")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="Whisper model id for --backends real")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="run only this scenario (repeatable)")
    parser.add_argument("--repeats", type=int, default=1, help="run every scenario this many times")
    parser.add_argument("--realtime", action="store_true", help="feed caller audio at real time (slower, closest to a browser)")
    parser.add_argument("--budget-scale", type=float, default=1.0, help="multiply every TTFA budget (slow CI machines)")
    parser.add_argument(
        "--turn-detector", choices=["silence", "smart", "both"], help="default: TURN_DETECTOR from the config"
    )
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
        out = RESULTS_DIR / f"{stamp}-{options.llm}-{options.backends}.json"
        out.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        print(f"results: {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
