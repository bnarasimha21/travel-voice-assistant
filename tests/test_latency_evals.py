"""Issue 4: per-turn latency records and the voice eval suite."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evals.judge import judge_report, parse_verdict
from evals.run import (
    SCENARIOS_DIR,
    Options,
    load_scenario,
    load_scenarios,
    main,
    missing_facts,
    run_eval,
    word_error_rate,
)
from voice_agent.config import ConfigError
from voice_agent.fakes import FakeCodec
from voice_agent.llm import OpenAICompatibleLLM
from voice_agent.metrics import STAGES, TurnLog, TurnTimer, format_breakdown, percentile, ttfa_summary
from voice_agent.testing.stub_llm import Plan, StubLLMServer

from .conftest import FAKE_ENV, require_real_models, whisper_tiny_available

CODEC = FakeCodec()
ORDERED = ["speech_end", "turn_end", "stt_done", "llm_first_token", "tts_first_chunk", "first_audio_sent", "reply_end"]


def spoken(text: str) -> np.ndarray:
    return CODEC.encode(text, 16000)


def fake_options(**kw) -> Options:
    return Options(backends="fake", llm="stub", env=dict(FAKE_ENV), env_files=(), save=False, **kw)


# Turn records ------------------------------------------------------------------


async def test_every_turn_writes_a_complete_json_line(make_backends, drivers, tmp_path):
    log_path = tmp_path / "logs" / "turns.jsonl"
    backends = make_backends(metrics_log=str(log_path))
    driver = await drivers(backends, "timed")

    await driver.say(spoken("hello there, can you help me plan a trip?"))
    await driver.wait_idle()
    await driver.say(spoken("what is the weather in san diego tomorrow?"))
    await driver.wait_idle()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    plain, weather = (json.loads(line) for line in lines)
    for record in (plain, weather):
        assert set(record["stages_ms"]) == set(STAGES)
        assert record["outcome"] == "completed" and record["interrupted"] is False
        assert record["session"] == "timed"
        stages = record["stages_ms"]
        values = [stages[s] for s in ORDERED]
        assert None not in values
        assert values == sorted(values), stages
        assert record["ttfa_ms"] == stages["first_audio_sent"]
        # Turn decided after the configured pause (700 ms of audio).
        assert stages["turn_end"] == pytest.approx(700, abs=40)
    assert plain["tools"] == [] and plain["stages_ms"]["tool_start"] is None and plain["stages_ms"]["tool_end"] is None
    assert weather["tools"] == ["get_weather"]
    w = weather["stages_ms"]
    assert w["stt_done"] <= w["tool_start"] <= w["tool_end"] <= w["llm_first_token"] <= w["first_audio_sent"]
    assert [plain["turn"], weather["turn"]] == [0, 1]


async def test_ui_shows_last_turn_breakdown(make_backends, drivers):
    backends = make_backends()
    driver = await drivers(backends, "ui")
    await driver.say(spoken("hello there, can you help me plan a trip?"))
    await driver.wait_idle()
    breakdown = driver.ui[-1].args[1]
    assert "First audio sent (TTFA)" in breakdown
    assert "Outcome: completed" in breakdown
    assert "TTFA p50" in breakdown and "over 1 turns" in breakdown


async def test_interrupted_turn_is_recorded(make_backends, drivers, stub, tmp_path):
    log_path = tmp_path / "turns.jsonl"
    stub.brain = lambda body: Plan(text="Here is a long answer about beaches. " * 6)
    backends = make_backends(metrics_log=str(log_path))
    driver = await drivers(backends, "barge", pace_output=True)
    await driver.say(spoken("tell me about beaches near los angeles please"))
    await driver.wait_until(lambda: driver.audio, what="reply audio")
    import asyncio

    await asyncio.sleep(0.5)
    await driver.say(spoken("stop, what about hotels instead?"))
    await driver.wait_idle(timeout=30)
    records = TurnLog(log_path).read()
    assert [r["outcome"] for r in records] == ["interrupted", "completed"]
    assert records[0]["interrupted"] is True
    assert records[0]["stages_ms"]["first_audio_sent"] is not None
    assert set(records[0]["stages_ms"]) == set(STAGES)


def test_timer_marks_first_occurrence_and_finishes_once():
    timer = TurnTimer("s", 0, end_of_turn_delay_ms=700.0, decided_at=10.0)
    timer.mark("stt_done", 10.2)
    timer.mark("stt_done", 10.9)  # ignored
    timer.mark("first_audio_sent", 10.5)
    record = timer.finish("completed")
    assert record["stages_ms"]["stt_done"] == pytest.approx(900.0)
    assert record["ttfa_ms"] == pytest.approx(1200.0)
    assert timer.finish("cancelled") is record
    with pytest.raises(ValueError):
        timer.mark("nonsense")


def test_percentiles_and_summary():
    assert percentile([], 95) is None
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile(range(1, 101), 95) == 95
    records = [{"ttfa_ms": v} for v in (100, 200, 300)] + [{"ttfa_ms": None}]
    assert ttfa_summary(records) == {"turns": 3, "p50_ms": 200, "p95_ms": 300}
    assert "after your first turn" in format_breakdown(None)


def test_metrics_log_setting(stub):
    from .conftest import fake_settings

    assert fake_settings(stub).metrics_log_path is None
    path = fake_settings(stub, metrics_log="logs/turns.jsonl").metrics_log_path
    assert path is not None and path.is_absolute() and path.parts[-2:] == ("logs", "turns.jsonl")


# Eval suite ------------------------------------------------------------------


def test_scenarios_cover_required_cases():
    scenarios = {s.name: s for s in load_scenarios()}
    assert {"plain_question", "weather_tool", "flight_tool", "hotel_tool", "barge_in", "noisy_audio"} <= set(scenarios)
    assert scenarios["noisy_audio"].noise_snr_db is not None
    assert any(t.interrupt_after_s for t in scenarios["barge_in"].turns)
    assert len({s.voice for s in scenarios.values()}) >= 4
    assert len({s.speed for s in scenarios.values()}) >= 2
    assert scenarios["plain_question"].turns[0].expect_tools == []


def test_scenario_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("turns:\n  - say: hi\n    interrupt_after_s: 1\n")
    with pytest.raises(ValueError, match="last turn"):
        load_scenario(bad)
    bad.write_text("turns:\n  - expect_tools: []\n")
    with pytest.raises(ValueError, match="say"):
        load_scenario(bad)


def test_wer_and_facts():
    assert word_error_rate("What is the weather in San Diego?", "what's the weather in san diego") == pytest.approx(2 / 7)
    assert word_error_rate("two nights", "2 nights") == 0
    assert word_error_rate("a b c", "") == 1
    assert missing_facts("Southwest at 98 USD", [["southwest"], ["98"], ["delta", "united"]]) == ["delta / united"]


async def test_stub_eval_passes_with_fake_backends():
    report = await run_eval(fake_options())
    assert report.passed, [s.failures for s in report.scenarios]
    assert len(report.scenarios) == len(list(SCENARIOS_DIR.glob("*.yaml")))
    barge = next(s for s in report.scenarios if s.name == "barge_in")
    assert [t.outcome for t in barge.turns] == ["interrupted", "completed"]
    flights = next(s for s in report.scenarios if s.name == "flight_tool")
    assert flights.turns[0].tools == ["search_flights"]
    assert report.summary()["ttfa_p95_ms"] is not None


async def test_injected_stub_delay_fails_the_latency_budget():
    # weather_tool budget is 3500 ms; scaled by 0.5 that is 1750 ms. Without
    # the delay the fake pipeline answers in about 0.7 s, so only the
    # injected delay can push it over.
    ok = await run_eval(fake_options(scenarios=["weather_tool"], budget_scale=0.5))
    assert ok.passed, ok.scenarios[0].failures
    slow = await run_eval(fake_options(scenarios=["weather_tool"], budget_scale=0.5, stub_delay_ms=1500))
    assert not slow.passed
    failures = slow.scenarios[0].failures
    assert any("TTFA p95" in f and "exceeds budget 1750 ms" in f for f in failures), failures
    # Only latency failed: the answer itself was still right.
    assert all("TTFA" in f for f in failures)


def test_cli_exits_non_zero_on_failure_and_zero_on_pass(monkeypatch, capsys):
    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    base = ["--backends", "fake", "--scenario", "weather_tool", "--no-save", "--budget-scale", "0.5"]
    assert main(base) == 0
    assert main(base + ["--stub-delay-ms", "1500"]) == 1
    out = capsys.readouterr().out
    assert "weather_tool" in out and "FAIL" in out
    assert main(base + ["--scenario", "nope"]) == 2


async def test_judge_requires_live_mode():
    with pytest.raises(ConfigError, match="--judge"):
        await run_eval(fake_options(judge=True))


async def test_judge_grades_completed_turns_offline():
    from evals.run import ScenarioResult, TurnResult

    def brain(body):
        prompt = body["messages"][0]["content"]
        verdict = {"helpful": 2 if "rambling" in prompt else 5, "length_ok": True, "reason": "ok"}
        return Plan(text="Verdict: " + json.dumps(verdict))

    def turn(answer, outcome="completed"):
        return TurnResult("q", "q", answer, 0.0, [], outcome, 500.0, {}, [])

    good = ScenarioResult("good", "pass", turns=[turn("Sunny and 70."), turn("cut", "interrupted")])
    bad = ScenarioResult("bad", "pass", turns=[turn("A rambling answer.")])
    with StubLLMServer(brain=brain) as stub:
        llm = OpenAICompatibleLLM(base_url=stub.base_url, api_key="k", model="judge-model")
        await judge_report([good, bad], settings=None, llm=llm)
        assert len(stub.requests) == 2  # the interrupted turn is not graded
    assert good.status == "pass" and good.turns[0].judge["helpful"] == 5 and good.turns[1].judge is None
    assert bad.status == "fail" and "judge" in bad.failures[0]
    with pytest.raises(ValueError):
        parse_verdict("no json here")
    with pytest.raises(ValueError):
        parse_verdict('{"helpful": 9, "length_ok": true}')


@pytest.mark.realmodels
async def test_stub_eval_with_real_speech_models():
    """The CI eval path: Kokoro caller voices, Whisper tiny, Silero VAD, Kokoro replies."""
    require_real_models("kokoro", "voices")
    if not whisper_tiny_available():
        pytest.skip("openai/whisper-tiny is not in the Hugging Face cache")
    report = await run_eval(
        Options(llm="stub", backends="real", scenarios=["weather_tool", "barge_in"], budget_scale=3, save=False)
    )
    assert report.passed, [s.failures for s in report.scenarios]
    weather = next(s for s in report.scenarios if s.name == "weather_tool")
    assert weather.turns[0].wer <= 0.25 and weather.turns[0].tools == ["get_weather"]
