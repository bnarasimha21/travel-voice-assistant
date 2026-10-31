"""Issue 5: Smart Turn end-of-turn detection, with the silence detector as fallback."""

from __future__ import annotations

import numpy as np
import pytest

from evals.run import Options, ScriptedTurn, load_scenario, load_scenarios, run_eval
from voice_agent.audio import frames, resample, silence
from voice_agent.backends import build_endpoint_model, turn_detector_factory
from voice_agent.config import ConfigError
from voice_agent.fakes import FakeCodec, FakeEndpointModel
from voice_agent.smart_turn import WINDOW_S, prepare_audio
from voice_agent.turn import SilenceTurnDetector, SmartTurnDetector, TurnEnded, TurnTaker
from voice_agent.vad import EnergyVAD

from .conftest import FAKE_ENV, fake_settings, require_real_models

CODEC = FakeCodec()
SR = 16000


def spoken(text: str) -> np.ndarray:
    return CODEC.encode(text, SR)


def split_line(first: str, pause_s: float, rest: str) -> np.ndarray:
    return np.concatenate([spoken(first), silence(pause_s, SR), spoken(rest)])


class CountingModel:
    name = "counting"

    def __init__(self, probability: float):
        self.probability = probability
        self.calls: list[int] = []

    def predict(self, audio16k: np.ndarray) -> float:
        self.calls.append(audio16k.size)
        return self.probability


# Detector logic ------------------------------------------------------------------


async def test_smart_detector_asks_once_per_pause_and_falls_back_to_max_silence():
    model = CountingModel(0.1)
    det = SmartTurnDetector(model, pause_ms=200, max_silence_ms=2000)
    audio = np.zeros(SR, dtype=np.float32)
    assert await det.should_end(audio, 100) is False
    assert model.calls == []  # too short a pause to ask
    assert await det.should_end(audio, 224) is False
    assert await det.should_end(audio, 256) is False
    assert await det.should_end(audio, 1500) is False
    assert len(model.calls) == 1 and det.last_probability == 0.1
    assert await det.should_end(audio, 2000) is True  # fallback, no new inference
    assert len(model.calls) == 1
    det.on_speech()  # the caller resumed: the next pause is asked again
    assert await det.should_end(audio, 224) is False
    assert len(model.calls) == 2


async def test_smart_detector_ends_at_the_first_pause_when_complete():
    det = SmartTurnDetector(CountingModel(0.97), pause_ms=200, threshold=0.5)
    assert await det.should_end(np.zeros(SR, dtype=np.float32), 224) is True
    assert det.predictions == 1


def test_smart_detector_rejects_pause_longer_than_fallback():
    with pytest.raises(ValueError):
        SmartTurnDetector(CountingModel(0.5), pause_ms=2000, max_silence_ms=1000)


def test_prepare_audio_keeps_last_window_padded_at_start():
    n = WINDOW_S * SR
    short = np.ones(SR, dtype=np.float32)
    out = prepare_audio(short)
    assert out.size == n and out[: n - SR].sum() == 0 and out[-SR:].sum() == SR
    long = np.arange(n + 500, dtype=np.float32)
    assert np.array_equal(prepare_audio(long), long[-n:])


# Configuration ---------------------------------------------------------------------


def test_turn_detector_config(stub, tmp_path):
    assert isinstance(turn_detector_factory(fake_settings(stub))(), SilenceTurnDetector)

    settings = fake_settings(stub, turn_detector="smart", smart_turn_backend="fake", smart_turn_pause_ms=150)
    make = turn_detector_factory(settings)
    a, b = make(), make()
    assert isinstance(a, SmartTurnDetector) and a is not b
    assert a.model is b.model  # one model shared by every connection
    assert isinstance(a.model, FakeEndpointModel) and a.pause_ms == 150 and a.max_silence_ms == 2000

    with pytest.raises(ConfigError, match="TURN_DETECTOR"):
        turn_detector_factory(fake_settings(stub, turn_detector="magic"))
    with pytest.raises(ConfigError, match="SMART_TURN_BACKEND"):
        build_endpoint_model(fake_settings(stub, smart_turn_backend="cloud"))
    with pytest.raises(ConfigError, match="SMART_TURN_MODEL_PATH is not set"):
        build_endpoint_model(fake_settings(stub, smart_turn_backend="onnx", smart_turn_model_path=""))
    with pytest.raises(ConfigError, match="huggingface.co/pipecat-ai/smart-turn-v3"):
        build_endpoint_model(
            fake_settings(stub, smart_turn_backend="onnx", smart_turn_model_path=str(tmp_path / "missing.onnx"))
        )


def test_env_example_selects_silence_by_default(stub):
    from voice_agent.config import REPO_ROOT, load_settings

    settings = load_settings(env={}, env_files=(REPO_ROOT / ".env.example",))
    assert settings.turn_detector == "silence"
    assert settings.smart_turn_model_path == "models/smart-turn-v3.2-cpu.onnx"


# Through the real handler (fake speech models) --------------------------------------


def smart_backends(make_backends, **kw):
    return make_backends(turn_detector="smart", smart_turn_backend="fake", **kw)


async def test_smart_turn_waits_through_a_mid_sentence_pause(make_backends, drivers):
    driver = await drivers(smart_backends(make_backends), "smart-split")
    await driver.say(split_line("i want to fly from los angeles to, um", 1.0, "san diego tomorrow."), trailing_silence_s=1.0)
    await driver.wait_until(lambda: driver.handler.turn_records, what="a turn")
    await driver.wait_idle()

    handler = driver.handler
    assert [r["outcome"] for r in handler.turn_records] == ["completed"]
    record = handler.turn_records[0]
    assert record["turn_detector"] == "smart"
    assert record["tools"] == ["search_flights"]
    # Ended at the first pause after the complete sentence, not after 700 ms.
    assert record["stages_ms"]["turn_end"] == pytest.approx(224, abs=40)
    [entry] = handler.session.turns
    assert "los angeles to, um" in entry.user and "san diego tomorrow" in entry.user


async def test_silence_detector_cuts_the_same_line_early(make_backends, drivers):
    driver = await drivers(make_backends(), "silence-split")
    await driver.say(split_line("i want to fly from los angeles to, um", 1.0, "san diego tomorrow."), trailing_silence_s=1.0)
    await driver.wait_until(lambda: len(driver.handler.turn_records) >= 2, what="two turn decisions")
    await driver.wait_idle()
    records = [r for r in driver.handler.turn_records if r["outcome"] != "empty"]
    assert len(records) == 2, records  # the pause ended a turn mid-sentence
    assert all(r["turn_detector"] == "silence" for r in records)


async def test_smart_turn_answers_a_complete_question_sooner(make_backends, drivers):
    driver = await drivers(smart_backends(make_backends), "smart-fast")
    await driver.say(spoken("what is the weather in san diego tomorrow?"))
    await driver.wait_idle()
    [record] = driver.handler.turn_records
    assert record["outcome"] == "completed" and record["tools"] == ["get_weather"]
    assert record["stages_ms"]["turn_end"] < 700  # the silence detector waits 700 ms
    assert driver.handler.turns.detector.predictions == 1


async def test_smart_turn_ends_an_unfinished_line_after_max_silence(make_backends, drivers):
    driver = await drivers(smart_backends(make_backends, smart_turn_max_silence_ms=1500), "smart-trail")
    await driver.say(spoken("tell me about hotels in, um"), trailing_silence_s=2.0)
    await driver.wait_until(lambda: driver.handler.turn_records, what="a turn")
    await driver.wait_idle()
    [record] = driver.handler.turn_records
    assert record["outcome"] == "completed"
    assert record["stages_ms"]["turn_end"] == pytest.approx(1500, abs=40)


# Eval reporting -------------------------------------------------------------------------


def test_pause_marks_in_scenarios(tmp_path):
    turn = ScriptedTurn(say="I want to fly to, um [pause 1.0] San Diego.")
    assert turn.text == "I want to fly to, um San Diego."
    assert turn.parts() == [("I want to fly to, um", 1.0), ("San Diego.", 0.0)]
    scenario = next(s for s in load_scenarios() if s.name == "mid_sentence_pause")
    assert scenario.allow_early_cutoff == ["silence"] and scenario.turns[0].pauses == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("turns:\n  - say: hi [pause 1] there\n    audio: x.wav\n")
    with pytest.raises(ValueError, match="pause"):
        load_scenario(bad)


def eval_options(**kw) -> Options:
    return Options(backends="fake", llm="stub", env=dict(FAKE_ENV), env_files=(), save=False, **kw)


async def test_eval_reports_early_cutoffs_per_detector():
    report = await run_eval(eval_options(scenarios=["mid_sentence_pause", "weather_tool"], turn_detector="both"))
    assert report.passed, [(s.label, s.failures) for s in report.scenarios]
    by = {(s.name, s.detector): s for s in report.scenarios}
    assert by[("mid_sentence_pause", "silence")].early_cutoffs == 1
    assert by[("mid_sentence_pause", "smart")].early_cutoffs == 0
    assert by[("weather_tool", "smart")].early_cutoffs == 0
    detectors = report.summary()["detectors"]
    assert detectors["silence"]["early_cutoffs"] == 1 and detectors["silence"]["scripted_turns"] == 2
    assert detectors["silence"]["early_cutoff_rate"] == 0.5
    assert detectors["smart"]["early_cutoff_rate"] == 0.0
    # The smart detector ends complete turns sooner.
    assert detectors["smart"]["ttfa_p50_ms"] < detectors["silence"]["ttfa_p50_ms"]


async def test_untolerated_audible_cutoff_fails_the_scenario(tmp_path):
    (tmp_path / "split.yaml").write_text(
        "name: split\n"
        "ttfa_budget_ms: 3500\n"
        "turns:\n"
        "  - say: I want to fly from Los Angeles to, um [pause 1.8] San Diego tomorrow.\n"
        "    expect_tools: [search_flights]\n"
    )
    silence_run = await run_eval(eval_options(scenarios_dir=tmp_path, turn_detector="silence"))
    [result] = silence_run.scenarios
    assert result.audible_cutoffs == 1 and result.status == "fail"
    assert any("audible early cutoff" in f for f in result.failures), result.failures
    smart_run = await run_eval(eval_options(scenarios_dir=tmp_path, turn_detector="smart"))
    assert smart_run.passed, smart_run.scenarios[0].failures


async def test_eval_rejects_unknown_detector():
    with pytest.raises(ConfigError, match="turn-detector"):
        await run_eval(eval_options(scenarios=["weather_tool"], turn_detector="psychic"))


# Real models ----------------------------------------------------------------------------


async def stream_turns(taker: TurnTaker, audio16k: np.ndarray) -> list[TurnEnded]:
    """Feed 20 ms frames, like the browser track, and collect turn decisions."""
    ended: list[TurnEnded] = []
    for frame in frames(audio16k, SR, 20):
        ended += [e for e in await taker.feed(frame) if isinstance(e, TurnEnded)]
    return ended


@pytest.mark.realmodels
async def test_real_smart_turn_on_kokoro_speech(tmp_path, record_property):
    """Kokoro clips through Silero VAD and the Smart Turn v3.2 ONNX model."""
    paths = require_real_models("kokoro", "voices", "smart_turn")
    from voice_agent.audio import read_wav, write_wav
    from voice_agent.smart_turn import SmartTurnOnnxModel
    from voice_agent.tts import KokoroTTS
    from voice_agent.vad import SileroVAD

    tts = KokoroTTS(str(paths["kokoro"]), str(paths["voices"]), voice="am_michael")
    model = SmartTurnOnnxModel(str(paths["smart_turn"]))

    async def clip(*parts) -> np.ndarray:
        pieces = [silence(0.3, SR)]
        for part in parts:
            if isinstance(part, str):
                pieces.append(resample(await tts.synthesize(part), tts.sample_rate, SR))
            else:
                pieces.append(silence(part, SR))
        pieces.append(silence(2.5, SR))
        wav = tmp_path / f"clip{len(list(tmp_path.iterdir()))}.wav"
        write_wav(wav, np.concatenate(pieces), SR)  # the test reads back a real WAV file
        rate, audio = read_wav(wav)
        return resample(audio, rate, SR)

    complete = await clip("What is the weather in San Diego tomorrow?")
    split = await clip("I want to fly to, um", 1.0, "San Diego tomorrow.")

    def smart():
        return TurnTaker(vad=SileroVAD(), detector=SmartTurnDetector(model))

    ended = await stream_turns(smart(), complete)
    assert len(ended) == 1
    delay_complete = ended[0].end_of_turn_delay_ms
    assert delay_complete < 700  # sooner than the silence detector
    ended = await stream_turns(smart(), split)
    assert len(ended) == 1, [e.end_of_turn_delay_ms for e in ended]  # no premature turn
    assert ended[0].speech_ms > 1500  # the turn holds both halves
    # The pause-only detector does cut this line in two.
    ended_silence = await stream_turns(TurnTaker(vad=SileroVAD(), detector=SilenceTurnDetector(700)), split)
    assert len(ended_silence) == 2
    record_property("smart_turn_end_of_turn_delay_ms", delay_complete)
    print(f"Smart Turn end-of-turn delay on a complete question: {delay_complete:.0f} ms")


@pytest.mark.realmodels
def test_onnx_model_returns_a_probability():
    paths = require_real_models("smart_turn")
    from voice_agent.smart_turn import SmartTurnOnnxModel

    p = SmartTurnOnnxModel(str(paths["smart_turn"])).predict(np.zeros(SR, dtype=np.float32))
    assert 0.0 <= p <= 1.0
