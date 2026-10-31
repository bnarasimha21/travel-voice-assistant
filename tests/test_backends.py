"""Issue 6: swappable STT backends, Deepgram Flux, and the speech-to-speech mode.

Local STT backends run the Issue 1 spoken-question flow through the real
`VoiceHandler`. Deepgram Flux, OpenAI Realtime and Gemini Live run against
local fake servers (`voice_agent.testing.fake_realtime`) that speak each
provider's WebSocket protocol, so the real adapters, the real handlers and the
real travel tools are exercised offline. The same flows against the hosted
services run only when their keys are set.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from dotenv import dotenv_values

from voice_agent.app import build_pipeline
from voice_agent.backends import build_backends, build_s2s_backends, build_stt
from voice_agent.config import REPO_ROOT, ConfigError, load_settings
from voice_agent.deepgram import DeepgramFluxSTT
from voice_agent.fakes import FakeCodec
from voice_agent.realtime import S2SBackends, S2SHandler
from voice_agent.realtime.gemini_live import GeminiLiveAdapter
from voice_agent.realtime.openai_realtime import OpenAIRealtimeAdapter
from voice_agent.testing.fake_realtime import FakeDeepgramFlux, FakeGeminiLive, FakeOpenAIRealtime
from voice_agent.testing.harness import HandlerDriver

from .conftest import fake_settings, require_real_models

CODEC = FakeCodec()
EXAMPLE = REPO_ROOT / ".env.example"
EXAMPLE_VALUES = {k: v for k, v in dotenv_values(EXAMPLE).items() if v}
QUESTION = "what is the weather in san diego tomorrow?"


def spoken(text: str, rate: int = 16000) -> np.ndarray:
    return CODEC.encode(text, rate)


def words(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def example_settings(**env: str):
    """Settings from `.env.example` alone plus the given variables (no `.env`, no process env)."""
    base = {"METRICS_LOG": "none", "VAD_BACKEND": "energy", "WEATHER_PROVIDER": "fixture"}
    return load_settings(env={**base, **env}, env_files=(EXAMPLE,))


def s2s_env(provider: str, fake) -> dict[str, str]:
    if provider == "openai":
        return {"PIPELINE": "s2s", "S2S_PROVIDER": "openai", "OPENAI_API_KEY": fake.api_key, "OPENAI_REALTIME_URL": fake.url}
    return {"PIPELINE": "s2s", "S2S_PROVIDER": "gemini", "GEMINI_API_KEY": fake.api_key, "GEMINI_LIVE_URL": fake.url}


FAKES = {"openai": FakeOpenAIRealtime, "gemini": FakeGeminiLive}


@pytest.fixture
async def s2s_drivers():
    created: list[HandlerDriver] = []

    async def make(backends: S2SBackends, session_id: str, pace_output: bool = False) -> HandlerDriver:
        driver = HandlerDriver(S2SHandler(backends, pace_output=pace_output).copy(), session_id=session_id)
        await driver.start()
        created.append(driver)
        return driver

    yield make
    for d in created:
        await d.close()
    await asyncio.sleep(0.05)  # let the adapters' close tasks finish before the fake servers stop


# Local STT backends: the Issue 1 flow -----------------------------------------------


def _hf_cached(repo: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, filename), str)
    except Exception:
        return False


LOCAL_STT = [
    pytest.param({"stt_backend": "fake"}, id="fake"),
    pytest.param(
        {"stt_backend": "transformers", "stt_model": "openai/whisper-tiny", "stt_device": "cpu"},
        id="transformers-whisper-tiny",
        marks=pytest.mark.realmodels,
    ),
    pytest.param(
        {"stt_backend": "faster-whisper", "stt_model": "Systran/faster-whisper-tiny", "stt_device": "cpu", "stt_compute_type": "int8"},
        id="faster-whisper-tiny",
        marks=pytest.mark.realmodels,
    ),
]


@pytest.mark.parametrize("stt", LOCAL_STT)
async def test_local_stt_backends_answer_the_spoken_question(stt, stub, drivers):
    if stt["stt_backend"] == "fake":
        audio, rate = spoken(QUESTION), 16000
        settings = fake_settings(stub)
    else:
        paths = require_real_models("kokoro", "voices")
        repo, filename = (stt["stt_model"], "config.json") if stt["stt_backend"] == "transformers" else (stt["stt_model"], "model.bin")
        if not _hf_cached(repo, filename):
            pytest.skip(f"{repo} is not in the Hugging Face cache")
        from voice_agent.tts import KokoroTTS

        kokoro = KokoroTTS(str(paths["kokoro"]), str(paths["voices"]), "af_sarah")
        audio, rate = await kokoro.synthesize("What is the weather in San Diego tomorrow?"), kokoro.sample_rate
        settings = fake_settings(stub, vad_backend="silero", **stt)
    backends = build_backends(settings)
    assert backends.stt.name == stt["stt_backend"]
    driver = await drivers(backends, f"local-{stt['stt_backend']}", input_rate=rate)
    await driver.say(audio)
    await driver.wait_until(lambda: driver.audio, timeout=60, what="reply audio")
    await driver.wait_idle(timeout=60)

    turns = backends.store.peek(f"local-{stt['stt_backend']}").turns
    assert len(turns) == 1
    heard = words(turns[0].user)
    assert "san diego" in heard and "weather" in heard
    if stt["stt_backend"] == "fake":
        assert turns[0].user == QUESTION
    # The weather tool ran against the fixture and the reply speaks its numbers.
    assert [m["role"] for m in turns[0].tool_messages] == ["assistant", "tool"]
    assert "high of" in turns[0].assistant and "San Diego" in turns[0].assistant
    assert {a.sample_rate for a in driver.audio} == {24000}


# Deepgram Flux (streaming STT with its own end of turn) ---------------------------------


async def test_deepgram_flux_decides_the_turn_and_supplies_the_transcript(stub, drivers):
    async with FakeDeepgramFlux() as flux:
        settings = fake_settings(
            stub,
            stt_backend="deepgram-flux",
            deepgram_url=flux.url,
            deepgram_model=EXAMPLE_VALUES["DEEPGRAM_MODEL"],
            deepgram_api_key=flux.api_key,
        )
        backends = build_backends(settings)
        assert isinstance(backends.stt, DeepgramFluxSTT) and backends.make_turn_source is not None
        driver = await drivers(backends, "flux")
        await driver.say(spoken(QUESTION))
        await driver.wait_until(lambda: driver.handler.turn_records, timeout=15, what="a finished turn")
        await driver.wait_idle()

        turns = backends.store.peek("flux").turns
        assert [t.user for t in turns] == [QUESTION]  # Flux's transcript; no local STT ran
        assert "high of" in turns[0].assistant
        record = driver.handler.turn_records[0]
        assert record["outcome"] == "completed" and record["turn_detector"] == "flux"
        # The end of turn delay is what the service took after the last voiced audio.
        assert 400 <= record["stages_ms"]["turn_end"] <= 1500
        conn = flux.connections[0]
        assert conn.headers["authorization"] == f"Token {flux.api_key}"
        assert conn.query == {
            "model": EXAMPLE_VALUES["DEEPGRAM_MODEL"],
            "encoding": "linear16",
            "sample_rate": "16000",
            "eot_threshold": "0.7",
            "eot_timeout_ms": "5000",
        }
        # Audio goes up in 80 ms chunks of 16-bit samples.
        sizes = {m["audio_bytes"] for m in conn.received if "audio_bytes" in m}
        assert sizes == {16000 * 80 // 1000 * 2}

        await driver.close()
        await asyncio.sleep(0.2)
        assert {"type": "CloseStream"} in conn.received


async def test_deepgram_flux_barge_in_cuts_the_reply(stub, drivers):
    async with FakeDeepgramFlux() as flux:
        settings = fake_settings(
            stub,
            stt_backend="deepgram-flux",
            deepgram_url=flux.url,
            deepgram_model=EXAMPLE_VALUES["DEEPGRAM_MODEL"],
            deepgram_api_key=flux.api_key,
        )
        backends = build_backends(settings)
        driver = await drivers(backends, "flux-barge", pace_output=True)
        await driver.say(spoken("tell me something long about the beaches near los angeles please."), trailing_silence_s=0.8)
        await driver.wait_until(lambda: driver.audio, timeout=15, what="reply audio")
        await asyncio.sleep(0.4)
        await driver.say(spoken("stop. what about hotels?"), realtime=True)
        await driver.wait_until(lambda: len(backends.store.peek("flux-barge").turns) == 2, timeout=20, what="second turn")
        await driver.wait_idle()
        first, second = backends.store.peek("flux-barge").turns
        assert first.interrupted
        assert len(first.assistant) < len("Happy to help with that. You asked: tell me something long about the beaches near los angeles please.")
        assert second.user == "stop. what about hotels?"


def test_deepgram_flux_needs_a_key():
    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        build_stt(example_settings(STT_BACKEND="deepgram-flux"))


async def test_deepgram_flux_bad_key_is_logged_not_fatal(stub, drivers, caplog):
    async with FakeDeepgramFlux() as flux:
        settings = fake_settings(
            stub, stt_backend="deepgram-flux", deepgram_url=flux.url, deepgram_model="m", deepgram_api_key="wrong"
        )
        driver = await drivers(build_backends(settings), "flux-badkey")
        await driver.say(spoken("hello there."))
        assert flux.rejected >= 1 and not flux.connections
        assert "Deepgram Flux connection failed" in caplog.text
        assert not driver.handler.turn_records


# Speech to speech -------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_s2s_holds_a_conversation_and_calls_get_weather(provider, s2s_drivers):
    async with FAKES[provider](reply_speedup=0) as fake:
        backends = build_pipeline(example_settings(**s2s_env(provider, fake)))
        assert isinstance(backends, S2SBackends) and backends.provider == provider
        driver = await s2s_drivers(backends, f"s2s-{provider}")
        await driver.say(spoken(QUESTION))
        await driver.wait_until(lambda: driver.handler.turn_records, timeout=15, what="first turn")
        await driver.say(spoken("thanks, and what about flights from lax to san francisco tomorrow?"))
        await driver.wait_until(lambda: len(driver.handler.turn_records) == 2, timeout=15, what="second turn")
        await driver.wait_idle()

        session = backends.store.peek(f"s2s-{provider}")
        assert [t.user for t in session.turns] == [QUESTION, "thanks, and what about flights from lax to san francisco tomorrow?"]
        weather, flights = session.turns
        # The provider asked for get_weather, it ran here against the fixture, and the result went back.
        assert weather.tool_messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        tool_out = json.loads(weather.tool_messages[1]["content"])
        assert tool_out["location"].startswith("San Diego")
        assert f"high of {round(tool_out['high_f'])}" in weather.assistant
        assert flights.tool_messages[0]["tool_calls"][0]["function"]["name"] == "search_flights"
        assert "flight" in flights.assistant
        # Replies arrive at 24 kHz and decode back to what the model said.
        assert {a.sample_rate for a in driver.audio} == {24000}
        assert "san diego" in words(CODEC.decode(driver.audio_samples(), 24000))
        # Latency records use the same stages as the cascade.
        for record in driver.handler.turn_records:
            assert record["outcome"] == "completed" and record["turn_detector"] == provider
            assert record["ttfa_ms"] is not None and record["ttfa_ms"] > 0
            assert record["stages_ms"]["tool_start"] is not None and record["stages_ms"]["tool_end"] is not None
            # OpenAI reports when it decided the turn was over; Gemini does not.
            assert (record["stages_ms"]["turn_end"] is None) == (provider == "gemini")
        conn = fake.connections[0]
        if provider == "openai":
            assert conn.headers["authorization"] == f"Bearer {fake.api_key}"
            assert conn.query["model"] == EXAMPLE_VALUES["OPENAI_REALTIME_MODEL"]
            update = conn.received[0]
            assert update["type"] == "session.update"
            session_cfg = update["session"]
            assert session_cfg["type"] == "realtime"
            assert session_cfg["audio"]["output"]["voice"] == EXAMPLE_VALUES["OPENAI_REALTIME_VOICE"]
            assert session_cfg["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
            assert session_cfg["audio"]["input"]["transcription"]["model"] == EXAMPLE_VALUES["OPENAI_REALTIME_TRANSCRIBE_MODEL"]
            assert {t["name"] for t in session_cfg["tools"]} == {"get_weather", "search_flights", "search_hotels"}
            outputs = [m for m in conn.received if m.get("type") == "conversation.item.create"]
            assert len(outputs) == 2 and all(m["item"]["type"] == "function_call_output" for m in outputs)
        else:
            assert conn.query["key"] == fake.api_key
            setup = conn.received[0]["setup"]
            assert setup["model"] == f"models/{EXAMPLE_VALUES['GEMINI_LIVE_MODEL']}"
            assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == EXAMPLE_VALUES["GEMINI_LIVE_VOICE"]
            names = {d["name"] for d in setup["tools"][0]["functionDeclarations"]}
            assert names == {"get_weather", "search_flights", "search_hotels"}
            chunks = [m["realtimeInput"] for m in conn.received if "realtimeInput" in m]
            assert chunks and {c["mimeType"] for c in chunks} == {"audio/pcm;rate=16000"}
            responses = [m for m in conn.received if "toolResponse" in m]
            assert len(responses) == 2


@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_s2s_barge_in_keeps_only_what_was_heard(provider, s2s_drivers):
    long_question = "tell me something long about the beaches near los angeles please."
    async with FAKES[provider](reply_speedup=4) as fake:
        backends = build_pipeline(example_settings(**s2s_env(provider, fake)))
        driver = await s2s_drivers(backends, f"barge-{provider}", pace_output=True)
        await driver.say(spoken(long_question), trailing_silence_s=0.8)
        await driver.wait_until(lambda: driver.audio, timeout=15, what="reply audio")
        await asyncio.sleep(0.5)
        await driver.say(spoken("stop. what about hotels in san diego?"), realtime=True)
        await driver.wait_until(lambda: len(backends.store.peek(f"barge-{provider}").turns) == 2, timeout=20, what="second turn")
        await driver.wait_idle()

        first, second = backends.store.peek(f"barge-{provider}").turns
        full = f"Happy to help with that. You asked: {long_question}"
        assert first.interrupted and first.user == long_question
        assert 0 < len(first.assistant.split()) < len(full.split())
        assert full.startswith(first.assistant)
        assert second.user == "stop. what about hotels in san diego?"
        assert second.tool_messages[0]["tool_calls"][0]["function"]["name"] == "search_hotels"
        outcomes = [r["outcome"] for r in driver.handler.turn_records]
        assert outcomes[0] == "interrupted" and outcomes[-1] == "completed"
        if provider == "openai":
            # The provider is told to forget the audio the caller never heard.
            truncs = [m for m in fake.connections[0].received if m.get("type") == "conversation.item.truncate"]
            assert len(truncs) == 1 and truncs[0]["audio_end_ms"] > 0
            assistant = [m for m in fake.connections[0].conversation if m.get("role") == "assistant" and m.get("content")]
            assert len(assistant[0]["content"]) < len(full)


@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_s2s_barge_in_after_the_whole_reply_arrived(provider, s2s_drivers):
    """Providers send audio faster than it plays; once the reply is fully sent the
    provider no longer reports the caller talking over it (Gemini sends nothing),
    so the handler's own VAD must stop the playback."""
    long_question = "tell me something long about the beaches near los angeles please."
    async with FAKES[provider](reply_speedup=50) as fake:
        backends = build_pipeline(example_settings(**s2s_env(provider, fake)))
        sid = f"late-barge-{provider}"
        driver = await s2s_drivers(backends, sid, pace_output=True)
        await driver.say(spoken(long_question), trailing_silence_s=0.8)
        await driver.wait_until(
            lambda: driver.handler._current is not None and driver.handler._current.generated,
            timeout=15,
            what="the whole reply to arrive",
        )
        assert driver.handler._out.qsize() > 50  # seconds of reply still queued to play
        await driver.say(spoken("stop. what about hotels in san diego?"), realtime=True)
        await driver.wait_until(lambda: len(backends.store.peek(sid).turns) == 2, timeout=20, what="second turn")
        await driver.wait_idle()

        first, second = backends.store.peek(sid).turns
        full = f"Happy to help with that. You asked: {long_question}"
        assert first.interrupted and first.user == long_question
        assert 0 < len(first.assistant.split()) < len(full.split())
        assert second.user == "stop. what about hotels in san diego?"
        assert [r["outcome"] for r in driver.handler.turn_records] == ["interrupted", "completed"]
        assert len(driver.handler.barge_ins) == 1


async def test_s2s_reconnects_with_the_history_after_a_drop(s2s_drivers):
    async with FakeOpenAIRealtime(reply_speedup=0) as fake:
        backends = build_pipeline(example_settings(**s2s_env("openai", fake)))
        driver = await s2s_drivers(backends, "drop")
        await driver.say(spoken("my name is ada."))
        await driver.wait_until(lambda: driver.handler.turn_records, timeout=15, what="first turn")
        await driver.wait_idle()
        await fake.drop_all()
        await driver.wait_until(lambda: driver.handler.errors, timeout=5, what="the drop to be noticed")
        assert "Lost the connection" in driver.handler.errors[0]
        await asyncio.sleep(2.1)  # reconnect backoff
        await driver.say(spoken("what is my name?"))
        await driver.wait_until(lambda: len(driver.handler.turn_records) == 2, timeout=15, what="second turn")
        assert len(fake.connections) == 2
        instructions = fake.connections[1].received[0]["session"]["instructions"]
        assert "Caller: my name is ada." in instructions
        assert [t.user for t in backends.store.peek("drop").turns] == ["my name is ada.", "what is my name?"]


async def test_s2s_wrong_key_shows_a_notice_and_does_not_crash(s2s_drivers):
    async with FakeGeminiLive() as fake:
        env = s2s_env("gemini", fake) | {"GEMINI_API_KEY": "wrong"}
        driver = await s2s_drivers(build_pipeline(example_settings(**env)), "badkey")
        await driver.say(spoken("hello."))
        assert fake.rejected >= 1
        assert driver.handler.errors and "Could not reach the gemini model" in driver.handler.errors[0]
        assert driver.ui and "Could not reach" in driver.ui[-1].args[0][-1]["content"]


async def test_clearing_one_s2s_session_leaves_the_other(s2s_drivers):
    async with FakeOpenAIRealtime(reply_speedup=0) as fake:
        backends = build_pipeline(example_settings(**s2s_env("openai", fake)))
        a = await s2s_drivers(backends, "tab-a")
        b = await s2s_drivers(backends, "tab-b")
        await asyncio.gather(a.say(spoken("my name is ada.")), b.say(spoken("my name is bob.")))
        await a.wait_until(lambda: a.handler.turn_records and b.handler.turn_records, timeout=15, what="both turns")
        await asyncio.gather(a.wait_idle(), b.wait_idle())
        backends.store.reset("tab-a")
        await asyncio.sleep(0.1)
        assert backends.store.peek("tab-a").turns == []
        assert [t.user for t in backends.store.peek("tab-b").turns] == ["my name is bob."]
        assert a.handler.adapter is None and b.handler.adapter is not None


# Configuration ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({"PIPELINE": "s2s", "S2S_PROVIDER": "openai"}, "OPENAI_API_KEY"),
        ({"PIPELINE": "s2s", "S2S_PROVIDER": "gemini"}, "GEMINI_API_KEY"),
        ({"PIPELINE": "s2s", "S2S_PROVIDER": "nope"}, "Unknown S2S_PROVIDER"),
        ({"PIPELINE": "s2s", "S2S_PROVIDER": "openai", "OPENAI_API_KEY": "k", "OPENAI_REALTIME_TURN_DETECTION": "semantic"}, "OPENAI_REALTIME_TURN_DETECTION"),
        ({"PIPELINE": "both"}, "Unknown PIPELINE"),
        ({"STT_BACKEND": "nope"}, "Unknown STT_BACKEND"),
    ],
)
def test_bad_backend_configuration_is_a_clear_error(env, missing):
    with pytest.raises(ConfigError, match=missing):
        build_pipeline(example_settings(**env))


def test_env_example_plus_a_key_builds_each_hosted_mode():
    openai = build_s2s_backends(example_settings(PIPELINE="s2s", S2S_PROVIDER="openai", OPENAI_API_KEY="k"))
    adapter = openai.make_adapter()
    assert isinstance(adapter, OpenAIRealtimeAdapter)
    assert (adapter.url, adapter.model, adapter.voice) == (
        EXAMPLE_VALUES["OPENAI_REALTIME_URL"],
        EXAMPLE_VALUES["OPENAI_REALTIME_MODEL"],
        EXAMPLE_VALUES["OPENAI_REALTIME_VOICE"],
    )
    assert adapter.turn_detection == EXAMPLE_VALUES["OPENAI_REALTIME_TURN_DETECTION"]
    gemini = build_s2s_backends(example_settings(PIPELINE="s2s", S2S_PROVIDER="gemini", GEMINI_API_KEY="k"))
    adapter = gemini.make_adapter()
    assert isinstance(adapter, GeminiLiveAdapter) and adapter.url == EXAMPLE_VALUES["GEMINI_LIVE_URL"]
    flux = build_stt(example_settings(STT_BACKEND="deepgram-flux", DEEPGRAM_API_KEY="k"))
    assert (flux.url, flux.model) == (EXAMPLE_VALUES["DEEPGRAM_URL"], EXAMPLE_VALUES["DEEPGRAM_MODEL"])


def test_env_example_plus_a_key_builds_the_cascade_config():
    settings = example_settings(LLM_API_KEY="k")
    settings.require("llm_base_url", "llm_model", "stt_model", "tts_voice")
    assert settings.pipeline == "cascade"


def _code_strings(path: Path) -> list[tuple[int, str]]:
    """String literals in a module, without docstrings."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def test_no_endpoint_model_or_voice_is_hardcoded_in_python():
    pattern = re.compile(r"^(wss?|https?)://|gpt-realtime|gpt-live|gemini-\d|flux-general|whisper-(tiny|base|small|large)|^af_|^Kore$|^marin$")
    hits = []
    for path in (REPO_ROOT / "voice_agent").rglob("*.py"):
        if "testing" in path.parts:
            continue
        for line, value in _code_strings(path):
            if pattern.search(value):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{line}: {value!r}")
    assert hits == []


# Hosted services, only with keys ----------------------------------------------------------


def _kokoro():
    paths = require_real_models("kokoro", "voices")
    from voice_agent.tts import KokoroTTS

    return KokoroTTS(str(paths["kokoro"]), str(paths["voices"]), "af_sarah")


@pytest.mark.realmodels
@pytest.mark.skipif(not os.environ.get("DEEPGRAM_API_KEY"), reason="DEEPGRAM_API_KEY is not set")
async def test_real_deepgram_flux(stub, drivers):
    kokoro = _kokoro()
    audio = await kokoro.synthesize("What is the weather in San Diego tomorrow?")
    settings = fake_settings(
        stub,
        stt_backend="deepgram-flux",
        deepgram_url=EXAMPLE_VALUES["DEEPGRAM_URL"],
        deepgram_model=EXAMPLE_VALUES["DEEPGRAM_MODEL"],
        deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
        vad_backend="silero",
    )
    backends = build_backends(settings)
    driver = await drivers(backends, "real-flux", input_rate=kokoro.sample_rate)
    await driver.say(audio, realtime=True)
    await driver.wait_until(lambda: driver.handler.turn_records, timeout=30, what="a finished turn")
    heard = words(backends.store.peek("real-flux").turns[0].user)
    assert "san diego" in heard and "weather" in heard


@pytest.mark.realmodels
@pytest.mark.parametrize(
    "provider",
    [
        pytest.param("openai", marks=pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")),
        pytest.param("gemini", marks=pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY is not set")),
    ],
)
async def test_real_s2s_calls_get_weather(provider, s2s_drivers):
    kokoro = _kokoro()
    audio = await kokoro.synthesize("What is the weather in San Diego tomorrow?")
    key = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    settings = example_settings(PIPELINE="s2s", S2S_PROVIDER=provider, **{key: os.environ[key]})
    backends = build_s2s_backends(settings)
    handler = S2SHandler(backends)
    driver = HandlerDriver(handler, session_id=f"real-{provider}", input_rate=kokoro.sample_rate)
    await driver.start()
    try:
        await driver.say(audio, trailing_silence_s=2.0, realtime=True)
        await driver.wait_until(lambda: handler.turn_records, timeout=45, what="a finished turn")
        turn = backends.store.peek(f"real-{provider}").turns[0]
        assert turn.tool_messages and turn.tool_messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert driver.audio
    finally:
        await driver.close()


# The comparison matrix (python -m evals.run --matrix) --------------------------


def matrix_options(**kw):
    from evals.run import Options

    from .conftest import FAKE_ENV

    return Options(backends="fake", llm="stub", matrix=True, env=dict(FAKE_ENV), env_files=(), save=False, **kw)


async def test_matrix_runs_the_same_scenarios_on_every_backend_offline():
    from evals.run import format_matrix, run_eval

    report = await run_eval(matrix_options(scenarios=["barge_in", "mid_sentence_pause"]))
    assert report.passed, [(s.label, s.status, s.failures) for s in report.scenarios if s.status != "pass"]
    configs = report.summary()["configs"]
    assert list(configs) == ["fake/silence", "fake/smart-turn", "fake-deepgram-flux", "fake-s2s/openai", "fake-s2s/gemini"]
    assert {name: d["detector"] for name, d in configs.items()} == {
        "fake/silence": "silence",
        "fake/smart-turn": "smart",
        "fake-deepgram-flux": "flux",
        "fake-s2s/openai": "openai",
        "fake-s2s/gemini": "gemini",
    }
    for d in configs.values():
        assert d["scenarios"] == 2 and d["pass_rate"] == 1.0 and d["ttfa_p50_ms"] is not None
    # Every backend stopped talking when the caller cut in, and kept what was heard.
    for s in report.scenarios:
        if s.name == "barge_in":
            assert [t.outcome for t in s.turns] == ["interrupted", "completed"], s.label
    # Semantic detectors wait through "fly to, um ..."; pause-based ones cut in.
    pause = {s.config: s.early_cutoffs for s in report.scenarios if s.name == "mid_sentence_pause"}
    assert pause == {"fake/silence": 1, "fake/smart-turn": 0, "fake-deepgram-flux": 0, "fake-s2s/openai": 0, "fake-s2s/gemini": 1}
    # Hosted rows are timed with caller audio at real time, so their TTFA includes
    # the provider's end-of-turn wait (500 ms in the fakes), like the cascade's.
    assert configs["fake-s2s/openai"]["ttfa_p50_ms"] > 400
    table = format_matrix(report)
    assert table.splitlines()[0].split("|")[1].strip() == "config"
    assert all(name in table for name in configs)


async def test_matrix_skips_rows_without_keys_and_reports_bad_ones(tmp_path):
    from evals.run import format_matrix, run_eval

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "fake:\n"
        "  base: {STT_BACKEND: fake, TTS_BACKEND: fake, VAD_BACKEND: energy}\n"
        "  matrix:\n"
        "    - {name: local, env: {TURN_DETECTOR: silence}}\n"
        "    - {name: hosted-flux, env: {STT_BACKEND: deepgram-flux}, requires: [DEEPGRAM_API_KEY]}\n"
        "    - {name: broken, env: {STT_BACKEND: nonsense}}\n"
    )
    report = await run_eval(matrix_options(scenarios=["weather_tool"], matrix_file=matrix))
    assert report.passed
    assert [s.config for s in report.scenarios] == ["local"]
    assert report.skipped_configs["hosted-flux"] == "needs DEEPGRAM_API_KEY"
    assert report.skipped_configs["broken"].startswith("configuration error") and "nonsense" in report.skipped_configs["broken"]
    assert report.summary()["skipped_configs"] == report.skipped_configs
    rows = {line.split("|")[1].strip(): line for line in format_matrix(report).splitlines()}
    assert "skipped" in rows["hosted-flux"] and "needs DEEPGRAM_API_KEY" in rows["hosted-flux"]


async def test_matrix_options_are_validated(tmp_path):
    from evals.run import run_eval

    with pytest.raises(ConfigError, match="turn-detector"):
        await run_eval(matrix_options(turn_detector="both"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("fake:\n  matrix:\n    - {name: a, fake_server: nope}\n")
    with pytest.raises(ConfigError, match="fake_server"):
        await run_eval(matrix_options(matrix_file=bad))
    dup = tmp_path / "dup.yaml"
    dup.write_text("fake:\n  matrix:\n    - {name: a}\n    - {name: a}\n")
    with pytest.raises(ConfigError, match="unique"):
        await run_eval(matrix_options(matrix_file=dup))


def test_matrix_cli_prints_the_comparison_table(monkeypatch, capsys):
    from evals.run import main

    from .conftest import FAKE_ENV

    for key, value in FAKE_ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert main(["--backends", "fake", "--matrix", "--scenario", "weather_tool", "--no-save"]) == 0
    out = capsys.readouterr().out
    table = [line for line in out.splitlines() if line.startswith("| ")]
    assert table[0].split("|")[1].strip() == "config"
    assert {line.split("|")[1].strip() for line in table[2:]} == {
        "fake/silence", "fake/smart-turn", "fake-deepgram-flux", "fake-s2s/openai", "fake-s2s/gemini"
    }
    assert all(line.split("|")[3].strip() == "100%" for line in table[2:])
    assert main(["--backends", "fake", "--matrix", "--turn-detector", "both", "--no-save"]) == 2


def test_real_matrix_lists_local_and_hosted_rows():
    from evals.run import MATRIX_FILE, load_matrix

    base, configs = load_matrix(MATRIX_FILE, "real")
    assert base["STT_MODEL"] == "openai/whisper-tiny"
    by = {c.name: c for c in configs}
    assert {"whisper-tiny/silence", "whisper-tiny/smart-turn", "faster-whisper-tiny/silence"} <= set(by)
    assert by["deepgram-flux"].requires == ["DEEPGRAM_API_KEY"] and by["deepgram-flux"].realtime
    assert by["s2s/openai"].requires == ["OPENAI_API_KEY"] and by["s2s/gemini"].requires == ["GEMINI_API_KEY"]
    assert not any(c.fake_server for c in configs)
