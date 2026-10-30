"""Issue 1: per-visitor sessions, browser playback, configurable endpoint, LLM errors.

These drive the real `VoiceHandler` the way FastRTC does (frames into
`receive`, audio out of `emit`) with fake speech models and the local stub
LLM reached through the real OpenAI client.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import numpy as np
import pytest

from voice_agent.app import build_ui, clear_session
from voice_agent.audio import read_wav, write_wav
from voice_agent.config import REPO_ROOT, ConfigError, load_settings
from voice_agent.fakes import FakeCodec, FakeTTS
from voice_agent.prompts import LLM_FALLBACK_LINE
from voice_agent.testing.stub_llm import Plan, travel_brain

from .conftest import fake_settings

CODEC = FakeCodec()


def spoken(text: str, rate: int = 16000) -> np.ndarray:
    return CODEC.encode(text, rate)


def heard(driver) -> str:
    """What a visitor would hear, decoded back to text."""
    return CODEC.decode(driver.audio_samples(), driver.audio[0].sample_rate) if driver.audio else ""


async def test_wav_question_gets_spoken_reply_in_same_session(make_backends, drivers, stub, tmp_path):
    question = "what is a good beach near los angeles?"
    wav = tmp_path / "question.wav"
    write_wav(wav, spoken(question, 48000), 48000)
    rate, audio = read_wav(wav)

    backends = make_backends()
    driver = await drivers(backends, "visitor-a", input_rate=rate)
    await driver.say(audio)
    await driver.wait_until(lambda: driver.audio, what="reply audio")
    await driver.wait_idle()

    session = backends.store.peek("visitor-a")
    assert [t.user for t in session.turns] == [question]
    assert len(session.turns) == 1
    assert session.turns[0].assistant == f"Happy to help with that. You asked: {question}"
    assert {a.sample_rate for a in driver.audio} == {24000}
    assert heard(driver) == CODEC.normalise(session.turns[0].assistant)
    # The chat panel got the transcript.
    assert driver.ui and driver.ui[-1].args[0][-1]["role"] == "assistant"
    # The LLM saw the system prompt and the question.
    sent = stub.requests[0]["messages"]
    assert sent[0]["role"] == "system" and sent[-1] == {"role": "user", "content": question}


async def test_two_concurrent_sessions_do_not_share_history(make_backends, drivers, stub):
    backends = make_backends()
    a = await drivers(backends, "visitor-a")
    b = await drivers(backends, "visitor-b")
    await asyncio.gather(a.say(spoken("my name is ada.")), b.say(spoken("my name is bob.")))
    await asyncio.gather(a.wait_idle(), b.wait_idle())
    await a.say(spoken("what is my name?"))
    await a.wait_idle()

    sa, sb = backends.store.peek("visitor-a"), backends.store.peek("visitor-b")
    assert [t.user for t in sa.turns] == ["my name is ada.", "what is my name?"]
    assert [t.user for t in sb.turns] == ["my name is bob."]
    last = stub.requests[-1]["messages"]
    texts = " ".join(str(m["content"]) for m in last)
    assert "ada" in texts and "bob" not in texts
    # Each visitor only heard their own replies.
    assert "bob" not in heard(a)
    assert "ada" not in heard(b)


async def test_clear_resets_only_the_callers_session(make_backends, drivers):
    backends = make_backends()
    a = await drivers(backends, "visitor-a")
    b = await drivers(backends, "visitor-b")
    await asyncio.gather(a.say(spoken("hello there.")), b.say(spoken("hi again.")))
    await asyncio.gather(a.wait_idle(), b.wait_idle())

    assert clear_session(backends, "visitor-a") == []
    assert backends.store.peek("visitor-a").turns == []
    assert len(backends.store.peek("visitor-b").turns) == 1
    # Unknown or empty ids are a no-op.
    assert clear_session(backends, None) == []


async def test_llm_error_speaks_fallback_and_keeps_history(make_backends, drivers, stub):
    stub.brain = lambda body: Plan(status=500)
    backends = make_backends()
    driver = await drivers(backends, "visitor-a")
    await driver.say(spoken("book me a trip."))
    await driver.wait_until(lambda: driver.audio, what="fallback audio")
    await driver.wait_idle()

    assert backends.store.peek("visitor-a").turns == []
    assert heard(driver) == CODEC.normalise(LLM_FALLBACK_LINE)
    # The next turn works and does not carry an orphan user message.
    stub.brain = travel_brain
    await driver.say(spoken("hello, are you there?"))
    await driver.wait_idle()
    turns = backends.store.peek("visitor-a").turns
    assert [t.user for t in turns] == ["hello, are you there?"]
    users = [m for m in stub.requests[-1]["messages"] if m["role"] == "user"]
    assert users == [{"role": "user", "content": "hello, are you there?"}]


async def test_reply_only_after_the_pause(make_backends, drivers):
    backends = make_backends(silence_ms=700)
    driver = await drivers(backends, "visitor-a")
    await driver.feed(np.concatenate([spoken("i am still"), np.zeros(int(16000 * 0.3), np.float32)]))
    await asyncio.sleep(0.2)
    assert backends.store.peek("visitor-a").turns == []
    assert not driver.audio
    await driver.feed(np.zeros(int(16000 * 1.0), np.float32))
    await driver.wait_until(lambda: driver.audio, what="reply after pause")


def test_no_server_side_audio_playback():
    banned = {"sounddevice", "simpleaudio", "pyaudio"}
    for path in [REPO_ROOT / "conv_agent.py", *sorted((REPO_ROOT / "voice_agent").rglob("*.py"))]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not banned & {n.split(".")[0] for n in names}, f"{path} imports a local audio device library"
    reqs = (REPO_ROOT / "requirements.txt").read_text().lower()
    assert "sounddevice" not in reqs


def test_endpoint_and_model_come_from_env():
    s = load_settings(env={"LLM_BASE_URL": "https://example.test/v1", "LLM_MODEL": "m1", "LLM_EXTRA_QUERY": "agent=true"}, env_files=())
    assert (s.llm_base_url, s.llm_model, s.llm_query) == ("https://example.test/v1", "m1", {"agent": "true"})
    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        load_settings(env={}, env_files=()).require("llm_base_url")
    source = "\n".join(p.read_text() for p in (REPO_ROOT / "voice_agent").rglob("*.py"))
    assert "agents.do-ai.run" not in source and "ondigitalocean.app" not in source


def test_missing_kokoro_model_is_a_clear_config_error(tmp_path):
    from voice_agent.backends import build_tts

    s = load_settings(env={"TTS_BACKEND": "kokoro", "TTS_MODEL_PATH": str(tmp_path / "nope.onnx")}, env_files=())
    with pytest.raises(ConfigError) as err:
        build_tts(s)
    assert "TTS_MODEL_PATH" in str(err.value) and "nope.onnx" in str(err.value)


def test_ui_builds_with_webrtc_component(make_backends, stub):
    import gradio as gr
    from fastrtc import WebRTC

    backends = make_backends()
    demo = build_ui(backends, fake_settings(stub))
    kinds = {type(b) for b in demo.blocks.values()}
    assert WebRTC in kinds and gr.Chatbot in kinds and gr.Button in kinds


def test_output_rate_follows_tts(make_backends):
    from voice_agent.handler import VoiceHandler

    backends = make_backends()
    backends.tts = FakeTTS(sample_rate=22050)
    handler = VoiceHandler(backends).copy()
    assert handler.output_sample_rate == 22050 and handler.input_sample_rate == 16000


@pytest.mark.realmodels
async def test_real_models_end_to_end(stub, drivers, tmp_path):
    """Kokoro speaks a question into a WAV, local Whisper transcribes it, Kokoro answers."""
    from .conftest import require_real_models, whisper_tiny_available

    paths = require_real_models("kokoro", "voices")
    if not whisper_tiny_available():
        pytest.skip("openai/whisper-tiny is not in the Hugging Face cache")
    from voice_agent.backends import build_backends
    from voice_agent.tts import KokoroTTS

    settings = fake_settings(
        stub,
        stt_backend="transformers",
        stt_model="openai/whisper-tiny",
        stt_device="cpu",
        tts_backend="kokoro",
        tts_model_path=str(paths["kokoro"]),
        tts_voices_path=str(paths["voices"]),
        tts_voice="af_sarah",
        vad_backend="silero",
    )
    backends = build_backends(settings)
    assert isinstance(backends.tts, KokoroTTS)
    question = await backends.tts.synthesize("What is the weather like in San Diego?")
    wav = tmp_path / "q.wav"
    write_wav(wav, question, 24000)
    rate, audio = read_wav(wav)

    driver = await drivers(backends, "real", input_rate=rate)
    await driver.say(audio)
    await driver.wait_until(lambda: driver.audio, timeout=60, what="reply audio")
    await driver.wait_idle(timeout=60)
    turns = backends.store.peek("real").turns
    assert len(turns) == 1
    assert "san diego" in turns[0].user.lower()
    assert {a.sample_rate for a in driver.audio} == {24000}
    assert driver.audio_samples().size > 24000 * 0.5
