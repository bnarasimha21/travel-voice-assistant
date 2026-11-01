"""Issue 3: tool calls inside a spoken turn, through the real handler.

The stub LLM scripts the tool call; the real registry runs the real tools
against the offline fixtures; the stub then reads the result back.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date

import numpy as np
import pytest

from voice_agent.fakes import FakeCodec
from voice_agent.testing.stub_llm import Plan, travel_brain
from voice_agent.tools import ToolRegistry
from voice_agent.tools.dates import resolve_date
from voice_agent.tools.flights import FixtureFlightProvider, FlightProvider, flights_tool
from voice_agent.tools.weather import FixtureWeatherProvider

from .conftest import fake_settings

CODEC = FakeCodec()


def spoken(text: str) -> np.ndarray:
    return CODEC.encode(text, 16000)


def squash(text: str) -> str:
    return "".join(CODEC.normalise(text).split())


def heard(driver) -> str:
    if not driver.audio:
        return ""
    return squash(CODEC.decode(driver.audio_samples(), driver.audio[0].sample_rate))


def tool_messages(request: dict) -> list[dict]:
    return [m for m in request["messages"] if m["role"] == "tool"]


async def test_flight_question_calls_search_flights(make_backends, drivers, stub, tmp_path):
    from voice_agent.audio import read_wav, write_wav

    question = "find me a flight from lax to san francisco tomorrow."
    wav = tmp_path / "flight.wav"
    write_wav(wav, CODEC.encode(question, 24000), 24000)
    rate, audio = read_wav(wav)

    backends = make_backends()
    driver = await drivers(backends, "v", input_rate=rate)
    await driver.say(audio)
    await driver.wait_until(lambda: driver.audio, what="reply audio")
    await driver.wait_idle()

    assert {t["function"]["name"] for t in stub.requests[0]["tools"]} == {"get_weather", "search_flights", "search_hotels"}
    first_call = stub.requests[1]["messages"][-2]["tool_calls"][0]["function"]
    assert first_call["name"] == "search_flights"
    assert json.loads(first_call["arguments"]) == {"origin": "LAX", "destination": "SFO", "date": "tomorrow"}
    # The stub's second request carries the real tool's fixture result.
    result = json.loads(tool_messages(stub.requests[1])[0]["content"])
    assert result["date"] == (date.today().fromordinal(date.today().toordinal() + 1)).isoformat()
    assert result["flights"][0]["flight_number"] == "WN 2287"

    turn = backends.store.peek("v").turns[0]
    assert turn.user == question
    assert "Southwest Airlines flight WN 2287" in turn.assistant
    assert heard(driver) == squash(turn.assistant)
    # History keeps the tool round so the next turn can refer to it.
    roles = [m["role"] for m in backends.store.peek("v").history()]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # The chat panel shows the call.
    chat = driver.ui[-1].args[0]
    assert any(m.get("metadata", {}).get("title") == "Tool call: search_flights" for m in chat)


async def test_weather_question_speaks_forecast_values(make_backends, drivers, stub):
    backends = make_backends()
    driver = await drivers(backends, "v")
    await driver.say(spoken("what's the weather in san diego this saturday?"))
    await driver.wait_idle()
    first_request = stub.requests[0]["messages"]
    assert stub.requests[1]["messages"][-2]["tool_calls"][0]["function"]["name"] == "get_weather"
    expected = await FixtureWeatherProvider().forecast("San Diego", resolve_date("saturday"))
    answer = backends.store.peek("v").turns[0].assistant
    assert f"high of {expected['high_f']}" in answer and f"low of {expected['low_f']}" in answer
    assert expected["conditions"] in answer
    assert squash(f"high of {expected['high_f']}") in heard(driver)
    assert first_request[-1]["content"].startswith("what's the weather")


class SlowFlights(FlightProvider):
    source = "slow test provider"

    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    async def search(self, origin, destination, day, adults, cabin):
        await asyncio.sleep(self.delay_s)
        return await FixtureFlightProvider().search(origin, destination, day, adults, cabin)


def flight_backends(make_backends, delay_s: float, timeout_s: float = 5.0):
    backends = make_backends()
    backends.tools = ToolRegistry([flights_tool(SlowFlights(delay_s))], timeout_s=timeout_s)
    return backends


FILLER = "Let me look up flights for you."
APOLOGY = "Sorry, I couldn't search flights just now."


async def test_filler_plays_when_tool_is_slow(make_backends, drivers, stub):
    backends = flight_backends(make_backends, delay_s=0.8)
    driver = await drivers(backends, "v")
    await driver.say(spoken("i want to fly from lax to sfo tomorrow."))
    await driver.wait_idle()
    out = heard(driver)
    assert out.startswith(squash(FILLER))
    assert out.count(squash(FILLER)) == 1
    # The filler is not part of the stored answer.
    assert "look up flights" not in backends.store.peek("v").turns[0].assistant


async def test_no_filler_on_fast_tool(make_backends, drivers, stub):
    backends = flight_backends(make_backends, delay_s=0.0)
    driver = await drivers(backends, "v")
    await driver.say(spoken("i want to fly from lax to sfo tomorrow."))
    await driver.wait_idle()
    assert squash(FILLER) not in heard(driver)
    assert "WN 2287" in backends.store.peek("v").turns[0].assistant


async def test_tool_timeout_apologises_and_conversation_continues(make_backends, drivers, stub):
    backends = flight_backends(make_backends, delay_s=3.0, timeout_s=0.5)
    driver = await drivers(backends, "v")
    await driver.say(spoken("i want to fly from lax to sfo tomorrow."))
    await driver.wait_idle(timeout=20)
    assert json.loads(tool_messages(stub.requests[1])[0]["content"]) == {"error": "search_flights timed out"}
    first = backends.store.peek("v").turns[0]
    assert first.assistant.startswith(APOLOGY)
    assert squash(APOLOGY) in heard(driver)

    n = len(driver.audio)
    await driver.say(spoken("never mind, thanks for the help."))
    await driver.wait_idle()
    turns = backends.store.peek("v").turns
    assert len(turns) == 2 and turns[1].assistant.endswith("never mind, thanks for the help.")
    assert len(driver.audio) > n


async def test_tool_error_from_bad_place_apologises(make_backends, drivers, stub):
    backends = make_backends()
    driver = await drivers(backends, "v")
    await driver.say(spoken("what's the weather in las vegas tomorrow?"))
    await driver.wait_idle()
    # Las Vegas is in the fixture, so this one succeeds.
    assert "Las Vegas" in backends.store.peek("v").turns[0].assistant
    def brain(body):
        if body["messages"][-1]["role"] == "tool":
            return travel_brain(body)
        return Plan(tool_calls=[("get_weather", {"location": "Atlantis", "date": "tomorrow"})])

    stub.brain = brain
    await driver.say(spoken("and how about atlantis?"))
    await driver.wait_idle()
    answer = backends.store.peek("v").turns[1].assistant
    assert answer.startswith("Sorry, I couldn't get the weather just now.")


async def test_tools_off_sends_no_tools(make_backends, drivers, stub):
    backends = make_backends(tools="none")
    driver = await drivers(backends, "v")
    await driver.say(spoken("what's the weather in san diego?"))
    await driver.wait_idle()
    assert "tools" not in stub.requests[0]
    assert len(stub.requests) == 1


@pytest.mark.realmodels
async def test_real_models_flight_question(stub, drivers):
    from voice_agent.backends import build_backends

    from .conftest import require_real_models, whisper_tiny_available

    paths = require_real_models("kokoro", "voices")
    if not whisper_tiny_available():
        pytest.skip("openai/whisper-tiny is not in the Hugging Face cache")
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
    question = await backends.tts.synthesize("Find me a flight from LAX to San Francisco tomorrow.")
    driver = await drivers(backends, "real", input_rate=24000)
    await driver.say(question)
    await driver.wait_until(lambda: len(stub.requests) >= 2, timeout=60, what="tool round")
    await driver.wait_idle(timeout=60)
    call = stub.requests[1]["messages"][-2]["tool_calls"][0]["function"]
    assert call["name"] == "search_flights"
    assert json.loads(call["arguments"])["destination"] == "SFO"
    assert "flight" in tool_messages(stub.requests[1])[0]["content"]
    turn = backends.store.peek("real").turns[0]
    assert "flight" in turn.assistant.lower()
    assert driver.audio_samples().size > 24000
