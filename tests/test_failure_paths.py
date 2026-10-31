"""Failure paths and edge cases found in review, each through the real code.

Cascade replies that hit an STT or TTS error, short sounds during a reply,
Clear racing a commit, blank config values, live flight codes, tool calls
without an index, the S2S handler's tool rounds and late transcripts, Gemini
tool declarations, Flux reconnects, Smart Turn time in TTFA and the
streaming resampler.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np

from voice_agent.audio import StreamResampler, resample
from voice_agent.config import load_settings
from voice_agent.deepgram import FluxTurnSource
from voice_agent.fakes import FakeCodec
from voice_agent.llm import OpenAICompatibleLLM, ToolCalls
from voice_agent.metrics import TurnTimer
from voice_agent.realtime import S2SBackends, S2SHandler
from voice_agent.realtime.base import AudioDelta, AssistantText, RealtimeToolCall, ResponseDone, UserTranscript
from voice_agent.realtime.gemini_live import GeminiLiveAdapter
from voice_agent.realtime.handler import S2SReply
from voice_agent.session import Session, SessionStore
from voice_agent.testing.stub_llm import Plan, travel_brain
from voice_agent.tools import build_registry
from voice_agent.tools.base import Tool, ToolRegistry
from voice_agent.tools.places import airport_code
from voice_agent.turn import SilenceTurnDetector, TurnTaker
from voice_agent.vad import EnergyVAD

from .conftest import FAKE_ENV

CODEC = FakeCodec()
LONG_REPLY = (
    "Here is a long answer about the coast. Start in Santa Barbara for the morning, then drive south "
    "along the coast road. Stop in Malibu for lunch by the pier. End the day with sunset in Laguna."
)


def spoken(text: str) -> np.ndarray:
    return CODEC.encode(text, 16000)


def tone(seconds: float, rate: int = 16000, hz: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# Cascade ---------------------------------------------------------------------


class _FailingOnce:
    """Wraps the fake TTS: the first synthesis raises."""

    def __init__(self, tts):
        self._tts, self.calls = tts, 0
        self.sample_rate = tts.sample_rate

    async def synthesize(self, text: str):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("TTS crashed")
        return await self._tts.synthesize(text)


async def test_a_tts_error_ends_the_reply_and_the_next_question_is_answered(make_backends, drivers, stub):
    backends = make_backends(barge_in=False)
    backends.tts = _FailingOnce(backends.tts)
    driver = await drivers(backends, "v")
    await driver.say(spoken("what's the weather in san diego tomorrow?"))
    await driver.wait_until(lambda: not driver.handler.is_replying and driver.handler.turn_records, what="failed reply to end")
    assert driver.handler.turn_records[-1]["outcome"] == "failed"
    assert backends.store.peek("v").turns == []  # nothing half-done was committed
    # The handler is listening again: the next question gets a full answer.
    await driver.say(spoken("what's the weather in san diego tomorrow?"))
    await driver.wait_idle()
    turns = backends.store.peek("v").turns
    assert len(turns) == 1 and "San Diego" in turns[0].assistant


async def test_a_short_sound_during_the_reply_does_not_cut_it(make_backends, drivers, stub):
    stub.brain = lambda body: Plan(text=LONG_REPLY, token_delay_s=0.01)
    backends = make_backends(barge_in_min_speech_ms=800)
    driver = await drivers(backends, "v", pace_output=True)
    await driver.say(spoken("plan me a coast drive."))
    await driver.wait_until(lambda: len(driver.audio) >= 10, what="reply audio")
    # 400 ms is a turn (over MIN_SPEECH_MS) but shorter than a barge-in.
    await driver.feed(np.concatenate([tone(0.4), np.zeros(16000, np.float32)]), realtime=True)
    await driver.wait_idle(timeout=30)
    assert not driver.handler.barge_ins
    turns = backends.store.peek("v").turns
    assert len(turns) == 1 and not turns[0].interrupted and turns[0].assistant == LONG_REPLY
    assert len(stub.requests) == 1


def test_a_commit_from_before_a_reset_is_dropped():
    session = Session("s")
    generation = session.generation
    session.reset()
    assert session.commit_turn("hi", "hello", generation=generation) is None
    assert session.turns == []
    assert session.commit_turn("hi", "hello", generation=session.generation) is not None


async def test_tool_apology_is_kept_in_history_across_a_retry(make_backends, drivers, stub):
    def brain(body):
        tool_results = [m for m in body["messages"] if m["role"] == "tool"]
        if not tool_results:
            return Plan(tool_calls=[("get_weather", {"location": "Atlantis", "date": "tomorrow"})])
        if len(tool_results) == 1:
            return Plan(tool_calls=[("get_weather", {"location": "San Diego", "date": "tomorrow"})])
        return travel_brain(body)

    stub.brain = brain
    backends = make_backends()
    driver = await drivers(backends, "v")
    await driver.say(spoken("what's the weather in atlantis?"))
    await driver.wait_idle()
    answer = backends.store.peek("v").turns[0].assistant
    # The apology was spoken, so it stays in what the model is told it said.
    assert answer.startswith("Sorry, I couldn't get the weather just now.")
    assert "San Diego" in answer


async def test_a_malformed_tool_call_is_left_to_the_model(make_backends, drivers, stub):
    def brain(body):
        if body["messages"][-1]["role"] == "tool":
            return Plan(text="Which city do you mean?")
        return Plan(tool_calls=[("get_weather", {"date": "tomorrow"})])  # no location

    stub.brain = brain
    backends = make_backends()
    driver = await drivers(backends, "v")
    await driver.say(spoken("what's the weather tomorrow?"))
    await driver.wait_idle()
    turn = backends.store.peek("v").turns[0]
    assert json.loads(turn.tool_messages[1]["content"]) == {"error": "missing arguments: location"}
    assert turn.assistant == "Which city do you mean?"  # no service apology


async def test_registry_marks_malformed_calls():
    async def handler(x: str):
        return {"x": x}

    reg = ToolRegistry([Tool("t", "d", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}, handler)])
    assert (await reg.run("nope", "{}")).model_error
    assert (await reg.run("t", "not json")).model_error
    assert (await reg.run("t", "{}")).model_error
    ok = await reg.run("t", '{"x": "1"}')
    assert ok.ok and not ok.model_error


# Config and tools ------------------------------------------------------------


def test_blank_values_fall_back_to_the_lower_layer(tmp_path):
    example = tmp_path / ".env.example"
    example.write_text("TOOLS=weather\nLLM_MODEL=from-example\nMETRICS_LOG=none\n")
    dotenv = tmp_path / ".env"
    dotenv.write_text("TOOLS=\nLLM_MODEL=from-dotenv\n")
    settings = load_settings(env={"LLM_MODEL": ""}, env_files=(dotenv, example))
    assert settings.tools == "weather"
    assert settings.llm_model == "from-dotenv"
    assert settings.metrics_log_path is None


async def test_live_flight_search_takes_any_airport_code_and_ends_before_the_tool_timeout():
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(201, json={"data": {"offers": []}})

    settings = load_settings(
        env=dict(FAKE_ENV, FLIGHTS_PROVIDER="duffel", DUFFEL_API_KEY="k", DUFFEL_BASE_URL="https://api.duffel.test", TOOL_TIMEOUT_S="8"),
        env_files=(),
    )
    registry = build_registry(settings, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await registry.run("search_flights", {"origin": "ORD", "destination": "blr", "date": "tomorrow"})
    assert result.ok, result.data
    body = json.loads(seen[0].content)["data"]["slices"][0]
    assert (body["origin"], body["destination"]) == ("ORD", "BLR")
    assert seen[0].url.params["supplier_timeout"] == "6500"
    # The sample data still only knows its own airports.
    assert airport_code("ORD") is None and airport_code("LAX") == "LAX"


async def test_tool_calls_without_an_index_are_reassembled():
    def chunk(tc=None, finish=None):
        delta = {"role": "assistant", **({"tool_calls": [tc]} if tc else {})}
        return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    chunks = [
        chunk({"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"loc'}}),
        chunk({"function": {"arguments": 'ation": "x"}'}}),
        chunk({"id": "b", "type": "function", "function": {"name": "search_hotels", "arguments": "{}"}}),
        chunk(finish="tool_calls"),
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    transport = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode()))
    llm = OpenAICompatibleLLM("http://llm.test/v1", "k", "m", http_client=httpx.AsyncClient(transport=transport))
    calls = [e for e in [e async for e in llm.stream([{"role": "user", "content": "hi"}])] if isinstance(e, ToolCalls)]
    assert [(c.id, c.name, c.arguments) for c in calls[0].calls] == [
        ("a", "get_weather", '{"location": "x"}'),
        ("b", "search_hotels", "{}"),
    ]


# Speech to speech -------------------------------------------------------------


class _NullAdapter:
    input_sample_rate = 16000
    output_sample_rate = 24000
    final_transcripts = True

    def __init__(self):
        self.sent: list = []
        self.interrupts: list = []

    async def connect(self, instructions, tools):
        pass

    async def send_audio(self, audio):
        pass

    async def send_tool_results(self, results):
        self.sent.append(results)

    async def interrupt(self, response_id, played_ms):
        self.interrupts.append((response_id, played_ms))

    async def events(self):
        await asyncio.Event().wait()
        yield  # pragma: no cover

    async def aclose(self):
        pass


def _s2s(tools=None, final_transcripts=True):
    adapter = _NullAdapter()
    adapter.final_transcripts = final_transcripts
    backends = S2SBackends(make_adapter=lambda: adapter, store=SessionStore(), make_vad=EnergyVAD, tools=tools, provider="openai")
    handler = S2SHandler(backends, session_id="s", pace_output=False)
    handler.adapter = adapter
    return handler, adapter, backends


async def _drain(handler: S2SHandler) -> None:
    while not handler._out.empty():
        await handler.emit()


def test_s2s_barge_in_after_a_tool_round_uses_the_answer_audio_only():
    reply = S2SReply(timer=TurnTimer(session_id="s", turn_index=0, end_of_turn_delay_ms=0))
    reply.response_starts = [(0, "r1"), (24000, "r2")]  # 1 s preamble, then the answer
    reply.text_start = 24000
    reply.received = 24000 * 5
    reply.text = "one two three four five six seven eight"
    # 0.5 s into the 4 s answer: an eighth of it was heard.
    assert reply.spoken_text(24000 + 12000) == "one"
    assert reply.spoken_text(12000) == ""  # still in the preamble
    assert reply.playing_response(24000 + 12000) == ("r2", 12000)
    assert reply.playing_response(12000) == ("r1", 12000)


async def test_s2s_truncates_the_response_being_played():
    handler, adapter, _ = _s2s()
    handler.bind_session()
    second = np.zeros(24000, np.float32)
    handler.on_event(UserTranscript("plan my day", final=True))
    handler.on_event(AudioDelta(second, "r1"))
    handler.on_event(AudioDelta(second, "r2"))
    handler._current.emitted = 24000 + 12000
    handler.barge_in()
    await asyncio.sleep(0)
    assert adapter.interrupts == [("r2", 500)]


async def test_s2s_late_final_transcript_lands_on_its_own_turn():
    handler, _, backends = _s2s()
    session = handler.bind_session()
    handler.on_event(AudioDelta(np.zeros(2400, np.float32), "r1"))
    handler.on_event(AssistantText("It will be sunny.", "r1"))
    handler.on_event(ResponseDone("r1"))
    await _drain(handler)
    assert [t.user for t in session.turns] == [""]
    # OpenAI sent the caller's transcript after the reply.
    handler.on_event(UserTranscript("weather in san diego", final=True))
    await _drain(handler)
    assert [t.user for t in session.turns] == ["weather in san diego"]
    # The next turn starts clean instead of inheriting it.
    handler.on_event(UserTranscript("and tomorrow", final=True))
    handler.on_event(AudioDelta(np.zeros(2400, np.float32), "r2"))
    assert handler._current.user == "and tomorrow"


async def test_s2s_without_final_transcripts_does_not_wait_for_one():
    handler, _, _ = _s2s(final_transcripts=False)
    handler.bind_session()
    handler.on_event(AudioDelta(np.zeros(2400, np.float32), "r1"))
    handler.on_event(ResponseDone("r1"))
    await _drain(handler)
    handler.on_event(UserTranscript("next ", final=False))
    handler.on_event(AudioDelta(np.zeros(2400, np.float32), "r2"))
    assert handler._current.user == "next"


async def test_s2s_turn_end_while_tools_run_waits_for_the_answer():
    release = asyncio.Event()

    async def slow(location: str):
        await release.wait()
        return {"forecast": "sunny"}

    tools = ToolRegistry([Tool("get_weather", "d", {"type": "object", "properties": {"location": {"type": "string"}}}, slow)])
    handler, adapter, _ = _s2s(tools=tools)
    session = handler.bind_session()
    call = RealtimeToolCall("c1", "get_weather", '{"location": "San Diego"}')
    handler.on_event(UserTranscript("weather in san diego", final=True))
    handler.on_event(ResponseDone("r1", [call]))
    handler.on_event(ResponseDone("r1"))  # a non-blocking model finished its turn early
    await _drain(handler)
    assert handler._current is not None and session.turns == []
    release.set()
    await asyncio.sleep(0.05)
    assert adapter.sent and adapter.sent[0][0][0].id == "c1"
    handler.on_event(AudioDelta(np.zeros(2400, np.float32), "r2"))
    handler.on_event(AssistantText("Sunny in San Diego.", "r2"))
    handler.on_event(ResponseDone("r2"))
    await _drain(handler)
    assert session.turns[0].assistant == "Sunny in San Diego."
    assert [m["role"] for m in session.turns[0].tool_messages] == ["assistant", "tool"]


def test_gemini_tools_are_declared_blocking():
    tools = ToolRegistry([Tool("get_weather", "d", {"type": "object", "properties": {}}, None)])
    setup = GeminiLiveAdapter(url="wss://x", api_key="k", model="m", voice="v").setup_message("hi", tools)
    declarations = setup["setup"]["tools"][0]["functionDeclarations"]
    assert declarations[0]["behavior"] == "BLOCKING" and declarations[0]["name"] == "get_weather"


# Streaming STT, turn timing, audio --------------------------------------------


class _BrokenSocket:
    def __init__(self):
        self.closed = False

    async def send(self, data):
        raise ConnectionError("gone")

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()


async def test_flux_send_failure_closes_the_socket_and_its_reader():
    sockets = []

    async def connect(url, additional_headers=None):
        sockets.append(_BrokenSocket())
        return sockets[-1]

    flux = FluxTurnSource(url="wss://flux.test", api_key="k", model="flux", vad=EnergyVAD(), connect=connect)
    flux.in_speech, flux.speech_ms = True, 500.0
    await flux.feed(np.zeros(1280 * 2, np.float32))
    reader = sockets and flux._reader
    assert sockets[0].closed and flux._ws is None and not reader
    assert not flux.in_speech and flux.speech_ms == 0  # the half-heard turn went with the stream
    await flux.aclose()


async def test_flux_closed_while_connecting_does_not_leave_a_socket():
    gate = asyncio.Event()
    sockets = []

    async def connect(url, additional_headers=None):
        await gate.wait()
        sockets.append(_BrokenSocket())
        return sockets[-1]

    flux = FluxTurnSource(url="wss://flux.test", api_key="k", model="flux", vad=EnergyVAD(), connect=connect)
    feeding = asyncio.create_task(flux.feed(np.zeros(1280, np.float32)))
    await asyncio.sleep(0.01)
    await flux.aclose()
    gate.set()
    await asyncio.gather(feeding, return_exceptions=True)
    assert all(s.closed for s in sockets) and flux._ws is None


class _SlowDetector(SilenceTurnDetector):
    async def should_end(self, audio, silence_ms):
        await asyncio.sleep(0.2)  # a model taking 200 ms to decide
        return await super().should_end(audio, silence_ms)


async def test_detector_time_counts_toward_ttfa():
    taker = TurnTaker(vad=EnergyVAD(), detector=_SlowDetector(silence_ms=200))
    audio = np.concatenate([tone(0.5), np.zeros(8000, np.float32)])
    events = []
    for i in range(0, audio.size, 320):
        events += await taker.feed(audio[i : i + 320])
    turn = next(e for e in events if hasattr(e, "decided_at"))
    timer = TurnTimer(session_id="s", turn_index=0, end_of_turn_delay_ms=turn.end_of_turn_delay_ms, decided_at=turn.decided_at)
    timer.mark("first_audio_sent")
    assert timer.offset_ms("first_audio_sent") >= turn.end_of_turn_delay_ms + 190


def test_stream_resampler_has_no_clicks_at_frame_edges():
    signal = tone(1.0, rate=48000)
    whole = resample(signal, 48000, 16000)
    stream = StreamResampler(16000)
    framed = np.concatenate([stream(signal[i : i + 960], 48000) for i in range(0, signal.size, 960)])
    one_shot = np.concatenate([resample(signal[i : i + 960], 48000, 16000) for i in range(0, signal.size, 960)])
    # Line the stream up with the whole-signal result (the stream delays slightly).
    lag = int(np.argmax(np.correlate(framed[:2000], whole[:1000], "valid")))
    n = min(framed.size - lag, whole.size) - 200
    stream_err = np.abs(framed[lag + 200 : lag + n] - whole[200:n]).max()
    one_shot_err = np.abs(one_shot[200:n] - whole[200:n]).max()
    assert stream_err < 0.01 < one_shot_err
