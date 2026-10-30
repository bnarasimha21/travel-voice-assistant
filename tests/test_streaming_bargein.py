"""Issue 2: streaming replies and barge-in, through the real handler."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from voice_agent.fakes import FakeCodec
from voice_agent.session import INTERRUPTED_MARK
from voice_agent.testing.stub_llm import Plan, travel_brain

CODEC = FakeCodec()

LONG_REPLY = (
    "Here is a long answer about the coast. Start in Santa Barbara for the morning, then drive south "
    "along the coast road. Stop in Malibu for lunch by the pier. In the afternoon walk the boardwalk in "
    "Venice. End the day with sunset in Laguna and a late dinner by the water."
)


def spoken(text: str) -> np.ndarray:
    return CODEC.encode(text, 16000)


def squash(text: str) -> str:
    return "".join(CODEC.normalise(text).split())


def decode(chunks) -> str:
    if not chunks:
        return ""
    return squash(CODEC.decode(np.concatenate([c.samples for c in chunks]), chunks[0].sample_rate))


async def test_first_audio_before_llm_finishes(make_backends, drivers, stub, caplog):
    stub.brain = lambda body: Plan(text=LONG_REPLY, token_delay_s=0.03)
    backends = make_backends()
    driver = await drivers(backends, "v")
    caplog.set_level(logging.INFO, logger="voice_agent.handler")
    await driver.say(spoken("plan me a coast drive."))
    await driver.wait_until(lambda: driver.audio, what="first audio")
    # Audio is flowing while the stub is still streaming tokens.
    assert stub.outcomes[0] == "pending"
    assert any("LLM still generating: True" in r.getMessage() for r in caplog.records)
    await driver.wait_idle(timeout=30)
    assert stub.outcomes[0] == "completed"
    assert backends.store.peek("v").turns[0].assistant == LONG_REPLY


async def test_barge_in_stops_reply_and_keeps_only_spoken_text(make_backends, drivers, stub):
    def brain(body):
        if len(stub.requests) == 1:
            return Plan(text=LONG_REPLY, token_delay_s=0.04)
        return travel_brain(body)

    stub.brain = brain
    backends = make_backends(barge_in_min_speech_ms=300)
    driver = await drivers(backends, "v", pace_output=True)
    await driver.say(spoken("plan me a coast drive."))
    await driver.wait_until(lambda: len(driver.audio) >= 25, what="half a second of reply audio")

    second = "actually, what about hotels in san diego?"
    await driver.say(spoken(second), realtime=True, trailing_silence_s=1.0)
    assert driver.handler.barge_ins, "talking over the reply did not interrupt it"
    assert stub.wait_for_outcome(0) == "cancelled"
    await driver.wait_idle(timeout=30)

    cut = driver.handler.barge_ins[0]
    before = [a for a in driver.audio if a.at <= cut]
    after = [a for a in driver.audio if a.at > cut]
    # Playback of the first reply stopped at the barge-in: everything after it is the second reply.
    turns = backends.store.peek("v").turns
    assert len(turns) == 2
    first, second_turn = turns
    assert decode(after) == squash(second_turn.assistant)

    # History holds only the part of the first reply that was heard.
    assert first.interrupted
    assert first.assistant, "some of the first reply was heard"
    assert LONG_REPLY.startswith(first.assistant)
    assert len(first.assistant) < len(LONG_REPLY) / 2
    heard = decode(before)
    assert squash(LONG_REPLY).startswith(heard[:-2])
    assert heard.startswith(squash(first.assistant))
    # At most one segment's worth of rounding between what was heard and what was stored.
    assert len(heard) - len(squash(first.assistant)) < 40

    # The second reply answers the new utterance, and the model was told about the cut.
    assert second_turn.user == second and not second_turn.interrupted
    sent = stub.requests[-1]["messages"]
    assert sent[-1] == {"role": "user", "content": second}
    assert sent[-2]["role"] == "assistant" and sent[-2]["content"].endswith(INTERRUPTED_MARK)


async def test_speech_before_reply_is_heard_merges_turns(make_backends, drivers, stub):
    stub.brain = lambda body: Plan(text="Sure thing. " + travel_brain(body).text, first_token_delay_s=1.5)
    backends = make_backends(silence_ms=500, barge_in_min_speech_ms=200)
    driver = await drivers(backends, "v", pace_output=True)
    await driver.feed(np.concatenate([spoken("i want to go to"), np.zeros(int(16000 * 0.6), np.float32)]), realtime=True)
    await driver.wait_until(lambda: stub.requests, what="first LLM request")
    await driver.say(spoken("palm springs."), realtime=True, trailing_silence_s=0.8)
    assert stub.wait_for_outcome(0) == "cancelled"
    await driver.wait_idle(timeout=30)
    turns = backends.store.peek("v").turns
    assert len(turns) == 1
    assert turns[0].user.startswith("i want to go to") and turns[0].user.endswith("palm springs.")


async def test_barge_in_disabled_ignores_speech(make_backends, drivers, stub):
    stub.brain = lambda body: Plan(text=LONG_REPLY[:120], token_delay_s=0.01)
    backends = make_backends(barge_in=False)
    driver = await drivers(backends, "v", pace_output=True)
    await driver.say(spoken("plan me a coast drive."))
    await driver.wait_until(lambda: len(driver.audio) >= 10, what="reply audio")
    await driver.say(spoken("stop talking please."), realtime=True, trailing_silence_s=0.3)
    await driver.wait_idle(timeout=30)
    assert not driver.handler.barge_ins
    turns = backends.store.peek("v").turns
    assert len(turns) == 1 and not turns[0].interrupted


async def test_clear_during_reply_stops_it(make_backends, drivers, stub):
    from voice_agent.app import clear_session

    stub.brain = lambda body: Plan(text=LONG_REPLY, token_delay_s=0.04)
    backends = make_backends()
    driver = await drivers(backends, "v", pace_output=True)
    await driver.say(spoken("plan me a coast drive."))
    await driver.wait_until(lambda: driver.audio, what="reply audio")
    # Gradio runs the Clear handler on a worker thread.
    await asyncio.to_thread(clear_session, backends, "v")
    await driver.wait_until(lambda: not driver.handler.is_replying, what="reply stop")
    assert stub.wait_for_outcome(0) == "cancelled"
    n = len(driver.audio)
    await asyncio.sleep(0.3)
    assert len(driver.audio) == n
    assert backends.store.peek("v").turns == []
