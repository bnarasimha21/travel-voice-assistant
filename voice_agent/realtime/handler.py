"""`PIPELINE=s2s`: one FastRTC handler per browser connection, talking to a
speech-to-speech realtime model instead of running STT, an LLM and TTS here.

The browser side is identical to the cascade: the same page, the same
WebRTC audio track, the same chat panel and latency breakdown. Microphone
audio is streamed straight to the provider; the provider decides when you
finished, speaks back, and asks for the same travel tools, which run here
through the same `ToolRegistry` as the cascade.

What stays local:
- a VAD on the microphone, to timestamp when you stopped talking (so TTFA is
  measured from the same point as in the cascade), and to catch you talking
  over a reply the provider has already finished sending: providers send
  audio faster than real time and only report an interruption while they
  are still generating
- playback pacing, so a barge-in knows how much of the reply you heard; the
  stored assistant text is cut to that share, and the provider is told to
  forget the unplayed audio (OpenAI `conversation.item.truncate`)
- the session history and the per-turn latency records
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from fastrtc.utils import current_context

from ..audio import STT_SAMPLE_RATE, resample, to_mono_float32
from ..metrics import TurnLog, TurnTimer, format_breakdown, ttfa_summary
from ..prompts import system_prompt
from ..session import Session, SessionStore
from ..vad import VAD, WINDOW_SAMPLES
from .base import (
    AdapterError,
    AssistantText,
    AudioDelta,
    RealtimeAdapter,
    RealtimeToolCall,
    ResponseDone,
    UserSpeechStarted,
    UserSpeechStopped,
    UserTranscript,
)

log = logging.getLogger(__name__)

FRAME_MS = 20
PLAYBACK_LEAD_S = 0.06
RECONNECT_AFTER_S = 2.0
HISTORY_TURNS = 12  # prior turns replayed into the instructions after a reconnect


@dataclass
class S2SBackends:
    make_adapter: Callable[[], RealtimeAdapter]
    store: SessionStore
    make_vad: Callable[[], VAD]
    tools: Any = None
    vad_threshold: float = 0.5
    turn_log: TurnLog | None = None
    provider: str = ""
    output_sample_rate: int = 24000
    # The provider only reports the caller talking over a reply it is still
    # generating. Once it has sent the whole reply, playback can run for many
    # seconds more, so the local VAD handles barge-in for that stretch.
    barge_in: bool = True
    barge_in_min_speech_ms: float = 300


@dataclass
class S2SReply:
    timer: TurnTimer
    user: str = ""
    user_final: bool = False
    carried: str = ""  # words from a cut-off turn nobody heard a reply to
    text: str = ""  # what the model said since its last tool round
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    response_ids: set[str] = field(default_factory=set)
    last_response_id: str = ""
    received: int = 0  # reply samples received from the provider
    emitted: int = 0  # reply samples handed to the transport
    first_audio_at: float | None = None
    generated: bool = False  # the provider has sent the whole reply
    tools_task: asyncio.Task | None = None
    failed: bool = False

    def spoken_text(self, heard_samples: int) -> str:
        """The share of the transcript covered by the audio actually heard."""
        words = self.text.split()
        if not words or self.received <= 0:
            return ""
        share = min(1.0, heard_samples / self.received)
        return " ".join(words[: int(len(words) * share)])


@dataclass
class _Audio:
    samples: np.ndarray
    reply: S2SReply


@dataclass
class _UI:
    outputs: AdditionalOutputs


@dataclass
class _End:
    reply: S2SReply


class S2SHandler(AsyncStreamHandler):
    def __init__(self, backends: S2SBackends, session_id: str | None = None, pace_output: bool = True):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=backends.output_sample_rate,
            input_sample_rate=STT_SAMPLE_RATE,
        )
        self.backends = backends
        self.pace_output = pace_output
        self._session_id = session_id
        self.session: Session | None = None
        self.vad = backends.make_vad()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self._last_voiced_at = 0.0
        self._voiced_ms = 0.0  # length of the current run of speech, by the local VAD
        self._speech_stopped_at: float | None = None
        self.adapter: RealtimeAdapter | None = None
        self._reader: asyncio.Task | None = None
        self._connecting: asyncio.Task | None = None
        self._failed_at = 0.0
        self._closed = False
        self._out: asyncio.Queue[_Audio | _UI | _End] = asyncio.Queue()
        self._current: S2SReply | None = None
        self._pending_user = ""
        self._pending_final = False
        self._carry_user = ""
        self._cancelled: set[str] = set()
        self._play_end = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task] = set()
        self.barge_ins: list[float] = []
        self.turn_records: list[dict[str, Any]] = []
        self.errors: list[str] = []

    # FastRTC interface -------------------------------------------------------

    def copy(self) -> "S2SHandler":
        return S2SHandler(self.backends, pace_output=self.pace_output)

    async def start_up(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.bind_session()
        await self._ensure_adapter()

    def shutdown(self) -> None:
        self._closed = True
        self._stop_reply()
        if self.session is not None:
            self.backends.store.drop(self.session.id)
        self._run_soon(self._close_adapter())

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        self.bind_session()
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        sample_rate, data = frame
        audio = to_mono_float32(data)
        audio16 = audio if sample_rate == STT_SAMPLE_RATE else resample(audio, sample_rate, STT_SAMPLE_RATE)
        self._track_speech(audio16)
        adapter = await self._ensure_adapter()
        if adapter is None:
            return
        rate = adapter.input_sample_rate
        out = audio16 if rate == STT_SAMPLE_RATE else resample(audio, sample_rate, rate)
        try:
            await adapter.send_audio(out)
        except Exception as exc:
            self._adapter_lost(adapter, f"send failed: {exc}")

    async def emit(self):
        try:
            item = await asyncio.wait_for(self._out.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None
        if isinstance(item, _UI):
            return item.outputs
        if isinstance(item, _End):
            return self._finish_reply(item.reply)
        if item.reply is not self._current:
            return None
        await self._wait_for_playback_slot()
        if item.reply is not self._current:
            return None
        self._play_end += item.samples.size / self.output_sample_rate
        reply = item.reply
        if reply.first_audio_at is None:
            reply.first_audio_at = time.perf_counter()
            reply.timer.mark("first_audio_sent", reply.first_audio_at)
        reply.emitted += item.samples.size
        return (self.output_sample_rate, item.samples.reshape(1, -1))

    # Session and connection --------------------------------------------------

    def bind_session(self) -> Session:
        if self.session is None:
            sid = self._session_id
            if sid is None:
                ctx = current_context.get()
                sid = ctx.webrtc_id if ctx is not None else uuid.uuid4().hex
            self._session_id = sid
            self.session = self.backends.store.get(sid)
            self.session.on_reset(self._on_session_reset)
        return self.session

    @property
    def is_replying(self) -> bool:
        return self._current is not None

    def instructions(self) -> str:
        """System prompt, plus the conversation so far when a session is reopened."""
        text = system_prompt(tools=bool(self.backends.tools))
        turns = list(self.bind_session().turns)[-HISTORY_TURNS:]
        if turns:
            lines = []
            for t in turns:
                lines.append(f"Caller: {t.user}")
                lines.append(f"You: {t.assistant}" + (" (the caller interrupted you here)" if t.interrupted else ""))
            text += "\n\nThe conversation so far, continue from here:\n" + "\n".join(lines)
        return text

    async def _ensure_adapter(self) -> RealtimeAdapter | None:
        if self.adapter is not None:
            return self.adapter
        if self._closed or time.monotonic() - self._failed_at < RECONNECT_AFTER_S:
            return None
        if self._connecting is None or self._connecting.done():
            self._connecting = asyncio.create_task(self._open())
        await asyncio.shield(self._connecting)
        return self.adapter

    async def _open(self) -> None:
        try:
            adapter = self.backends.make_adapter()
            await adapter.connect(self.instructions(), self.backends.tools)
        except Exception as exc:
            self._failed_at = time.monotonic()
            self._notice(f"Could not reach the {self.backends.provider or 'realtime'} model: {exc}")
            return
        if self._closed:
            await adapter.aclose()
            return
        self.adapter = adapter
        self._reader = asyncio.create_task(self._read(adapter))

    async def _close_adapter(self) -> None:
        adapter, self.adapter = self.adapter, None
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        if adapter is not None:
            await adapter.aclose()
        for task in list(self._tasks):
            task.cancel()

    def _adapter_lost(self, adapter: RealtimeAdapter, why: str) -> None:
        if adapter is not self.adapter:
            return
        self.adapter = None
        self._failed_at = time.monotonic()
        self._spawn(adapter.aclose())
        if self._current is not None:
            self._current.failed = True
            self._out.put_nowait(_End(self._current))
        if not self._closed:
            self._notice(f"Lost the connection to the {self.backends.provider or 'realtime'} model ({why}); reconnecting.")

    def _notice(self, message: str) -> None:
        log.warning("session %s: %s", self._session_id, message)
        self.errors.append(message)
        chat = self.bind_session().chat_messages() + [{"role": "assistant", "content": f"({message})"}]
        self._out.put_nowait(_UI(AdditionalOutputs(chat, self.latency_markdown())))

    def _on_session_reset(self) -> None:
        """Clear conversation: drop the reply and start a fresh provider session."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._reset_now()
        else:
            loop.call_soon_threadsafe(self._reset_now)

    def _reset_now(self) -> None:
        self._stop_reply()
        self._pending_user, self._pending_final, self._carry_user = "", False, ""
        self._spawn(self._close_adapter())  # the next microphone frame reconnects with no history

    def _run_soon(self, coro) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._spawn(coro)
        else:
            loop.call_soon_threadsafe(lambda: self._spawn(coro))

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # Provider events ---------------------------------------------------------

    async def _read(self, adapter: RealtimeAdapter) -> None:
        try:
            async for event in adapter.events():
                if adapter is not self.adapter:
                    return
                if isinstance(event, AdapterError):
                    if event.fatal:
                        self._adapter_lost(adapter, event.message)
                        return
                    log.warning("session %s: %s", self._session_id, event.message)
                    self.errors.append(event.message)
                    continue
                self.on_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a bug in event handling must not leave a dead session
            log.exception("session %s: realtime event handling failed", self._session_id)
            self._adapter_lost(adapter, str(exc))

    def on_event(self, event: Any) -> None:
        if isinstance(event, AudioDelta):
            if event.response_id in self._cancelled or event.audio.size == 0:
                return
            reply = self._ensure_reply(event.response_id)
            reply.timer.mark("llm_first_token")
            reply.timer.mark("tts_first_chunk")
            reply.received += event.audio.size
            frame = int(self.output_sample_rate * FRAME_MS / 1000)
            for start in range(0, event.audio.size, frame):
                self._out.put_nowait(_Audio(event.audio[start : start + frame], reply))
        elif isinstance(event, AssistantText):
            if event.response_id in self._cancelled or not event.text:
                return
            reply = self._ensure_reply(event.response_id)
            reply.timer.mark("llm_first_token")
            reply.text += event.text
        elif isinstance(event, UserTranscript):
            self._on_transcript(event)
        elif isinstance(event, UserSpeechStarted):
            self._speech_stopped_at = None
            if self._current is not None:
                self.barge_in()
        elif isinstance(event, UserSpeechStopped):
            if self._current is None:
                self._speech_stopped_at = time.perf_counter()
        elif isinstance(event, ResponseDone):
            self._on_response_done(event)

    def _on_transcript(self, event: UserTranscript) -> None:
        reply = self._current
        if reply is not None:
            if event.final:
                reply.user = " ".join(t for t in (reply.carried, event.text) if t)
                reply.user_final = True
            elif not reply.user_final:
                reply.user += event.text
            reply.timer.mark("stt_done")
            self._show(reply)
        elif event.final:
            self._pending_user, self._pending_final = event.text, True
        elif not self._pending_final:
            self._pending_user += event.text

    def _on_response_done(self, event: ResponseDone) -> None:
        if event.cancelled or event.response_id in self._cancelled:
            return
        reply = self._current
        if reply is None:
            if not event.calls:
                return  # nothing was said or heard for this response
            reply = self._ensure_reply(event.response_id)
        reply.last_response_id = event.response_id or reply.last_response_id
        if event.calls:
            reply.timer.mark("llm_first_token")
            reply.tools_task = self._spawn(self._run_tools(reply, event.calls))
            return
        reply.timer.mark("llm_done")
        reply.generated = True
        self._out.put_nowait(_End(reply))

    def _ensure_reply(self, response_id: str) -> S2SReply:
        reply = self._current
        if reply is None:
            now = time.perf_counter()
            speech_end = self._last_voiced_at or now
            stopped = self._speech_stopped_at
            known = stopped is not None and stopped >= speech_end
            session = self.bind_session()
            timer = TurnTimer(
                session_id=session.id,
                turn_index=len(self.turn_records),
                end_of_turn_delay_ms=(stopped - speech_end) * 1000 if known else 0.0,
                decided_at=stopped if known else speech_end,
                turn_detector=self.backends.provider,
                turn_end_known=known,
            )
            reply = S2SReply(timer=timer, carried=self._carry_user)
            reply.user = " ".join(t for t in (self._carry_user, self._pending_user.strip()) if t)
            reply.user_final = self._pending_final
            if self._pending_user:
                timer.mark("stt_done")
            self._carry_user, self._pending_user, self._pending_final = "", "", False
            self._speech_stopped_at = None
            self._current = reply
        if response_id:
            reply.response_ids.add(response_id)
            reply.last_response_id = response_id
        return reply

    async def _run_tools(self, reply: S2SReply, calls: list[RealtimeToolCall]) -> None:
        reply.timer.mark("tool_start")
        reply.timer.tools.extend(c.name for c in calls)
        registry = self.backends.tools
        assistant = {"role": "assistant", "content": reply.text.strip() or None, "tool_calls": [c.as_chat_call() for c in calls]}
        reply.text = ""
        self._show(reply, pending=[assistant])
        if registry:
            results = await asyncio.gather(*(registry.run(c.name, c.arguments) for c in calls))
            outputs = [r.content for r in results]
        else:
            outputs = ['{"error": "no tools are enabled"}' for _ in calls]
        if reply is not self._current:
            return  # interrupted while the tools ran
        reply.tool_messages.append(assistant)
        reply.tool_messages.extend(
            {"role": "tool", "tool_call_id": c.id, "content": out} for c, out in zip(calls, outputs)
        )
        reply.timer.mark("tool_end")
        self._show(reply)
        adapter = self.adapter
        if adapter is None:
            return
        try:
            await adapter.send_tool_results(list(zip(calls, outputs)))
        except Exception as exc:
            self._adapter_lost(adapter, f"sending tool results failed: {exc}")

    # Reply lifecycle ---------------------------------------------------------

    def _show(self, reply: S2SReply, pending: list[dict[str, Any]] | None = None) -> None:
        chat = self.bind_session().chat_messages(pending_user=reply.user or "...", pending_tools=reply.tool_messages + (pending or []))
        self._out.put_nowait(_UI(AdditionalOutputs(chat, self.latency_markdown())))

    def latency_markdown(self) -> str:
        shown = [r for r in self.turn_records if r["outcome"] != "empty"]
        return format_breakdown(shown[-1] if shown else None, ttfa_summary(shown))

    def _record_turn(self, reply: S2SReply, outcome: str) -> None:
        if reply.timer.done:
            return
        record = reply.timer.finish(outcome)
        self.turn_records.append(record)
        if self.backends.turn_log is not None:
            try:
                self.backends.turn_log.write(record)
            except OSError as exc:
                log.warning("could not write turn log: %s", exc)

    def _finish_reply(self, reply: S2SReply) -> AdditionalOutputs | None:
        if reply is not self._current:
            return None
        self._current = None
        session = self.bind_session()
        if reply.failed:
            self._record_turn(reply, "failed")
            return AdditionalOutputs(session.chat_messages(), self.latency_markdown())
        if not reply.user and not reply.text and reply.received == 0:
            self._record_turn(reply, "empty")
            return None
        self._record_turn(reply, "completed")
        session.commit_turn(reply.user.strip(), reply.text.strip(), tool_messages=reply.tool_messages)
        return AdditionalOutputs(session.chat_messages(), self.latency_markdown())

    def barge_in(self) -> None:
        """The caller talked over the reply (the provider already stopped generating)."""
        reply = self._current
        if reply is None:
            return
        self.barge_ins.append(time.perf_counter())
        heard = max(0, reply.emitted - int(round(self.unplayed_seconds() * self.output_sample_rate)))
        self._record_turn(reply, "merged" if heard == 0 else "interrupted")
        self._cancelled.update(reply.response_ids)
        adapter = self.adapter
        if adapter is not None and reply.last_response_id:
            played_ms = int(heard * 1000 / self.output_sample_rate)
            self._spawn(self._interrupt(adapter, reply.last_response_id, played_ms))
        self._stop_reply()
        session = self.bind_session()
        if heard == 0:
            # Nothing was heard: the caller was still talking. The provider keeps
            # both utterances in its context; keep the words for the next turn.
            self._carry_user = reply.user.strip()
            return
        session.commit_turn(reply.user.strip(), reply.spoken_text(heard), interrupted=True, tool_messages=reply.tool_messages)
        self._out.put_nowait(_UI(AdditionalOutputs(session.chat_messages(), self.latency_markdown())))
        log.info("session %s: barge-in after %.2f s of reply audio", session.id, heard / self.output_sample_rate)

    async def _interrupt(self, adapter: RealtimeAdapter, response_id: str, played_ms: int) -> None:
        try:
            await adapter.interrupt(response_id, played_ms)
        except Exception as exc:
            log.warning("session %s: interrupt failed: %s", self._session_id, exc)

    def _stop_reply(self) -> None:
        reply, self._current = self._current, None
        if reply is not None:
            self._record_turn(reply, "cancelled")
            if reply.tools_task is not None and not reply.tools_task.done():
                reply.tools_task.cancel()
        while not self._out.empty():
            self._out.get_nowait()
        if self._clear_queue is not None:
            self._clear_queue()
        self._play_end = time.perf_counter()

    # Local VAD and playback pacing -------------------------------------------

    def _track_speech(self, audio16: np.ndarray) -> None:
        buf = np.concatenate([self._vad_buf, audio16]) if self._vad_buf.size else audio16
        n = buf.size // WINDOW_SAMPLES
        for i in range(n):
            if self.vad.speech_probability(buf[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]) >= self.backends.vad_threshold:
                self._last_voiced_at = time.perf_counter()
                self._voiced_ms += WINDOW_SAMPLES * 1000 / STT_SAMPLE_RATE
            else:
                self._voiced_ms = 0.0
        self._vad_buf = buf[n * WINDOW_SAMPLES :].copy()
        reply = self._current
        if (
            reply is not None
            and reply.generated
            and self.backends.barge_in
            and self._voiced_ms >= self.backends.barge_in_min_speech_ms
        ):  # the reply stays current until its last queued frame has been played
            self.barge_in()

    async def _wait_for_playback_slot(self) -> None:
        now = time.perf_counter()
        if self._play_end < now:
            self._play_end = now
        if self.pace_output:
            wait = self._play_end - PLAYBACK_LEAD_S - now
            if wait > 0:
                await asyncio.sleep(wait)

    def unplayed_seconds(self) -> float:
        return max(0.0, self._play_end - time.perf_counter()) if self.pace_output else 0.0
