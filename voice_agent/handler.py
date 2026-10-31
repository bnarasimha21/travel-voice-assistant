"""The FastRTC stream handler: one instance per browser connection.

Audio arrives from the browser over WebRTC in `receive` and goes through VAD
turn taking. Each finished turn starts a reply: STT, then the LLM streams
tokens into a sentence chunker, and a TTS worker synthesizes each chunk while
the LLM keeps generating. Reply audio is handed to FastRTC in `emit`, which
sends it to that browser. Nothing is played on the server.

Barge-in: if the visitor starts talking while a reply is active, the reply
is cancelled (which closes the LLM HTTP stream), queued audio is dropped,
and only the part of the reply the visitor actually heard is stored in the
history, marked as interrupted. `emit` paces audio at real time, so the
handler knows how much has been heard.
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

from .audio import STT_SAMPLE_RATE, StreamResampler, to_mono_float32
from .chunker import SentenceChunker
from .llm import LLMError, TextDelta, ToolFinished, ToolStarted, stream_with_tools
from .metrics import TurnLog, TurnTimer, format_breakdown, ttfa_summary
from .prompts import LLM_FALLBACK_LINE, system_prompt
from .session import Session, SessionStore
from .turn import TurnDetector, TurnEnded, TurnTaker
from .vad import VAD

log = logging.getLogger(__name__)

FRAME_MS = 20
PLAYBACK_LEAD_S = 0.06


@dataclass
class Backends:
    """Shared, already-loaded models and clients. Built once in `app.main()`."""

    stt: Any
    tts: Any
    llm: Any
    store: SessionStore
    make_vad: Callable[[], VAD]
    make_turn_detector: Callable[[], TurnDetector]
    vad_threshold: float = 0.5
    min_speech_ms: float = 250
    max_turn_ms: float = 30000
    barge_in: bool = True
    barge_in_min_speech_ms: float = 300
    tools: Any = None  # tools.ToolRegistry, or None for no tools
    max_tool_rounds: int = 4
    filler_after_ms: float = 300
    filler_cache: dict[str, np.ndarray] = field(default_factory=dict)
    turn_log: TurnLog | None = None  # one JSON line per turn, or None to skip writing
    # Streaming STT with its own turn taking (Deepgram Flux): replaces the local
    # VAD turn taker, and its turns arrive with the transcript already done.
    make_turn_source: Callable[[], Any] | None = None

    async def filler_audio(self, text: str) -> np.ndarray:
        """Filler lines are synthesized once and reused, so they start instantly."""
        audio = self.filler_cache.get(text)
        if audio is None:
            audio = np.asarray(await self.tts.synthesize(text), dtype=np.float32)
            self.filler_cache[text] = audio
        return audio

    async def warm_fillers(self) -> None:
        for tool in getattr(self.tools, "tools", None) or []:
            await self.filler_audio(tool.filler)


@dataclass
class Segment:
    """One synthesized chunk of the reply."""

    text: str
    samples: int
    filler: bool = False  # "let me check" lines are not part of the answer


@dataclass
class Reply:
    """Everything about the reply currently being produced and played."""

    turn: TurnEnded
    user: str = ""
    text: str = ""  # assistant text since the last tool round (the answer to commit)
    apologies: str = ""  # spoken tool-failure apologies, committed ahead of `text`
    generation: int = 0  # session generation when the reply started (see Session.commit_turn)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)  # complete tool rounds
    pending_tools: list[Any] = field(default_factory=list)
    filler_used: bool = False
    segments: list[Segment] = field(default_factory=list)
    emitted: int = 0  # samples handed to the transport
    failed: bool = False
    task: asyncio.Task | None = None
    timer: TurnTimer | None = None
    first_audio_at: float | None = None
    llm_done_at: float | None = None

    def spoken_text(self, heard_samples: int) -> str:
        """The words covered by the first `heard_samples` of reply audio."""
        words: list[str] = []
        start = 0
        for seg in self.segments:
            end = start + seg.samples
            if heard_samples <= start:
                break
            if not seg.filler:
                if heard_samples >= end:
                    words.extend(seg.text.split())
                elif seg.samples > 0:
                    seg_words = seg.text.split()
                    share = (heard_samples - start) / seg.samples
                    words.extend(seg_words[: int(len(seg_words) * share)])
            start = end
        return " ".join(words)


@dataclass
class _Audio:
    samples: np.ndarray
    reply: Reply


@dataclass
class _UI:
    outputs: AdditionalOutputs


@dataclass
class _End:
    reply: Reply


@dataclass
class _Prerendered:
    """A chunk whose audio is already synthesized (filler lines)."""

    text: str
    audio: np.ndarray


class VoiceHandler(AsyncStreamHandler):
    def __init__(self, backends: Backends, session_id: str | None = None, pace_output: bool = True):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=backends.tts.sample_rate,
            input_sample_rate=STT_SAMPLE_RATE,
        )
        self.backends = backends
        self.pace_output = pace_output
        self._session_id = session_id
        self.session: Session | None = None
        if backends.make_turn_source is not None:
            self.turns = backends.make_turn_source()
            self.turns.set_listener(self._on_turn_events)
        else:
            self.turns = TurnTaker(
                vad=backends.make_vad(),
                detector=backends.make_turn_detector(),
                threshold=backends.vad_threshold,
                min_speech_ms=backends.min_speech_ms,
                max_turn_ms=backends.max_turn_ms,
            )
        self._out: asyncio.Queue[_Audio | _UI | _End] = asyncio.Queue()
        self._resampler = StreamResampler(STT_SAMPLE_RATE)
        self._current: Reply | None = None
        self._carry: TurnEnded | None = None
        self._play_end = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self.barge_ins: list[float] = []  # perf_counter timestamps, for tests and logs
        self.turn_records: list[dict[str, Any]] = []  # this connection's latency records

    # FastRTC interface -------------------------------------------------------

    def copy(self) -> "VoiceHandler":
        return VoiceHandler(self.backends, pace_output=self.pace_output)

    async def start_up(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.bind_session()

    def shutdown(self) -> None:
        self._stop_reply()
        if self.session is not None:
            self.backends.store.drop(self.session.id)
        aclose = getattr(self.turns, "aclose", None)
        if aclose is not None:
            self._run_soon(aclose())

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        self.bind_session()
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        sample_rate, data = frame
        audio = to_mono_float32(data)
        audio = self._resampler(audio, sample_rate)
        if self._current is not None and not self.backends.barge_in:
            # Barge-in disabled: ignore the microphone while the agent answers.
            self.turns.reset()
            return
        self._on_turn_events(await self.turns.feed(audio))
        if (
            self._current is not None
            and self.turns.in_speech
            and self.turns.speech_ms >= self.backends.barge_in_min_speech_ms
        ):
            self.barge_in()

    def _on_turn_events(self, events: list) -> None:
        """Turn events from the local turn taker (returned by `feed`) or a
        streaming STT (pushed from its network reader)."""
        for event in events:
            if isinstance(event, TurnEnded):
                if self._current is not None:
                    if not self.backends.barge_in:
                        continue  # a streaming STT finished a turn spoken over the reply
                    if self._current.emitted and event.speech_ms < self.backends.barge_in_min_speech_ms:
                        # Shorter than a barge-in (a cough, an "uh-huh") while the
                        # reply is playing: keep talking. Before any of it plays,
                        # the visitor was still talking, so the turns are merged.
                        log.info("ignoring %.0f ms of speech during the reply", event.speech_ms)
                        continue
                    self.barge_in()
                self._start_reply(event)

    def _run_soon(self, coro) -> None:
        """Run a coroutine on the handler's loop from sync code (any thread)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

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
            return None  # left over from an interrupted reply
        await self._wait_for_playback_slot()
        if item.reply is not self._current:
            return None  # interrupted while waiting: never send it
        self._play_end += item.samples.size / self.output_sample_rate
        reply = item.reply
        if reply.first_audio_at is None:
            reply.first_audio_at = time.perf_counter()
            reply.timer.mark("first_audio_sent", reply.first_audio_at)
            log.info(
                "first audio %.0f ms after end of speech; LLM still generating: %s",
                reply.timer.offset_ms("first_audio_sent"),
                reply.llm_done_at is None,
            )
        reply.emitted += item.samples.size
        return (self.output_sample_rate, item.samples.reshape(1, -1))

    # Session -----------------------------------------------------------------

    def bind_session(self) -> Session:
        if self.session is None:
            sid = self._session_id
            if sid is None:
                ctx = current_context.get()
                sid = ctx.webrtc_id if ctx is not None else uuid.uuid4().hex
            self.session = self.backends.store.get(sid)
            self.session.on_reset(self._on_session_reset)
        return self.session

    def _outputs(self, chat: list) -> AdditionalOutputs:
        """UI update: the chat panel plus the latest latency breakdown."""
        return AdditionalOutputs(chat, self.latency_markdown())

    def latency_markdown(self) -> str:
        shown = [r for r in self.turn_records if r["outcome"] != "empty"]
        return format_breakdown(shown[-1] if shown else None, ttfa_summary(shown))

    def _record_turn(self, reply: Reply, outcome: str) -> None:
        timer = reply.timer
        if timer is None or timer.done:
            return
        record = timer.finish(outcome)
        self.turn_records.append(record)
        if self.backends.turn_log is not None:
            try:
                self.backends.turn_log.write(record)
            except OSError as exc:
                log.warning("could not write turn log: %s", exc)

    def _on_session_reset(self) -> None:
        """Called by Session.reset, possibly from a Gradio worker thread."""
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop:
            self._reset_now()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(self._reset_now)

    def _reset_now(self) -> None:
        self._stop_reply()
        self._carry = None
        self.turns.reset()

    # Reply -------------------------------------------------------------------

    @property
    def is_replying(self) -> bool:
        return self._current is not None

    def _start_reply(self, turn: TurnEnded) -> None:
        if self._carry is not None:
            # The previous turn was cut off before any of its reply was heard:
            # the visitor had not finished, so answer both parts together.
            turn = _merge_turns(self._carry, turn)
            self._carry = None
        session = self.bind_session()
        reply = Reply(turn=turn)
        reply.timer = TurnTimer(
            session_id=session.id,
            turn_index=len(self.turn_records),
            end_of_turn_delay_ms=turn.end_of_turn_delay_ms,
            turn_detector=getattr(self.turns.detector, "name", ""),
            **({"decided_at": turn.decided_at} if turn.decided_at is not None else {}),
        )
        reply.generation = session.generation
        self._current = reply
        reply.task = asyncio.create_task(self._reply(reply))

    async def _reply(self, reply: Reply) -> None:
        """Produce the reply, then always queue its end marker, so a failing
        STT or TTS call cannot leave the handler stuck "replying" forever."""
        try:
            await self._produce(reply)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reply failed for session %s", self.bind_session().id)
            reply.failed = True
        self._out.put_nowait(_End(reply))

    async def _produce(self, reply: Reply) -> None:
        session = self.bind_session()
        turn = reply.turn
        transcript = turn.transcript
        if transcript is None:
            transcript = (await self.backends.stt.transcribe(turn.audio)).strip()
        reply.timer.mark("stt_done")
        if not transcript:
            return
        reply.user = transcript
        self._out.put_nowait(_UI(self._outputs(session.chat_messages(pending_user=transcript))))
        messages = session.llm_messages(system_prompt(tools=bool(self.backends.tools)), transcript)

        chunks: asyncio.Queue[str | _Prerendered | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_worker(reply, chunks))
        filler_task: asyncio.Task | None = None
        try:
            chunker = SentenceChunker()
            spoke_this_round = False
            try:
                events = stream_with_tools(
                    self.backends.llm, messages, self.backends.tools, self.backends.max_tool_rounds
                )
                async with contextlib.aclosing(events) as events:
                    async for event in events:
                        if isinstance(event, TextDelta):
                            reply.text += event.text
                            if event.text.strip():
                                reply.timer.mark("llm_first_token")
                            spoke_this_round = spoke_this_round or bool(event.text.strip())
                            for chunk in chunker.push(event.text):
                                chunks.put_nowait(chunk)
                        elif isinstance(event, ToolStarted):
                            # Say whatever came before the call first; that text is kept
                            # in the tool round's assistant message, not in the answer.
                            for chunk in chunker.flush():
                                chunks.put_nowait(chunk)
                            reply.text = ""
                            reply.pending_tools.append(event.call)
                            reply.timer.mark("tool_start")
                            reply.timer.tools.append(event.call.name)
                            if filler_task is None and not reply.filler_used and not spoke_this_round:
                                filler_task = asyncio.create_task(self._filler_after(reply, chunks, event.call.name))
                            self._show(session, reply)
                        elif isinstance(event, ToolFinished):
                            reply.pending_tools = [c for c in reply.pending_tools if c.id != event.call.id]
                            if not reply.pending_tools and filler_task is not None:
                                filler_task.cancel()
                                filler_task = None
                            if not event.result.ok and not event.result.model_error:
                                # The service failed (timeout, error, unknown place).
                                # Malformed calls are left to the model, which sees
                                # the error and usually retries correctly.
                                tool = self.backends.tools.get(event.call.name)
                                apology = tool.apology if tool else "Sorry, something went wrong with that lookup."
                                for chunk in chunker.flush():
                                    chunks.put_nowait(chunk)
                                chunks.put_nowait(apology)
                                reply.apologies += apology + " "
                            if not event.result.ok:
                                log.warning("tool %s failed: %s", event.call.name, event.result.data.get("error"))
                            if event.messages:
                                reply.timer.mark("tool_end")
                                reply.tool_messages.extend(event.messages)
                                spoke_this_round = False
                                self._show(session, reply)
                for chunk in chunker.flush():
                    chunks.put_nowait(chunk)
                reply.llm_done_at = time.perf_counter()
                reply.timer.mark("llm_done", reply.llm_done_at)
            except LLMError as exc:
                log.warning("LLM call failed for session %s: %s", session.id, exc)
                reply.failed = True
                chunks.put_nowait(LLM_FALLBACK_LINE)
            chunks.put_nowait(None)
            await tts_task
        finally:
            tts_task.cancel()
            if filler_task is not None:
                filler_task.cancel()

    def _show(self, session: Session, reply: Reply) -> None:
        """Update the chat panel with the turn in progress, tool calls included."""
        chat = session.chat_messages(pending_user=reply.user, pending_tools=reply.tool_messages)
        self._out.put_nowait(_UI(self._outputs(chat)))

    async def _filler_after(self, reply: Reply, chunks: asyncio.Queue, tool_name: str) -> None:
        """If the tool is still running after `filler_after_ms`, say a short line."""
        await asyncio.sleep(self.backends.filler_after_ms / 1000)
        tool = self.backends.tools.get(tool_name) if self.backends.tools else None
        text = tool.filler if tool else "One moment."
        audio = await self.backends.filler_audio(text)
        if reply.pending_tools and not reply.filler_used:
            reply.filler_used = True
            chunks.put_nowait(_Prerendered(text, audio))

    async def _tts_worker(self, reply: Reply, chunks: asyncio.Queue) -> None:
        frame = int(self.output_sample_rate * FRAME_MS / 1000)
        while True:
            item = await chunks.get()
            if item is None:
                return
            if isinstance(item, _Prerendered):
                text, audio, filler = item.text, item.audio, True
            else:
                text, filler = item, False
                audio = np.asarray(await self.backends.tts.synthesize(text), dtype=np.float32)
            reply.segments.append(Segment(text, audio.size, filler))
            if audio.size:
                reply.timer.mark("tts_first_chunk")
            for start in range(0, audio.size, frame):
                self._out.put_nowait(_Audio(audio[start : start + frame], reply))

    def _finish_reply(self, reply: Reply) -> AdditionalOutputs | None:
        """The last frame of the reply has been handed out: commit it."""
        if reply is not self._current:
            return None
        self._current = None
        session = self.bind_session()
        if not reply.user:
            self._record_turn(reply, "empty")
            return None
        self._record_turn(reply, "failed" if reply.failed else "completed")
        if not reply.failed:
            answer = (reply.apologies + reply.text).strip()
            session.commit_turn(
                reply.user, answer, tool_messages=reply.tool_messages, generation=reply.generation
            )
        return self._outputs(session.chat_messages())

    def barge_in(self) -> None:
        """The visitor talked over the reply: stop it and keep only what was heard."""
        reply = self._current
        if reply is None:
            return
        self.barge_ins.append(time.perf_counter())
        heard = max(0, reply.emitted - int(round(self.unplayed_seconds() * self.output_sample_rate)))
        self._record_turn(reply, "merged" if heard == 0 else "interrupted")
        self._stop_reply()
        session = self.bind_session()
        if heard == 0:
            # Nothing of the reply was heard: the visitor was still talking.
            self._carry = reply.turn
            log.info("session %s: speech resumed before the reply was heard; merging turns", session.id)
            return
        if reply.user and not reply.failed:
            spoken = reply.spoken_text(heard)
            session.commit_turn(
                reply.user, spoken, interrupted=True, tool_messages=reply.tool_messages, generation=reply.generation
            )
            self._out.put_nowait(_UI(self._outputs(session.chat_messages())))
        log.info("session %s: barge-in after %.2f s of reply audio", session.id, heard / self.output_sample_rate)

    def _stop_reply(self) -> None:
        reply, self._current = self._current, None
        if reply is not None:
            self._record_turn(reply, "cancelled")  # no-op if barge_in already recorded it
            if reply.task is not None and not reply.task.done():
                reply.task.cancel()
        self._drain()
        self._play_end = time.perf_counter()

    def _drain(self) -> None:
        while not self._out.empty():
            self._out.get_nowait()
        if self._clear_queue is not None:
            self._clear_queue()  # also drop frames FastRTC already buffered for sending

    # Playback pacing ---------------------------------------------------------

    async def _wait_for_playback_slot(self) -> None:
        """Hand frames to FastRTC at real time plus a small lead, so the server
        always knows how much of the reply the visitor has actually heard."""
        now = time.perf_counter()
        if self._play_end < now:
            self._play_end = now  # playback was idle or starved: restart the clock
        if self.pace_output:
            wait = self._play_end - PLAYBACK_LEAD_S - now
            if wait > 0:
                await asyncio.sleep(wait)

    def unplayed_seconds(self) -> float:
        """Audio already handed to the transport that has not been heard yet."""
        return max(0.0, self._play_end - time.perf_counter()) if self.pace_output else 0.0


def _merge_turns(first: TurnEnded, second: TurnEnded) -> TurnEnded:
    transcript = None
    if first.transcript is not None or second.transcript is not None:
        transcript = " ".join(t for t in (first.transcript, second.transcript) if t)
    return TurnEnded(
        audio=np.concatenate([first.audio, second.audio]),
        speech_ms=first.speech_ms + second.speech_ms,
        speech_end_at=second.speech_end_at,
        end_of_turn_delay_ms=second.end_of_turn_delay_ms,
        transcript=transcript,
        decided_at=second.decided_at,
    )
