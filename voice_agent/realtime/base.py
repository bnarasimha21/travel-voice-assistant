"""The contract between `S2SHandler` and a speech-to-speech provider adapter.

An adapter owns one WebSocket session with a realtime model. The handler
pushes microphone audio in with `send_audio`, and reads provider-neutral
events from `events()`. Tool calls come back as `ResponseDone(calls=...)`;
the handler runs them with the same `ToolRegistry` the cascade uses and
answers with `send_tool_results`, after which the model keeps talking.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Sequence

import numpy as np

from ..audio import float_to_int16


@dataclass
class RealtimeToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as the model produced it

    def as_chat_call(self) -> dict[str, Any]:
        """The same call as a Chat Completions `tool_calls` entry, for the session history."""
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class AudioDelta:
    audio: np.ndarray  # float32 mono at the adapter's output_sample_rate
    response_id: str


@dataclass
class AssistantText:
    """Transcript of what the model is saying (arrives alongside its audio)."""

    text: str
    response_id: str


@dataclass
class UserTranscript:
    """What the provider heard the caller say. `final` text replaces earlier pieces.

    `item_id` names the caller's conversation item when the provider has one
    (OpenAI does, Gemini does not); a transcript can arrive after the reply."""

    text: str
    final: bool
    item_id: str = ""


@dataclass
class UserItemCommitted:
    """The provider stored a stretch of the caller's audio as a conversation
    item; its transcript, if any, arrives later under the same `item_id`."""

    item_id: str


@dataclass
class UserSpeechStarted:
    """The provider detected the caller talking (over a reply this means barge-in)."""


@dataclass
class UserSpeechStopped:
    """The provider decided the caller finished (OpenAI reports this; Gemini does not)."""


@dataclass
class ResponseDone:
    """The model finished a response. With `calls` it is waiting for tool results."""

    response_id: str
    calls: list[RealtimeToolCall] = field(default_factory=list)
    cancelled: bool = False


@dataclass
class AdapterError:
    message: str
    fatal: bool = False  # the session is gone and must be reopened


RealtimeEvent = AudioDelta | AssistantText | UserTranscript | UserItemCommitted | UserSpeechStarted | UserSpeechStopped | ResponseDone | AdapterError


class RealtimeAdapter(Protocol):
    name: str
    input_sample_rate: int
    output_sample_rate: int
    reports_speech_stopped: bool

    async def connect(self, instructions: str, tools: Any) -> None: ...

    async def send_audio(self, audio: np.ndarray) -> None: ...

    async def send_tool_results(self, results: list[tuple[RealtimeToolCall, str]]) -> None: ...

    async def interrupt(self, response_id: str, played_ms: int, unheard: Sequence[str] = (), active: str = "") -> None: ...

    def events(self) -> AsyncIterator[RealtimeEvent]: ...

    async def aclose(self) -> None: ...


def pcm16_b64(audio: np.ndarray) -> str:
    return base64.b64encode(float_to_int16(audio).tobytes()).decode("ascii")


def b64_pcm16(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    return (np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0).copy()


def dumps(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"))
