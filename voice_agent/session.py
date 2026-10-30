"""Per-visitor conversation state.

Each WebRTC connection gets its own `Session`, keyed by FastRTC's
`webrtc_id`. History is only ever changed through `commit_turn`, which adds
the user turn and the assistant turn together, so a failed LLM call cannot
leave an orphan user message behind.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

INTERRUPTED_MARK = "[interrupted by the user]"


@dataclass
class Turn:
    user: str
    assistant: str
    interrupted: bool = False
    tool_messages: list[dict[str, Any]] = field(default_factory=list)


class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.created = time.time()
        self.turns: list[Turn] = []
        self._reset_listeners: list[Callable[[], None]] = []
        self._lock = threading.Lock()

    # History -----------------------------------------------------------------

    def history(self) -> list[dict[str, Any]]:
        """Conversation so far in OpenAI message format (without the system prompt)."""
        with self._lock:
            turns = list(self.turns)
        messages: list[dict[str, Any]] = []
        for turn in turns:
            messages.append({"role": "user", "content": turn.user})
            messages.extend(turn.tool_messages)
            content = turn.assistant
            if turn.interrupted:
                content = f"{content} {INTERRUPTED_MARK}".strip()
            messages.append({"role": "assistant", "content": content})
        return messages

    def llm_messages(self, system_prompt: str, user_text: str) -> list[dict[str, Any]]:
        return [{"role": "system", "content": system_prompt}, *self.history(), {"role": "user", "content": user_text}]

    def commit_turn(
        self,
        user: str,
        assistant: str,
        interrupted: bool = False,
        tool_messages: list[dict[str, Any]] | None = None,
    ) -> Turn:
        turn = Turn(user=user, assistant=assistant, interrupted=interrupted, tool_messages=list(tool_messages or []))
        with self._lock:
            self.turns.append(turn)
        return turn

    # Display -----------------------------------------------------------------

    def chat_messages(self, pending_user: str | None = None) -> list[dict[str, Any]]:
        """Messages for the Gradio chatbot (type="messages"), tool activity included.

        `pending_user` shows what the visitor just said before the reply is committed.
        """
        with self._lock:
            turns = list(self.turns)
        out: list[dict[str, Any]] = []
        for turn in turns:
            out.append({"role": "user", "content": turn.user})
            out.extend(_tool_display(turn.tool_messages))
            text = turn.assistant or "(no reply)"
            if turn.interrupted:
                text = f"{text} ... (interrupted)"
            out.append({"role": "assistant", "content": text})
        if pending_user:
            out.append({"role": "user", "content": pending_user})
        return out

    # Reset -------------------------------------------------------------------

    def on_reset(self, callback: Callable[[], None]) -> None:
        self._reset_listeners.append(callback)

    def reset(self) -> None:
        with self._lock:
            self.turns.clear()
        for callback in list(self._reset_listeners):
            callback()


def _tool_display(tool_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render assistant tool calls and tool results as collapsible chat entries."""
    out: list[dict[str, Any]] = []
    results = {m.get("tool_call_id"): m.get("content", "") for m in tool_messages if m.get("role") == "tool"}
    for message in tool_messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            out.append(
                {
                    "role": "assistant",
                    "content": f"{fn.get('name')}({fn.get('arguments', '')})\n\n{results.get(call.get('id'), '')}",
                    "metadata": {"title": f"Tool call: {fn.get('name')}"},
                }
            )
    return out


class SessionStore:
    """All live sessions, keyed by WebRTC connection id."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._sessions[session_id] = Session(session_id)
            return session

    def peek(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> bool:
        session = self.peek(session_id)
        if session is None:
            return False
        session.reset()
        return True

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
