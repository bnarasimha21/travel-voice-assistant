"""Per-turn latency: where the time goes between the caller pausing and hearing a reply.

A `TurnTimer` is attached to every reply. The handler marks each stage the
first time it happens, and when the reply ends (finished, interrupted,
failed) the timer becomes one JSON line in `logs/turns.jsonl`.

Every stage is an offset in milliseconds from the end of the caller's speech.
The turn detector works in audio time, so the end of speech is anchored like
this: the turn was decided `end_of_turn_delay_ms` of audio after speech ended,
and every later stage is that delay plus the wall-clock time since the
decision. When audio arrives in real time (a browser) this equals plain wall
clock; when an eval pushes audio faster than real time it still counts the
pause a real caller would have waited.

The headline number is time to first audio (TTFA): end of speech to the first
reply audio frame handed to the WebRTC transport.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Stage keys, in pipeline order. All are present in every record (null when
# the stage did not happen, e.g. no tool call or the caller interrupted first).
STAGES = (
    "speech_end",  # last voiced audio from the caller (always 0)
    "turn_end",  # the turn detector decided the caller was done
    "stt_done",  # transcript ready
    "llm_first_token",  # first non-empty text from the LLM (any round)
    "tool_start",  # first tool call requested by the LLM
    "tool_end",  # first tool round finished (all its calls returned)
    "tts_first_chunk",  # first chunk of reply audio synthesized
    "first_audio_sent",  # first reply audio frame handed to the transport (TTFA)
    "llm_done",  # LLM stream finished
    "reply_end",  # reply finished, interrupted or abandoned
)

OUTCOMES = ("completed", "interrupted", "merged", "failed", "empty", "cancelled")


@dataclass
class TurnTimer:
    session_id: str
    turn_index: int
    end_of_turn_delay_ms: float
    decided_at: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    outcome: str | None = None
    turn_detector: str = ""
    record: dict[str, Any] | None = None

    def mark(self, stage: str, at: float | None = None) -> None:
        """Record a stage the first time it happens; later calls are ignored."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        if stage not in self.marks:
            self.marks[stage] = time.perf_counter() if at is None else at

    def offset_ms(self, stage: str) -> float | None:
        if stage == "speech_end":
            return 0.0
        if stage == "turn_end":
            return round(self.end_of_turn_delay_ms, 1)
        at = self.marks.get(stage)
        if at is None:
            return None
        return round(self.end_of_turn_delay_ms + (at - self.decided_at) * 1000, 1)

    @property
    def done(self) -> bool:
        return self.record is not None

    def finish(self, outcome: str) -> dict[str, Any]:
        """Close the turn once; returns the JSON-ready record."""
        if self.record is not None:
            return self.record
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        self.mark("reply_end")
        self.outcome = outcome
        stages = {s: self.offset_ms(s) for s in STAGES}
        self.record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session": self.session_id,
            "turn": self.turn_index,
            "outcome": outcome,
            "interrupted": outcome in ("interrupted", "merged"),
            "turn_detector": self.turn_detector,
            "tools": list(self.tools),
            "ttfa_ms": stages["first_audio_sent"],
            "stages_ms": stages,
        }
        return self.record


class TurnLog:
    """Append-only JSON lines file, safe to share between sessions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def percentile(values: Iterable[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100). None for an empty input."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    rank = max(1, math.ceil(q / 100 * len(data)))
    return data[min(rank, len(data)) - 1]


def ttfa_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """p50 and p95 time to first audio over the turns that produced audio."""
    values = [r["ttfa_ms"] for r in records if r.get("ttfa_ms") is not None]
    return {"turns": len(values), "p50_ms": percentile(values, 50), "p95_ms": percentile(values, 95)}


_LABELS = {
    "turn_end": "Turn detected",
    "stt_done": "Transcript ready",
    "llm_first_token": "LLM first token",
    "tool_start": "Tool call started",
    "tool_end": "Tool results back",
    "tts_first_chunk": "First audio synthesized",
    "first_audio_sent": "First audio sent (TTFA)",
    "llm_done": "LLM finished",
    "reply_end": "Reply ended",
}


def format_breakdown(record: dict[str, Any] | None, summary: dict[str, Any] | None = None) -> str:
    """Markdown for the UI: the last turn's stages, plus p50/p95 for this session."""
    if record is None:
        return "Latency appears here after your first turn."
    lines = ["| Stage | ms after you stopped talking |", "| --- | ---: |"]
    for stage, label in _LABELS.items():
        value = record["stages_ms"].get(stage)
        if value is not None:
            lines.append(f"| {label} | {value:.0f} |")
    notes = [f"Outcome: {record['outcome']}"]
    if record["tools"]:
        notes.append("tools: " + ", ".join(record["tools"]))
    text = "\n".join(lines) + "\n\n" + "; ".join(notes) + "."
    if summary and summary.get("turns"):
        text += f"\n\nThis session: TTFA p50 {summary['p50_ms']:.0f} ms, p95 {summary['p95_ms']:.0f} ms over {summary['turns']} turns."
    return text
