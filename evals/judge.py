"""Optional LLM grader for live eval runs (`python -m evals.run --llm live --judge`).

The deterministic checks in `evals/run.py` (WER, tool calls, facts,
interruptions, latency) never depend on this. The judge only adds a score for
what those cannot measure: whether a spoken answer is helpful and short
enough to listen to. Interrupted replies are not graded.
"""

from __future__ import annotations

import json
import re
from typing import Any

from voice_agent.backends import build_llm
from voice_agent.config import Settings
from voice_agent.llm import LLMError

JUDGE_PROMPT = """You grade replies from a voice travel assistant. The reply is spoken aloud, so
it should be short (one to three sentences), direct, and free of lists, links or markdown.

Caller said: {user}
Assistant replied: {answer}

Return only JSON: {{"helpful": <1-5>, "length_ok": <true|false>, "reason": "<one sentence>"}}"""


def parse_verdict(text: str) -> dict[str, Any]:
    """Pull the JSON verdict out of the grader's reply; raises ValueError if it is not there."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge reply has no JSON: {text!r}")
    data = json.loads(match.group(0))
    helpful = int(data["helpful"])
    if not 1 <= helpful <= 5:
        raise ValueError(f"helpful score out of range: {helpful}")
    return {"helpful": helpful, "length_ok": bool(data["length_ok"]), "reason": str(data.get("reason", ""))}


async def judge_turn(llm: Any, user: str, answer: str) -> dict[str, Any]:
    reply = await llm.complete([{"role": "user", "content": JUDGE_PROMPT.format(user=user, answer=answer)}])
    return parse_verdict(reply)


async def judge_report(results: list[Any], settings: Settings, llm: Any = None, min_helpful: int = 3) -> None:
    """Grade every completed turn in place. A low score or a too-long answer fails the scenario."""
    llm = llm or build_llm(settings)
    for scenario in results:
        if scenario.status == "skip":
            continue
        for i, turn in enumerate(scenario.turns, 1):
            if turn.outcome != "completed":
                continue
            try:
                verdict = await judge_turn(llm, turn.heard, turn.answer)
            except (LLMError, ValueError, KeyError, json.JSONDecodeError) as exc:
                verdict = {"error": str(exc)}
                scenario.failures.append(f"turn {i}: judge failed: {exc}")
            else:
                if verdict["helpful"] < min_helpful or not verdict["length_ok"]:
                    scenario.failures.append(f"turn {i}: judge {verdict}")
            turn.judge = verdict
        if scenario.failures:
            scenario.status = "fail"
