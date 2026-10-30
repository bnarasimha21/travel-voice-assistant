"""Turn the date words a model or caller uses into a calendar date."""

from __future__ import annotations

import re
from datetime import date, timedelta

from .base import ToolError

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_date(text: str | None, today: date | None = None) -> date:
    """ISO dates, "today", "tonight", "tomorrow", "day after tomorrow", and weekday names.

    A weekday means its next occurrence, today included ("saturday" on a
    Saturday is today). "next saturday" is the one after that when today is
    already Saturday, otherwise the same as "saturday".
    """
    today = today or date.today()
    raw = (text or "today").strip().lower()
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        pass
    words = re.sub(r"[^a-z ]", " ", raw).split()
    phrase = " ".join(words)
    if phrase in ("", "today", "tonight", "now", "this evening", "this afternoon"):
        return today
    if "day after tomorrow" in phrase:
        return today + timedelta(days=2)
    if "tomorrow" in phrase:
        return today + timedelta(days=1)
    for i, name in enumerate(WEEKDAYS):
        if name in words or name[:3] in words:
            ahead = (i - today.weekday()) % 7
            if ahead == 0 and "next" in words:
                ahead = 7
            return today + timedelta(days=ahead)
    if "weekend" in words:
        return today + timedelta(days=(5 - today.weekday()) % 7)
    raise ToolError(f"could not understand the date {text!r}; use YYYY-MM-DD")
