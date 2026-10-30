"""Prompt text and fixed spoken lines."""

from __future__ import annotations

from datetime import date

PERSONA = (
    "You are a Southern California Travel Agency Assistant. Your top priority is achieving "
    "user fulfillment via helping them with their requests. Your replies are spoken aloud by "
    "a text to speech voice, so write plain sentences: no markdown, no lists, no emoji, no URLs. "
    "Limit all responses to 2 concise sentences and no more than 100 words."
)

LLM_FALLBACK_LINE = "Sorry, I'm having trouble reaching my travel desk right now. Please try asking again in a moment."


def system_prompt(today: date | None = None) -> str:
    today = today or date.today()
    return f"{PERSONA}\n\nToday is {today.strftime('%A')}, {today.isoformat()}."
