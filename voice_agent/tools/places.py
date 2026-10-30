"""Place names: fixture lookups shared by the offline providers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"


@lru_cache(maxsize=None)
def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def normalise_place(text: str) -> str:
    """'San Diego, CA' -> 'san diego'."""
    text = text.split(",")[0].lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text)).strip()


def airport_code(place: str) -> str | None:
    """An IATA code for a code or a city we know, else None."""
    raw = place.strip()
    if re.fullmatch(r"[A-Za-z]{3}", raw) and raw.upper() in load_fixture("airports.json")["codes"]:
        return raw.upper()
    return load_fixture("airports.json")["cities"].get(normalise_place(raw))
