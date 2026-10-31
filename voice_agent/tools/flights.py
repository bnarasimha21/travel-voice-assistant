"""search_flights: one-way flight options for a route and date.

`FixtureFlightProvider` (the default) serves sample schedules so the demo
runs with no keys. `DuffelFlightProvider` searches live offers through the
Duffel API (needs DUFFEL_API_KEY; test-mode keys return Duffel's sandbox
airline).
"""

from __future__ import annotations

import datetime as dt
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .base import Tool, ToolError
from .dates import resolve_date
from .places import airport_code, load_fixture

MAX_RESULTS = 3


class FlightProvider(ABC):
    source = "base"
    any_airport_code = False  # True: accept any IATA code, not only the sample list

    @abstractmethod
    async def search(self, origin: str, destination: str, day: dt.date, adults: int, cabin: str) -> list[dict[str, Any]]:
        """Flights sorted cheapest first, at most MAX_RESULTS."""


class FixtureFlightProvider(FlightProvider):
    source = "sample data"

    async def search(self, origin, destination, day, adults, cabin):
        flights = load_fixture("flights.json")["routes"].get(f"{origin}-{destination}", [])
        out = [dict(f, origin=origin, destination=destination, price=f["price"] * adults) for f in flights]
        return sorted(out, key=lambda f: f["price"])[:MAX_RESULTS]


def _iso_duration_minutes(text: str | None) -> int | None:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?", text or "")
    if not m:
        return None
    d, h, mins = (int(x) if x else 0 for x in m.groups())
    return d * 1440 + h * 60 + mins


class DuffelFlightProvider(FlightProvider):
    source = "duffel"
    any_airport_code = True

    def __init__(
        self,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 20.0,
        supplier_timeout_ms: int = 10000,
    ):
        if not api_key:
            raise ToolError("DUFFEL_API_KEY is not set")
        # Duffel accepts 2 to 60 s. It must end before the tool timeout, so a
        # slow search returns the offers found so far instead of timing out.
        self.supplier_timeout_ms = max(2000, min(60000, int(supplier_timeout_ms)))
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Duffel-Version": "v2", "Accept": "application/json"}
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def search(self, origin, destination, day, adults, cabin):
        body = {
            "data": {
                "slices": [{"origin": origin, "destination": destination, "departure_date": day.isoformat()}],
                "passengers": [{"type": "adult"} for _ in range(adults)],
                "cabin_class": cabin,
            }
        }
        resp = await self._client.post(
            f"{self.base_url}/air/offer_requests",
            params={"return_offers": "true", "supplier_timeout": self.supplier_timeout_ms},
            json=body,
            headers=self._headers,
        )
        if resp.status_code >= 400:
            raise ToolError(f"flight search failed with HTTP {resp.status_code}")
        offers = (resp.json().get("data") or {}).get("offers") or []
        flights = []
        for offer in offers:
            try:
                sl = offer["slices"][0]
                segs = sl["segments"]
                first, last = segs[0], segs[-1]
                carrier = first.get("marketing_carrier") or {}
                flights.append(
                    {
                        "airline": carrier.get("name") or (offer.get("owner") or {}).get("name", "unknown airline"),
                        "flight_number": f"{carrier.get('iata_code', '')} {first.get('marketing_carrier_flight_number', '')}".strip(),
                        "origin": first["origin"]["iata_code"],
                        "destination": last["destination"]["iata_code"],
                        "depart_time": first["departing_at"][11:16],
                        "arrive_time": last["arriving_at"][11:16],
                        "duration_min": _iso_duration_minutes(sl.get("duration")),
                        "stops": len(segs) - 1,
                        "price": round(float(offer["total_amount"])),
                        "currency": offer["total_currency"],
                    }
                )
            except (KeyError, IndexError, TypeError, ValueError):
                continue  # skip offers missing fields rather than failing the search
        return sorted(flights, key=lambda f: f["price"])[:MAX_RESULTS]


CABINS = {"economy", "premium_economy", "business", "first"}


def flights_tool(provider: FlightProvider, today: dt.date | None = None) -> Tool:
    async def search_flights(origin: str, destination: str, date: str = "today", adults: int = 1, cabin: str = "economy") -> dict[str, Any]:
        base = today or dt.date.today()
        day = resolve_date(date, base)
        if day < base:
            raise ToolError("that date is in the past")
        any_code = provider.any_airport_code
        o, d = airport_code(origin, any_code), airport_code(destination, any_code)
        if o is None:
            raise ToolError(f"unknown airport or city {origin!r}; use an IATA code like LAX")
        if d is None:
            raise ToolError(f"unknown airport or city {destination!r}; use an IATA code like SFO")
        if o == d:
            raise ToolError("origin and destination are the same")
        adults = max(1, min(int(adults), 9))
        cabin = cabin if cabin in CABINS else "economy"
        flights = await provider.search(o, d, day, adults, cabin)
        return {
            "origin": o,
            "destination": d,
            "date": day.isoformat(),
            "day": day.strftime("%A"),
            "adults": adults,
            "flights": flights,
            "source": provider.source,
        }

    return Tool(
        name="search_flights",
        description=(
            "Search one-way flights between two airports on a date. Returns up to three options, "
            "cheapest first, with airline, flight number, local departure and arrival times and total price."
        ),
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure airport IATA code (LAX) or city name"},
                "destination": {"type": "string", "description": "Arrival airport IATA code (SFO) or city name"},
                "date": {"type": "string", "description": "YYYY-MM-DD, or 'today', 'tomorrow' or a weekday name"},
                "adults": {"type": "integer", "description": "Number of adult passengers", "minimum": 1, "maximum": 9},
                "cabin": {"type": "string", "enum": sorted(CABINS)},
            },
            "required": ["origin", "destination", "date"],
        },
        handler=search_flights,
        filler="Let me look up flights for you.",
        apology="Sorry, I couldn't search flights just now.",
    )
