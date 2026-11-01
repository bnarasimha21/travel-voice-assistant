"""search_hotels: places to stay in a city for given dates.

`FixtureHotelProvider` (the default) serves sample hotels offline.
`DuffelStaysProvider` searches live rates through Duffel Stays, which needs
coordinates, so it geocodes the city with Open-Meteo first.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

import httpx

from .base import Tool, ToolError
from .dates import resolve_date
from .places import load_fixture, normalise_place

MAX_RESULTS = 3


class HotelProvider(ABC):
    source = "base"

    @abstractmethod
    async def search(self, city: str, check_in: dt.date, check_out: dt.date, guests: int) -> list[dict[str, Any]]:
        """Hotels with a per-night price, at most MAX_RESULTS."""


class FixtureHotelProvider(HotelProvider):
    source = "sample data"

    async def search(self, city, check_in, check_out, guests):
        hotels = load_fixture("hotels.json")["cities"].get(normalise_place(city))
        if hotels is None:
            raise ToolError(f"no sample hotels for {city!r}")
        return [dict(h) for h in sorted(hotels, key=lambda h: -h["rating"])[:MAX_RESULTS]]


Geocoder = Callable[[str], Awaitable[dict[str, Any]]]


class DuffelStaysProvider(HotelProvider):
    source = "duffel"

    def __init__(self, api_key: str, base_url: str, geocode: Geocoder, client: httpx.AsyncClient | None = None, radius_km: int = 10, timeout_s: float = 20.0):
        if not api_key:
            raise ToolError("DUFFEL_API_KEY is not set")
        self.base_url = base_url.rstrip("/")
        self.geocode = geocode
        self.radius_km = radius_km
        self._headers = {"Authorization": f"Bearer {api_key}", "Duffel-Version": "v2", "Accept": "application/json"}
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def search(self, city, check_in, check_out, guests):
        place = await self.geocode(city)
        body = {
            "data": {
                "rooms": 1,
                "location": {
                    "radius": self.radius_km,
                    "geographic_coordinates": {"latitude": place["latitude"], "longitude": place["longitude"]},
                },
                "check_in_date": check_in.isoformat(),
                "check_out_date": check_out.isoformat(),
                "guests": [{"type": "adult"} for _ in range(guests)],
            }
        }
        resp = await self._client.post(f"{self.base_url}/stays/search", json=body, headers=self._headers)
        if resp.status_code >= 400:
            raise ToolError(f"hotel search failed with HTTP {resp.status_code}")
        nights = max(1, (check_out - check_in).days)
        hotels = []
        for r in (resp.json().get("data") or {}).get("results") or []:
            try:
                acc = r["accommodation"]
                total = float(r["cheapest_rate_total_amount"])
                currency = r.get("cheapest_rate_currency") or r.get("cheapest_rate_public_currency") or "USD"
                hotels.append(
                    {
                        "name": acc["name"],
                        "area": ((acc.get("location") or {}).get("address") or {}).get("city_name", city),
                        "price_per_night": round(total / nights),
                        "currency": currency,
                        "rating": acc.get("rating"),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(hotels, key=lambda h: h["price_per_night"])[:MAX_RESULTS]


def hotels_tool(provider: HotelProvider, today: dt.date | None = None) -> Tool:
    async def search_hotels(city: str, check_in: str = "today", nights: int = 1, guests: int = 2) -> dict[str, Any]:
        base = today or dt.date.today()
        start = resolve_date(check_in, base)
        if start < base:
            raise ToolError("that check-in date is in the past")
        nights = max(1, min(int(nights), 30))
        guests = max(1, min(int(guests), 8))
        end = start + dt.timedelta(days=nights)
        hotels = await provider.search(city, start, end, guests)
        return {
            "city": city,
            "check_in": start.isoformat(),
            "check_out": end.isoformat(),
            "nights": nights,
            "guests": guests,
            "hotels": hotels,
            "source": provider.source,
        }

    return Tool(
        name="search_hotels",
        description="Find hotels in a city for a check-in date and number of nights. Returns up to three with nightly price and star rating.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, for example 'San Diego'"},
                "check_in": {"type": "string", "description": "YYYY-MM-DD, or 'today', 'tomorrow' or a weekday name"},
                "nights": {"type": "integer", "minimum": 1, "maximum": 30},
                "guests": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "required": ["city", "check_in"],
        },
        handler=search_hotels,
        filler="Let me see which hotels have rooms.",
        apology="Sorry, I couldn't search hotels just now.",
    )
