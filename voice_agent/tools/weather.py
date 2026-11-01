"""get_weather: a daily forecast for a place and date.

Live data comes from Open-Meteo (free, no key): the geocoding API finds the
place, the forecast API returns daily highs, lows, rain chance and a WMO
weather code. `FixtureWeatherProvider` serves sample data offline.
"""

from __future__ import annotations

import datetime as dt
from datetime import date
from typing import Any, Protocol

import httpx

from .base import Tool, ToolError
from .dates import resolve_date
from .places import load_fixture, normalise_place

FORECAST_DAYS = 16

WMO_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


def describe_code(code: int | None) -> str:
    return WMO_CODES.get(int(code), "mixed conditions") if code is not None else "unknown conditions"


class WeatherProvider(Protocol):
    async def forecast(self, location: str, day: date) -> dict[str, Any]: ...


def _result(name: str, day: date, high: float, low: float, code: int | None, rain: float | None, source: str) -> dict[str, Any]:
    return {
        "location": name,
        "date": day.isoformat(),
        "day": day.strftime("%A"),
        "high_f": round(high),
        "low_f": round(low),
        "conditions": describe_code(code),
        "precipitation_chance": None if rain is None else round(rain),
        "source": source,
    }


class FixtureWeatherProvider:
    def __init__(self, today: date | None = None):
        self.today = today

    async def forecast(self, location: str, day: date) -> dict[str, Any]:
        place = load_fixture("weather.json")["places"].get(normalise_place(location))
        if place is None:
            raise ToolError(f"no sample weather for {location!r}")
        swing = day.toordinal() % 5 - 2
        rain = 60 if place["code"] >= 51 else 10
        return _result(place["name"], day, place["high_f"] + swing, place["low_f"] + swing, place["code"], rain, "sample data")


class OpenMeteoProvider:
    def __init__(self, geocoding_url: str, forecast_url: str, client: httpx.AsyncClient | None = None, timeout_s: float = 6.0):
        self.geocoding_url = geocoding_url
        self.forecast_url = forecast_url
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def geocode(self, location: str) -> dict[str, Any]:
        name = location.split(",")[0].strip()
        resp = await self._client.get(self.geocoding_url, params={"name": name, "count": 1, "language": "en", "format": "json"})
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            raise ToolError(f"could not find a place called {location!r}")
        return results[0]

    async def forecast(self, location: str, day: date) -> dict[str, Any]:
        place = await self.geocode(location)
        resp = await self._client.get(
            self.forecast_url,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "start_date": day.isoformat(),
                "end_date": day.isoformat(),
            },
        )
        if resp.status_code == 400:
            raise ToolError(resp.json().get("reason", "the forecast service rejected that date"))
        resp.raise_for_status()
        daily = resp.json().get("daily") or {}
        try:
            high = daily["temperature_2m_max"][0]
            low = daily["temperature_2m_min"][0]
        except (KeyError, IndexError) as exc:
            raise ToolError("no forecast for that date") from exc
        if high is None or low is None:
            raise ToolError("no forecast for that date")
        rain = (daily.get("precipitation_probability_max") or [None])[0]
        code = (daily.get("weather_code") or [None])[0]
        label = ", ".join(p for p in (place.get("name"), place.get("admin1")) if p)
        return _result(label, day, high, low, code, rain, "open-meteo")


def weather_tool(provider: WeatherProvider, today: date | None = None) -> Tool:
    async def get_weather(location: str, date: str = "today") -> dict[str, Any]:
        base = today or dt.date.today()
        day = resolve_date(date, base)
        delta = (day - base).days
        if delta < 0:
            raise ToolError("that date is in the past; I can only forecast ahead")
        if delta >= FORECAST_DAYS:
            raise ToolError(f"forecasts only go {FORECAST_DAYS} days ahead")
        return await provider.forecast(location, day)

    return Tool(
        name="get_weather",
        description=(
            "Daily weather forecast (high and low in Fahrenheit, conditions, chance of rain) for a city "
            "on a date up to 16 days ahead. Use whenever the user asks about weather or what to pack."
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, for example 'San Diego'"},
                "date": {"type": "string", "description": "YYYY-MM-DD, or 'today', 'tomorrow' or a weekday name"},
            },
            "required": ["location"],
        },
        handler=get_weather,
        filler="Let me check the forecast.",
        apology="Sorry, I couldn't get the weather just now.",
    )
