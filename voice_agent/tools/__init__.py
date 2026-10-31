"""Travel tools the model can call mid-conversation."""

from __future__ import annotations

import datetime as dt

import httpx

from ..config import ConfigError, Settings
from .base import Tool, ToolError, ToolRegistry, ToolResult
from .flights import DuffelFlightProvider, FixtureFlightProvider, flights_tool
from .hotels import DuffelStaysProvider, FixtureHotelProvider, hotels_tool
from .weather import FixtureWeatherProvider, OpenMeteoProvider, weather_tool

__all__ = ["Tool", "ToolError", "ToolRegistry", "ToolResult", "build_registry"]

ALL_TOOLS = ("weather", "flights", "hotels")


def build_registry(settings: Settings, client: httpx.AsyncClient | None = None, today: dt.date | None = None) -> ToolRegistry:
    """Tools named in TOOLS (comma separated, or "none"), with providers from env."""
    wanted = [t.strip().lower() for t in settings.tools.split(",") if t.strip()]
    if wanted in (["none"], ["off"]):
        return ToolRegistry([], settings.tool_timeout_s)
    unknown = set(wanted) - set(ALL_TOOLS)
    if unknown:
        raise ConfigError(f"Unknown TOOLS entries: {', '.join(sorted(unknown))} (expected {', '.join(ALL_TOOLS)} or none)")
    client = client or httpx.AsyncClient(timeout=settings.tool_timeout_s)

    def open_meteo() -> OpenMeteoProvider:
        settings.require("open_meteo_geocoding_url", "open_meteo_forecast_url")
        return OpenMeteoProvider(settings.open_meteo_geocoding_url, settings.open_meteo_forecast_url, client)

    tools: list[Tool] = []
    if "weather" in wanted:
        kind = settings.weather_provider
        if kind == "open-meteo":
            tools.append(weather_tool(open_meteo(), today))
        elif kind == "fixture":
            tools.append(weather_tool(FixtureWeatherProvider(), today))
        else:
            raise ConfigError(f"Unknown WEATHER_PROVIDER={kind!r} (open-meteo or fixture)")
    if "flights" in wanted:
        kind = settings.flights_provider
        if kind == "fixture":
            tools.append(flights_tool(FixtureFlightProvider(), today))
        elif kind == "duffel":
            settings.require("duffel_api_key", "duffel_base_url")
            # Leave Duffel 1.5 s of the tool timeout to send its offers back.
            supplier_ms = int((settings.tool_timeout_s - 1.5) * 1000)
            provider = DuffelFlightProvider(
                settings.duffel_api_key, settings.duffel_base_url, client, supplier_timeout_ms=supplier_ms
            )
            tools.append(flights_tool(provider, today))
        else:
            raise ConfigError(f"Unknown FLIGHTS_PROVIDER={kind!r} (fixture or duffel)")
    if "hotels" in wanted:
        kind = settings.hotels_provider
        if kind == "fixture":
            tools.append(hotels_tool(FixtureHotelProvider(), today))
        elif kind == "duffel":
            settings.require("duffel_api_key", "duffel_base_url")
            tools.append(
                hotels_tool(DuffelStaysProvider(settings.duffel_api_key, settings.duffel_base_url, open_meteo().geocode, client), today)
            )
        else:
            raise ConfigError(f"Unknown HOTELS_PROVIDER={kind!r} (fixture or duffel)")
    return ToolRegistry(tools, settings.tool_timeout_s)
