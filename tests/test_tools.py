"""Tool registry, date parsing and providers. Live providers run against mocked HTTP."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import httpx
import pytest

from voice_agent.config import ConfigError, load_settings
from voice_agent.tools import ToolError, ToolRegistry, build_registry
from voice_agent.tools.base import Tool
from voice_agent.tools.dates import resolve_date
from voice_agent.tools.flights import DuffelFlightProvider, FixtureFlightProvider, flights_tool
from voice_agent.tools.hotels import DuffelStaysProvider, FixtureHotelProvider, hotels_tool
from voice_agent.tools.weather import FixtureWeatherProvider, OpenMeteoProvider, weather_tool

FRIDAY = date(2026, 10, 30)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026-11-03", date(2026, 11, 3)),
        ("today", FRIDAY),
        ("Tomorrow", date(2026, 10, 31)),
        ("day after tomorrow", date(2026, 11, 1)),
        ("saturday", date(2026, 10, 31)),
        ("this Saturday", date(2026, 10, 31)),
        ("friday", FRIDAY),
        ("next friday", date(2026, 11, 6)),
        ("mon", date(2026, 11, 2)),
        ("this weekend", date(2026, 10, 31)),
    ],
)
def test_resolve_date(text, expected):
    assert resolve_date(text, FRIDAY) == expected


def test_resolve_date_rejects_nonsense():
    with pytest.raises(ToolError):
        resolve_date("whenever", FRIDAY)


def registry(*tools: Tool, timeout_s: float = 1.0) -> ToolRegistry:
    return ToolRegistry(list(tools), timeout_s)


async def test_registry_turns_failures_into_error_results():
    async def slow(**_):
        await asyncio.sleep(5)
        return {}

    async def boom(**_):
        raise RuntimeError("provider bug")

    params = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    reg = registry(Tool("slow", "", params, slow), Tool("boom", "", params, boom), timeout_s=0.05)
    assert (await reg.run("nope", "{}")).data == {"error": "unknown tool 'nope'"}
    assert "invalid arguments" in (await reg.run("slow", "{not json")).data["error"]
    assert (await reg.run("slow", "{}")).data == {"error": "missing arguments: x"}
    assert (await reg.run("slow", '{"x": "1"}')).data == {"error": "slow timed out"}
    result = await reg.run("boom", {"x": "1"})
    assert not result.ok and "RuntimeError" in result.data["error"]


async def test_registry_drops_unknown_arguments():
    seen = {}

    async def handler(**kw):
        seen.update(kw)
        return {"ok": True}

    reg = registry(Tool("t", "", {"type": "object", "properties": {"a": {"type": "string"}}}, handler))
    assert (await reg.run("t", '{"a": "1", "evil": 2}')).ok
    assert seen == {"a": "1"}


def test_schemas_for_every_api_shape():
    reg = registry(weather_tool(FixtureWeatherProvider()), flights_tool(FixtureFlightProvider()), hotels_tool(FixtureHotelProvider()))
    assert [s["function"]["name"] for s in reg.openai_schemas()] == ["get_weather", "search_flights", "search_hotels"]
    assert reg.realtime_schemas()[0]["name"] == "get_weather" and "function" not in reg.realtime_schemas()[0]
    assert reg.gemini_declarations()[1]["parameters"]["required"] == ["origin", "destination", "date"]


async def test_fixture_flights_by_code_or_city():
    tool = flights_tool(FixtureFlightProvider(), today=FRIDAY)
    result = await tool.handler(origin="LAX", destination="San Francisco", date="tomorrow")
    assert result["origin"] == "LAX" and result["destination"] == "SFO" and result["date"] == "2026-10-31"
    assert [f["flight_number"] for f in result["flights"]] == ["WN 2287", "AS 1341", "UA 1523"]
    two = await tool.handler(origin="lax", destination="sfo", date="tomorrow", adults=2)
    assert two["flights"][0]["price"] == 196
    empty = await tool.handler(origin="SEA", destination="LAX", date="tomorrow")
    assert empty["flights"] == []
    with pytest.raises(ToolError, match="unknown airport"):
        await tool.handler(origin="Atlantis", destination="LAX", date="tomorrow")
    with pytest.raises(ToolError, match="past"):
        await tool.handler(origin="LAX", destination="SFO", date="2026-01-01")


async def test_fixture_weather_and_hotels():
    weather = weather_tool(FixtureWeatherProvider(), today=FRIDAY)
    w = await weather.handler(location="San Diego, CA", date="saturday")
    assert w["location"] == "San Diego, California" and w["date"] == "2026-10-31" and w["day"] == "Saturday"
    assert isinstance(w["high_f"], int) and w["high_f"] > w["low_f"]
    with pytest.raises(ToolError, match="16 days"):
        await weather.handler(location="San Diego", date="2026-12-31")
    hotels = hotels_tool(FixtureHotelProvider(), today=FRIDAY)
    h = await hotels.handler(city="san diego", check_in="tomorrow", nights=2)
    assert h["check_out"] == "2026-11-02" and h["hotels"][0]["name"] == "Hotel del Coronado"


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_open_meteo_provider_parses_daily_forecast():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "geocoding" in request.url.host:
            assert request.url.params["name"] == "San Diego"
            return httpx.Response(200, json={"results": [{"name": "San Diego", "admin1": "California", "latitude": 32.7, "longitude": -117.2}]})
        p = request.url.params
        assert (p["latitude"], p["start_date"], p["end_date"], p["temperature_unit"]) == ("32.7", "2026-10-31", "2026-10-31", "fahrenheit")
        return httpx.Response(200, json={"daily": {"temperature_2m_max": [72.4], "temperature_2m_min": [60.6], "precipitation_probability_max": [5], "weather_code": [2]}})

    provider = OpenMeteoProvider("https://geocoding-api.open-meteo.com/v1/search", "https://api.open-meteo.com/v1/forecast", mock_client(handler))
    result = await weather_tool(provider, today=FRIDAY).handler(location="San Diego", date="saturday")
    assert result == {
        "location": "San Diego, California",
        "date": "2026-10-31",
        "day": "Saturday",
        "high_f": 72,
        "low_f": 61,
        "conditions": "partly cloudy",
        "precipitation_chance": 5,
        "source": "open-meteo",
    }
    assert len(calls) == 2


async def test_open_meteo_unknown_place():
    provider = OpenMeteoProvider("https://g.test/search", "https://f.test/forecast", mock_client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ToolError, match="could not find"):
        await provider.forecast("Nowhereville", FRIDAY)


async def test_duffel_flights_parses_offers():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/air/offer_requests"
        assert request.headers["Duffel-Version"] == "v2" and request.headers["Authorization"] == "Bearer k"
        body = json.loads(request.content)["data"]
        assert body["slices"] == [{"origin": "LAX", "destination": "SFO", "departure_date": "2026-10-31"}]
        seg = lambda n, dep, arr, o, d: {  # noqa: E731
            "marketing_carrier": {"name": "Duffel Airways", "iata_code": "ZZ"},
            "marketing_carrier_flight_number": n,
            "origin": {"iata_code": o},
            "destination": {"iata_code": d},
            "departing_at": dep,
            "arriving_at": arr,
        }
        offers = [
            {"total_amount": "210.50", "total_currency": "USD", "slices": [{"duration": "PT3H10M", "segments": [seg("1", "2026-10-31T06:00:00", "2026-10-31T07:00:00", "LAX", "SJC"), seg("2", "2026-10-31T08:00:00", "2026-10-31T09:10:00", "SJC", "SFO")]}]},
            {"total_amount": "99.00", "total_currency": "USD", "slices": [{"duration": "PT1H25M", "segments": [seg("7", "2026-10-31T10:00:00", "2026-10-31T11:25:00", "LAX", "SFO")]}]},
            {"total_amount": "1.00"},  # malformed offer is skipped
        ]
        return httpx.Response(201, json={"data": {"offers": offers}})

    provider = DuffelFlightProvider("k", "https://api.duffel.test", mock_client(handler))
    result = await flights_tool(provider, today=FRIDAY).handler(origin="LAX", destination="SFO", date="tomorrow")
    assert result["source"] == "duffel"
    assert result["flights"] == [
        {"airline": "Duffel Airways", "flight_number": "ZZ 7", "origin": "LAX", "destination": "SFO", "depart_time": "10:00", "arrive_time": "11:25", "duration_min": 85, "stops": 0, "price": 99, "currency": "USD"},
        {"airline": "Duffel Airways", "flight_number": "ZZ 1", "origin": "LAX", "destination": "SFO", "depart_time": "06:00", "arrive_time": "09:10", "duration_min": 190, "stops": 1, "price": 210, "currency": "USD"},
    ]


async def test_duffel_http_error_is_a_tool_error():
    provider = DuffelFlightProvider("k", "https://api.duffel.test", mock_client(lambda r: httpx.Response(401, json={})))
    with pytest.raises(ToolError, match="401"):
        await provider.search("LAX", "SFO", FRIDAY, 1, "economy")


async def test_duffel_stays_geocodes_then_searches():
    async def geocode(city):
        return {"latitude": 32.7, "longitude": -117.2}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)["data"]
        assert body["location"]["geographic_coordinates"] == {"latitude": 32.7, "longitude": -117.2}
        assert (body["check_in_date"], body["check_out_date"], len(body["guests"])) == ("2026-10-31", "2026-11-02", 2)
        return httpx.Response(200, json={"data": {"results": [
            {"accommodation": {"name": "Seaside Inn", "rating": 4}, "cheapest_rate_total_amount": "300.00", "cheapest_rate_currency": "USD"},
            {"accommodation": {"name": "Budget Stay", "rating": 2}, "cheapest_rate_total_amount": "120.00", "cheapest_rate_public_currency": "USD"},
        ]}})

    provider = DuffelStaysProvider("k", "https://api.duffel.test", geocode, mock_client(handler))
    result = await hotels_tool(provider, today=FRIDAY).handler(city="San Diego", check_in="tomorrow", nights=2)
    assert [(h["name"], h["price_per_night"]) for h in result["hotels"]] == [("Budget Stay", 60), ("Seaside Inn", 150)]


def test_build_registry_from_settings():
    base = {"WEATHER_PROVIDER": "fixture"}
    assert build_registry(load_settings(env=base, env_files=())).names() == ["get_weather", "search_flights", "search_hotels"]
    assert not build_registry(load_settings(env=dict(base, TOOLS="none"), env_files=()))
    assert build_registry(load_settings(env=dict(base, TOOLS="flights"), env_files=())).names() == ["search_flights"]
    with pytest.raises(ConfigError, match="DUFFEL_API_KEY"):
        build_registry(load_settings(env=dict(base, FLIGHTS_PROVIDER="duffel", DUFFEL_BASE_URL="https://x"), env_files=()))
    with pytest.raises(ConfigError, match="TOOLS"):
        build_registry(load_settings(env=dict(base, TOOLS="taxis"), env_files=()))
