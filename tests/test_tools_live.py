"""Optional: the real configured model picks get_weather for a weather question.

Skipped unless RUN_LIVE_LLM=1 and LLM_BASE_URL, LLM_MODEL and LLM_API_KEY are
set (in the environment or .env). Costs one small request.
"""

from __future__ import annotations

import os

import pytest

from voice_agent.backends import build_llm
from voice_agent.config import load_settings
from voice_agent.llm import ToolCalls
from voice_agent.prompts import system_prompt
from voice_agent.tools import build_registry

pytestmark = pytest.mark.live


@pytest.fixture
def live_settings():
    if os.environ.get("RUN_LIVE_LLM") != "1":
        pytest.skip("set RUN_LIVE_LLM=1 to call the real LLM")
    settings = load_settings(weather_provider="fixture")
    if not (settings.llm_base_url and settings.llm_model and settings.llm_api_key):
        pytest.skip("LLM_BASE_URL, LLM_MODEL and LLM_API_KEY are needed")
    return settings


async def test_live_model_chooses_get_weather(live_settings):
    llm = build_llm(live_settings)
    registry = build_registry(live_settings)
    messages = [
        {"role": "system", "content": system_prompt(tools=True)},
        {"role": "user", "content": "What's the weather in San Diego this Saturday?"},
    ]
    calls = None
    async for event in llm.stream(messages, registry.openai_schemas()):
        if isinstance(event, ToolCalls):
            calls = event.calls
    assert calls, "the model answered without calling a tool"
    assert calls[0].name == "get_weather"
    assert "san diego" in calls[0].parsed_arguments()["location"].lower()
