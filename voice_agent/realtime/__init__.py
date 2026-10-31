"""Speech-to-speech mode (`PIPELINE=s2s`): a realtime model hears and speaks directly."""

from .base import RealtimeAdapter, RealtimeToolCall
from .handler import S2SBackends, S2SHandler

__all__ = ["RealtimeAdapter", "RealtimeToolCall", "S2SBackends", "S2SHandler"]
