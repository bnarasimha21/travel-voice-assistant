"""Gradio page and server entry point.

The page holds a FastRTC WebRTC component in send-receive audio mode: the
browser streams the microphone to the server and plays the reply track the
server sends back. Each connection gets its own handler (FastRTC calls
`handler.copy()` per connection) and its own `Session`: a `VoiceHandler` for
the cascade, or an `S2SHandler` for `PIPELINE=s2s`.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import gradio as gr
from fastrtc import WebRTC

from .config import ConfigError, Settings, load_settings
from .handler import Backends, VoiceHandler
from .realtime.handler import S2SBackends, S2SHandler

log = logging.getLogger(__name__)

MAX_SESSIONS = 20


def rtc_configuration(settings: Settings):
    """ICE servers for visitors on other networks. Local and LAN use needs none."""
    kind = settings.turn_server.strip().lower()
    if kind in ("", "none"):
        return None
    import fastrtc

    providers = {
        "hf": fastrtc.get_hf_turn_credentials_async,
        "cloudflare": fastrtc.get_cloudflare_turn_credentials_async,
    }
    if kind == "twilio":
        return fastrtc.get_twilio_turn_credentials()
    if kind not in providers:
        raise ConfigError(f"Unknown TURN_SERVER={settings.turn_server!r} (none, hf, cloudflare or twilio)")
    return providers[kind]


def clear_session(backends: Backends | S2SBackends, webrtc_value: Any) -> list:
    """Reset only the caller's own session."""
    webrtc_id = getattr(webrtc_value, "webrtc_id", webrtc_value)
    if isinstance(webrtc_id, str) and webrtc_id:
        backends.store.reset(webrtc_id)
    return []


def build_ui(backends: Backends | S2SBackends, settings: Settings) -> gr.Blocks:
    handler = S2SHandler(backends) if isinstance(backends, S2SBackends) else VoiceHandler(backends)
    with gr.Blocks(title="Travel Voice Assistant") as demo:
        gr.Markdown(
            "# Travel Voice Assistant\n"
            "Press the microphone, then just talk. Pause when you are done and the "
            "assistant answers in this browser."
        )
        with gr.Row():
            with gr.Column(scale=1):
                webrtc = WebRTC(
                    label="Talk",
                    mode="send-receive",
                    modality="audio",
                    rtc_configuration=rtc_configuration(settings),
                )
                clear = gr.Button("Clear conversation")
                latency = gr.Markdown("Latency appears here after your first turn.", label="Last turn latency")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", label="Conversation", height=480)
        webrtc.stream(
            fn=handler,
            inputs=[webrtc],
            outputs=[webrtc],
            concurrency_limit=MAX_SESSIONS,
            time_limit=None,
        )
        webrtc.on_additional_outputs(
            lambda chat, breakdown: (chat, breakdown), outputs=[chatbot, latency], concurrency_limit=MAX_SESSIONS
        )
        clear.click(lambda value: clear_session(backends, value), inputs=[webrtc], outputs=[chatbot])
    return demo


def build_pipeline(settings: Settings) -> Backends | S2SBackends:
    from .backends import build_backends, build_s2s_backends

    if settings.pipeline == "s2s":
        log.info("Speech to speech: %s", settings.s2s_provider)
        backends = build_s2s_backends(settings)
        if backends.tools:
            log.info("Tools: %s", ", ".join(backends.tools.names()))
        return backends
    if settings.pipeline != "cascade":
        raise ConfigError(f"Unknown PIPELINE={settings.pipeline!r} (expected cascade or s2s)")
    log.info("Loading models (STT=%s %s, TTS=%s, LLM=%s)", settings.stt_backend, settings.stt_model, settings.tts_backend, settings.llm_model)
    backends = build_backends(settings)
    if backends.tools:
        log.info("Tools: %s", ", ".join(backends.tools.names()))
        asyncio.run(backends.warm_fillers())
    return backends


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
        backends = build_pipeline(settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    demo = build_ui(backends, settings)
    demo.launch(server_name=settings.server_name, server_port=settings.server_port, share=settings.share)


if __name__ == "__main__":
    main()
