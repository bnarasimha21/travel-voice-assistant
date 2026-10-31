# Architecture

How a spoken question becomes a spoken answer, module by module. Paths are
relative to the repository root.

## Entry point and wiring

`python -m voice_agent` (or `python conv_agent.py`) runs `voice_agent/app.py`:

1. `load_settings()` (`voice_agent/config.py`) reads the process
   environment, then `.env`, then `.env.example`. `Settings` holds every
   tunable; each field maps to an upper-case variable (`stt_model` is
   `STT_MODEL`). `.env.example` carries all defaults, including every URL,
   model name and voice, so the Python code has none of its own.
2. `build_pipeline(settings)` picks the pipeline from `PIPELINE`:
   - `cascade` calls `build_backends` (`voice_agent/backends.py`), which
     builds the speech-to-text, LLM, text-to-speech, VAD and turn detector
     into a `Backends` record, and pre-synthesizes the filler lines.
   - `s2s` calls `build_s2s_backends`, which builds an adapter factory for
     the chosen provider into an `S2SBackends` record.
   A missing key or an unknown backend name raises `ConfigError`, which
   stops the app with the setting to fix.
3. `build_ui` puts a FastRTC `WebRTC` component, a chat panel, a latency
   panel and a Clear button on a Gradio page, with a `VoiceHandler` or an
   `S2SHandler` as the stream handler. FastRTC calls `copy()` on it for every
   browser connection, so each visitor has a handler of their own.

## Shared pieces

- `voice_agent/session.py`: a `SessionStore` of `Session` objects keyed by
  the WebRTC connection id. History changes only through `commit_turn`,
  which adds the caller's line, any tool messages and the answer together.
  Clear conversation calls `SessionStore.reset`, which runs the handler's
  reset callback.
- `voice_agent/tools/`: the `ToolRegistry` with `get_weather`,
  `search_flights` and `search_hotels`. It produces the schemas for each API
  shape (Chat Completions, OpenAI Realtime, Gemini) and runs a call with a
  timeout, returning a JSON result or an error.
- `voice_agent/metrics.py`: a `TurnTimer` per reply records each stage in
  milliseconds after the caller stopped speaking. `finish(outcome)` produces
  the record that goes to `logs/turns.jsonl` and to the latency panel.
- `voice_agent/vad.py`: Silero (default) or an energy gate, both scoring
  32 ms windows of 16 kHz audio.

## Cascade (`PIPELINE=cascade`)

`voice_agent/handler.py`, `VoiceHandler`:

```
receive(frame) -> resample to 16 kHz -> turn source -> TurnEnded
                                                      |
_start_reply -> _reply task: STT -> stream_with_tools -> SentenceChunker
                                                      |
                              _tts_worker -> output queue -> emit (paced)
```

- **Turn source.** By default a `TurnTaker` (`voice_agent/turn.py`) runs the
  VAD in audio time, keeps the current utterance with 300 ms of pre-roll,
  and at each pause asks a `TurnDetector`: `SilenceTurnDetector` ends after
  `SILENCE_MS`; `SmartTurnDetector` asks the Smart Turn model
  (`voice_agent/smart_turn.py`) after `SMART_TURN_PAUSE_MS` and falls back
  to `SMART_TURN_MAX_SILENCE_MS`. With `STT_BACKEND=deepgram-flux` the turn
  source is a `FluxTurnSource` (`voice_agent/deepgram.py`) instead: audio
  streams to Deepgram in 80 ms chunks, and its `EndOfTurn` event arrives
  with the transcript, pushed into `_on_turn_events` from the network
  reader.
- **Speech to text.** `voice_agent/stt.py`: `TransformersWhisperSTT`,
  `FasterWhisperSTT` or the fake. It is skipped when the turn already has a
  transcript (Flux).
- **LLM.** `stream_with_tools` (`voice_agent/llm.py`) streams one model
  round from an OpenAI-compatible endpoint. If the round ends in tool calls,
  they run concurrently through the registry and the model is called again,
  up to `MAX_TOOL_ROUNDS`. A filler line plays if a tool is still running
  after `FILLER_AFTER_MS`.
- **Text to speech.** `SentenceChunker` (`voice_agent/chunker.py`) cuts the
  token stream into speakable chunks; the TTS worker synthesizes each with
  Kokoro (`voice_agent/tts.py`) while the model keeps generating.
- **Playback.** `emit` hands 20 ms frames to FastRTC at real time, so the
  handler always knows how much of the reply the caller has heard.
- **Barge-in.** Speech of at least `BARGE_IN_MIN_SPEECH_MS` during a reply
  cancels the reply task (closing the LLM stream), drops queued audio and
  clears FastRTC's buffer. The words actually heard are committed, marked
  interrupted. If nothing was heard yet, the cut-off turn is merged with the
  next one and recorded as `merged`.

## Speech to speech (`PIPELINE=s2s`)

`voice_agent/realtime/handler.py`, `S2SHandler`, with one adapter per
connection:

```
receive(frame) -> local VAD (timestamps, late barge-in)
               -> adapter.send_audio -> provider WebSocket
provider events -> adapter.events() -> on_event -> output queue -> emit (paced)
```

- **Adapters** (`voice_agent/realtime/openai_realtime.py`,
  `gemini_live.py`) implement the `RealtimeAdapter` protocol in
  `voice_agent/realtime/base.py`: `connect` (session config, instructions,
  tool declarations), `send_audio`, `send_tool_results`, `interrupt` and an
  `events()` stream translated into provider-neutral events: `AudioDelta`,
  `AssistantText`, `UserTranscript`, `UserSpeechStarted`,
  `UserSpeechStopped`, `ResponseDone` (with any tool calls) and
  `AdapterError`.
- **Turns** are decided by the provider. OpenAI reports
  `speech_stopped`, which becomes `turn_end`; Gemini does not, so its
  `turn_end` is null.
- **Tools.** A `ResponseDone` with calls runs them through the same
  `ToolRegistry` and sends the results back; the provider then keeps
  talking. The calls and results are stored in the session history in the
  Chat Completions shape, as in the cascade.
- **Barge-in.** `UserSpeechStarted` during a reply, or the local VAD hearing
  `BARGE_IN_MIN_SPEECH_MS` of speech after the provider finished sending,
  stops playback. The heard share of the reply is committed as interrupted,
  and the adapter's `interrupt` tells OpenAI to truncate the unheard audio
  (Gemini stops on its own when it hears the caller, but after a local barge-in on a reply it had already finished sending, its context still holds the whole reply).
- **Failures.** A dropped or rejected connection shows a notice in the chat
  panel. The next microphone frame reconnects after a short back-off, with
  the conversation so far appended to the instructions.

## Testing and evals

- `voice_agent/fakes.py`: the fake TTS encodes each character as a tone,
  and the fake STT decodes it, so tests can speak exact text with no model
  files.
- `voice_agent/testing/harness.py`: `HandlerDriver` drives either handler
  the way FastRTC does (20 ms frames into `receive`, frames out of `emit`).
- `voice_agent/testing/stub_llm.py`: a local OpenAI-compatible server with a
  scripted travel agent, reached through the real `openai` client.
- `voice_agent/testing/fake_realtime.py`: local WebSocket fakes of OpenAI
  Realtime, Gemini Live and Deepgram Flux that speak their event formats,
  check credentials and answer with the same scripted agent.
- `evals/run.py`: replays the scripted conversations in `evals/scenarios/`
  through a real handler and checks word error rate, tools, facts,
  interruptions, early cutoffs and the TTFA budget. `--matrix` repeats the
  run for each configuration in `evals/matrix.yaml` and prints a comparison
  table.
