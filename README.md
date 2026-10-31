# Travel Voice Assistant

A voice assistant for a Southern California travel agency. Open the page,
press the microphone and talk; pause when you are done and it answers out
loud in your browser.

Audio travels between the browser and the server over WebRTC
([FastRTC](https://github.com/gradio-app/fastrtc)). Each turn runs speech to text, an LLM and
text to speech, and the model can look up weather, flights and hotels while
you talk. The defaults are free and run locally: Whisper for speech
recognition and Kokoro for the voice. The LLM is any OpenAI-compatible chat
completions endpoint with tool calling (DigitalOcean serverless inference by
default).

## How it works

```
browser mic --WebRTC--> VoiceHandler.receive --> VAD turn taking
                                                    |
                                       turn ended (you paused)
                                                    v
                         Whisper STT --> LLM --> Kokoro TTS
                                                    |
browser speaker <--WebRTC-- VoiceHandler.emit <------+
```

- Every browser connection gets its own `VoiceHandler` and its own `Session`
  (keyed by the WebRTC connection id), so two visitors never share a
  conversation. "Clear conversation" resets only your session.
- Replies play in the visitor's browser. Nothing plays on the server.
- A turn ends when the voice activity detector (Silero by default) has heard
  speech followed by `SILENCE_MS` of silence. No start or stop buttons.
- History changes only after the LLM call succeeds: the user turn and the
  assistant turn are added together. If the LLM call fails, you hear a short
  apology and the history is left as it was.

### Streaming and barge-in

The LLM reply is streamed. `voice_agent/chunker.py` cuts the token stream
into speakable chunks (a sentence end, a clause boundary once there are
enough words, or a hard word limit; it knows "St.", "Dr.", initials and
"3.5" are not sentence ends), and a TTS worker synthesizes each chunk while
the LLM keeps generating. The first sentence is playing before the model has
finished. The log line `first audio N ms after end of speech; LLM still
generating: True` shows it per turn.

Talk over the agent and it stops: the LLM stream is cancelled (the HTTP
request is closed, so the endpoint stops generating), queued audio is
dropped, and FastRTC's send buffer is cleared. Audio is handed to the
transport at real time, so the server knows how much of the reply you heard;
only those words go into the history, marked `[interrupted by the user]`, so
the model does not assume you heard the rest. If you start talking again
before any of the reply was heard (you had only paused), the two parts are
answered together as one turn.

Barge-in depends on the browser's echo cancellation (on by default) so the
agent does not hear itself; with loud speakers and no echo cancellation, use
headphones or set `BARGE_IN=false`. `BARGE_IN_MIN_SPEECH_MS` sets how long
you must talk before it counts as an interruption.

Kokoro synthesizes one chunk per call. Its `create_stream` helper splits
text into sentences itself, which the chunker already does, so it is not
used.

### Travel tools

The model can call three tools mid-conversation (`voice_agent/tools/`):

| Tool | Default data | Live option |
| --- | --- | --- |
| `get_weather` | [Open-Meteo](https://open-meteo.com) forecast and geocoding (free, no key) | (already live) |
| `search_flights` | Sample schedules in `voice_agent/tools/fixtures/flights.json` | [Duffel](https://duffel.com/docs/api/v2/offer-requests) offer requests, `FLIGHTS_PROVIDER=duffel` |
| `search_hotels` | Sample hotels in `voice_agent/tools/fixtures/hotels.json` | [Duffel Stays](https://duffel.com/docs/guides/getting-started-with-stays) search, `HOTELS_PROVIDER=duffel` |

Live flight and hotel APIs are paid or gated (Amadeus closed its free
self-service API in 2026), so the sample data keeps the demo runnable with
no keys, and the `FlightProvider` and `HotelProvider` interfaces keep the
vendor choice open. Answers built from sample data are labelled
`"source": "sample data"` in the tool result. Set `WEATHER_PROVIDER=fixture`
to run fully offline.

The tool loop (`stream_with_tools` in `voice_agent/llm.py`) streams a model
round; if it ends in tool calls, they run concurrently with a per-call
timeout (`TOOL_TIMEOUT_S`), their results go back as `tool` messages and the
model is called again. If a call is still running after `FILLER_AFTER_MS`
(300 ms) and nothing has been said yet, the agent says a short line such as
"Let me look up flights for you." Those lines are synthesized once and
cached, and fast calls never trigger one. A tool error or timeout produces
a fixed spoken apology, then the model continues with the error in front of
it. Tool calls and results appear in the chat panel as collapsible entries
and stay in the session history, so a follow-up like "book the cheaper one"
has the data.

### Which LLM endpoint (DigitalOcean finding)

The original app called a DigitalOcean Gradient agent endpoint. Its
[Agent Inference API reference](https://docs.digitalocean.com/reference/api/reference/agent-inference/)
lists `tools` and `tool_choice` in the request schema, but nothing in the
docs says tools sent by the caller are run, streamed back as `tool_calls`,
or left for the caller to run. The agent how-to examples send only
`messages`, and agent tools are configured on the DigitalOcean side as
[function routes backed by DigitalOcean Functions](https://docs.digitalocean.com/products/inference/how-to/route-agent-functions/),
whose results come back as `functions.called_functions`. So caller-side tool
calling against an agent endpoint is not documented, and this repo does not
rely on it. (Not tested against a live agent: no key was used while writing
this.)

The default is therefore
[DigitalOcean serverless inference](https://docs.digitalocean.com/products/inference/how-to/use-chat-completions-api/)
at `https://inference.do-ai.run/v1` with a model access key. Its
[API reference](https://docs.digitalocean.com/products/inference/reference/api/serverless-inference/)
documents `tools`, `tool_choice`, streaming and `tool_calls`. The default
model is `openai-gpt-5.6-luna`; any model marked as supporting tool calling
on the
[foundation models page](https://docs.digitalocean.com/products/inference/details/foundation-models/)
works. Any other OpenAI-compatible endpoint with tool calling (OpenAI,
vLLM, Ollama) works the same way. To keep using an agent endpoint, set its
URL, `LLM_EXTRA_QUERY=agent=true` and `TOOLS=none`.

### Why a custom FastRTC handler

FastRTC ships a `ReplyOnPause` handler. This project uses its own
`AsyncStreamHandler` subclass (`voice_agent/handler.py`) instead, because it
needs a pluggable turn detector, needs to know exactly how much of a reply
the visitor has heard (for barge-in), and has to run offline in tests with a
VAD that needs no model download. The page is a small `gr.Blocks` around
FastRTC's `WebRTC` component rather than the stock `Stream.ui`, so the Clear
button can find the caller's own connection id.

LiveKit Agents and Pipecat are fuller frameworks (SFU, telephony, many
integrations), but they would replace the Gradio page and add
infrastructure. FastRTC keeps this repo small. The pipeline modules
(`stt.py`, `llm.py`, `tts.py`, `turn.py`, `session.py`) do not import FastRTC,
so a port stays possible.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/bnarasimha21/travel-voice-assistant.git
cd travel-voice-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in the LLM settings
```

Download the Kokoro voice files from the
[kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)
into `models/`:

```bash
mkdir -p models
curl -L -o models/kokoro-v1.0.int8.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
curl -L -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

The full-precision `kokoro-v1.0.onnx` also works; point `TTS_MODEL_PATH` at
it. If either file is missing the app stops at startup and names the setting
to fix. Whisper and the Silero VAD download from Hugging Face on first run.

## Run

```bash
python conv_agent.py        # or: python -m voice_agent
```

Open http://127.0.0.1:7860, press the microphone and talk. To reach it from
another machine set `SERVER_NAME=0.0.0.0`. Browsers only allow the
microphone on `localhost` or HTTPS, so for other machines use `SHARE=true`
(a Gradio share link) or put it behind HTTPS. Visitors behind strict NATs
need a TURN server: set `TURN_SERVER` to `hf`, `cloudflare` or `twilio` and
the matching credentials (see the
[FastRTC deployment guide](https://github.com/gradio-app/fastrtc/blob/main/docs/deployment.md)).

## Configuration

Everything is read from the environment, then `.env`, then `.env.example`.
See `.env.example` for every setting and its default.

| Setting | What it does |
| --- | --- |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | The OpenAI-compatible endpoint, model and key |
| `LLM_EXTRA_QUERY` | Extra query string for every LLM call, for example `agent=true` |
| `TOOLS`, `TOOL_TIMEOUT_S`, `FILLER_AFTER_MS` | Which tools the model gets, per-call timeout, filler delay |
| `WEATHER_PROVIDER`, `FLIGHTS_PROVIDER`, `HOTELS_PROVIDER`, `DUFFEL_API_KEY` | Tool data sources |
| `STT_BACKEND`, `STT_MODEL` | `transformers` with a Whisper model id |
| `TTS_BACKEND`, `TTS_MODEL_PATH`, `TTS_VOICES_PATH`, `TTS_VOICE` | Kokoro files and voice |
| `VAD_BACKEND`, `SILENCE_MS`, `MIN_SPEECH_MS` | Turn taking |
| `BARGE_IN`, `BARGE_IN_MIN_SPEECH_MS` | Talking over the agent |
| `METRICS_LOG` | Per-turn latency log (`logs/turns.jsonl`), or `none` |
| `SERVER_NAME`, `SERVER_PORT`, `SHARE`, `TURN_SERVER` | Web server |

## Latency budget

Every turn is timed (`voice_agent/metrics.py`) and written as one JSON line
to `logs/turns.jsonl`. Each stage is in milliseconds after you stopped
talking, and every key is present in every line (null when the stage did
not happen):

| Stage | Meaning |
| --- | --- |
| `turn_end` | the turn detector decided you were done (the `SILENCE_MS` pause) |
| `stt_done` | transcript ready |
| `llm_first_token` | first text from the model |
| `tool_start`, `tool_end` | first tool call requested, first tool round returned |
| `tts_first_chunk` | first chunk of the reply synthesized |
| `first_audio_sent` | first reply frame handed to WebRTC: time to first audio (TTFA) |
| `llm_done`, `reply_end` | model finished; reply finished or was interrupted |

The record also has the outcome (`completed`, `interrupted`, `failed`, ...),
the tools called and `ttfa_ms`. The page shows the last turn's breakdown and
the session's TTFA p50 and p95 under the microphone.

Measured with `python -m evals.run --llm stub --repeats 3` (27 turns) on an
Apple M5 laptop with 16 GB, everything on the CPU: Whisper tiny through
transformers, Kokoro int8 through ONNX Runtime, Silero VAD. The stub LLM
answers in a few milliseconds, so these numbers are the speech pipeline
alone; add your endpoint's time to first token. Other heavy jobs were
running on the machine at the time, which is why p95 is well above p50.

| Stage | Median ms after end of speech |
| --- | ---: |
| Turn detected (700 ms pause) | 704 |
| Transcript ready | 1295 |
| LLM first token (stub) | 1303 |
| First audio synthesized | 1996 |
| First audio sent (TTFA) | 1997 |

TTFA p50 is 2.0 s and p95 2.9 s. The pause before the turn ends (0.7 s),
Whisper (about 0.6 s) and synthesizing the first chunk (about 0.7 s) each
take roughly a third. Lowering `SILENCE_MS` is the cheapest win but cuts
people off more often. A GPU droplet has not been measured yet; run the
same command there and compare.

## Voice evals

```bash
python -m evals.run --llm stub    # offline: stub LLM, Whisper tiny, Kokoro, Silero
python -m evals.run --llm live    # your configured LLM endpoint
python -m evals.run --llm live --judge
python -m evals.run --backends fake   # no model files at all
```

Each file in `evals/scenarios/` is a short scripted conversation: the
caller's lines, the tools each answer must call, facts the answer must
contain, an optional interruption point, and a TTFA budget. Caller audio is
synthesized with Kokoro in a different voice and speed per scenario, with
white noise mixed in for `noisy_audio`, and pushed into a real
`VoiceHandler` 20 ms at a time at 48 kHz, as a browser would send it. Tool
data comes from the offline sample providers so answers can be checked.

Per turn the runner checks the transcript's word error rate, the tools
called, the required facts, and that an interrupted reply stopped and was
stored as interrupted. Per scenario it checks that p95 TTFA is within the
budget (`--budget-scale` multiplies all budgets; CI uses 2.5 because hosted
runners are slower). It prints a table, saves JSON to `evals/results/`, and
exits non-zero if anything failed. By default caller audio is pushed faster
than real time; pass `--realtime` to feed it at real time instead. TTFA
counts the end-of-turn pause either way.

Synthetic callers are cleaner and more cooperative than real people: they
do not hesitate, mumble or talk over background noise. Put real recordings
in `evals/recordings/` (gitignored) and point a turn at one with
`audio: recordings/clip.wav`, keeping `say` as the reference transcript.
Scenarios whose recordings are missing are skipped.

`evals/judge.py` is optional and only runs with `--llm live --judge`: it
asks the model to grade each completed answer for helpfulness and spoken
length. The deterministic checks never depend on it.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The default run is offline: no API keys and no model downloads. It uses
deterministic fake speech models (`voice_agent/fakes.py`: the fake voice
encodes each character as a tone and the fake recogniser decodes it back),
an energy VAD, and a local OpenAI-compatible stub server
(`voice_agent/testing/stub_llm.py`) reached through the real `openai`
client. The tests drive the real `VoiceHandler` the way FastRTC does,
pushing 20 ms microphone frames into `receive` and pulling reply audio out of
`emit` (`voice_agent/testing/harness.py`).

Tests marked `realmodels` use real Whisper (tiny), Kokoro and Silero and are
skipped when the files are not on disk. They include a short stub-mode run
of the eval suite on the real speech models. Run only them with
`pytest -m realmodels`. `tests/test_tools_live.py` checks that the real
configured model picks `get_weather` for a weather question; it runs only
with `RUN_LIVE_LLM=1` and LLM credentials set.

## Screenshot

![Screenshot](images/image.png)

## License

MIT. See [LICENSE](LICENSE).
