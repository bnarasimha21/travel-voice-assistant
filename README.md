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

Each stage can be swapped through settings alone: Whisper through
transformers or faster-whisper, Deepgram Flux as hosted streaming speech
recognition, or a speech-to-speech model (OpenAI Realtime or Gemini Live)
in place of the whole cascade. `python -m evals.run --matrix` runs the same
scripted conversations through each and compares them (see
[Backends](#backends)).

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
  speech followed by `SILENCE_MS` of silence, or, with
  `TURN_DETECTOR=smart`, when the Smart Turn model says a short pause ends
  a finished sentence (see [Turn detection](#turn-detection)). No start or
  stop buttons.
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

For `TURN_DETECTOR=smart`, also download the Smart Turn v3.2 model (about
8 MB, BSD-2-Clause):

```bash
curl -L -o models/smart-turn-v3.2-cpu.onnx https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
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
| `PIPELINE` | `cascade` (default) or `s2s` (see [Backends](#backends)) |
| `STT_BACKEND`, `STT_MODEL`, `STT_DEVICE`, `STT_COMPUTE_TYPE` | `transformers` or `faster-whisper` with a Whisper model, or `deepgram-flux` |
| `DEEPGRAM_API_KEY`, `DEEPGRAM_URL`, `DEEPGRAM_MODEL`, `DEEPGRAM_EOT_THRESHOLD`, `DEEPGRAM_EOT_TIMEOUT_MS` | Deepgram Flux |
| `S2S_PROVIDER` | `openai` or `gemini`, for `PIPELINE=s2s` |
| `OPENAI_API_KEY`, `OPENAI_REALTIME_URL`, `OPENAI_REALTIME_MODEL`, `OPENAI_REALTIME_VOICE`, `OPENAI_REALTIME_TRANSCRIBE_MODEL`, `OPENAI_REALTIME_TURN_DETECTION` | OpenAI Realtime |
| `GEMINI_API_KEY`, `GEMINI_LIVE_URL`, `GEMINI_LIVE_MODEL`, `GEMINI_LIVE_VOICE` | Gemini Live |
| `TTS_BACKEND`, `TTS_MODEL_PATH`, `TTS_VOICES_PATH`, `TTS_VOICE` | Kokoro files and voice |
| `VAD_BACKEND`, `SILENCE_MS`, `MIN_SPEECH_MS` | Turn taking |
| `TURN_DETECTOR` | `silence` (default) or `smart` |
| `SMART_TURN_MODEL_PATH`, `SMART_TURN_PAUSE_MS`, `SMART_TURN_MAX_SILENCE_MS`, `SMART_TURN_THRESHOLD` | Smart Turn model file, when to ask it, the fallback pause, and the "complete" cutoff |
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
| `turn_end` | the turn detector decided you were done (the `SILENCE_MS` pause, or the Smart Turn decision) |
| `stt_done` | transcript ready |
| `llm_first_token` | first text from the model |
| `tool_start`, `tool_end` | first tool call requested, first tool round returned |
| `tts_first_chunk` | first chunk of the reply synthesized |
| `first_audio_sent` | first reply frame handed to WebRTC: time to first audio (TTFA) |
| `llm_done`, `reply_end` | model finished; reply finished or was interrupted |

The record also has the outcome (`completed`, `interrupted`, `merged`,
`failed`, ...), the turn detector, the tools called and `ttfa_ms`. The page
shows the last turn's breakdown and the session's TTFA p50 and p95 under the
microphone.

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

## Turn detection

`TURN_DETECTOR` picks how the agent decides you have finished talking.

- `silence` (the default) waits for `SILENCE_MS` (700 ms) of quiet. It is
  simple and predictable, but it cannot tell "I want to fly to, um..." from
  a finished question, and every turn pays the full 700 ms.
- `smart` runs [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)
  (an 8 MB ONNX model that listens to the last 8 seconds of your turn) after
  every pause of `SMART_TURN_PAUSE_MS` (200 ms). If it says the sentence is
  complete (probability at least `SMART_TURN_THRESHOLD`, 0.5) the turn ends
  there; otherwise the agent keeps listening, asks again at the next pause,
  and gives up waiting after `SMART_TURN_MAX_SILENCE_MS` (2000 ms) of quiet.
  It costs about 20 ms of CPU per question asked
  (`voice_agent/smart_turn.py`, `voice_agent/turn.py`).

When either detector ends a turn too early and you keep talking before any
reply audio has played, the handler drops the half-built reply and merges
both parts into one turn (outcome `merged`), so the answer is right but
later. Only a cutoff after reply audio started is heard by you.

Measured on 2026-10-31 with
`python -m evals.run --llm stub --backends real --turn-detector both --repeats 3`
on the same Apple M5 laptop (CPU only, other jobs running), 30 scripted
turns per detector:

| | `silence` | `smart` |
| --- | ---: | ---: |
| Turn decided (median ms after end of speech) | 704 | 224 |
| TTFA p50 | 2108 ms | 1914 ms |
| TTFA p95 | 2922 ms | 3995 ms |
| Early cutoffs (all recovered by merging) | 3 of 30 (10%) | 9 of 30 (30%) |
| Cutoffs you would hear | 0 | 0 |
| `mid_sentence_pause` TTFA p50 | 2880 ms (cut every time) | 1939 ms (never cut) |

So Smart Turn does what it is for: it waits through "I want to fly to, um"
followed by a one second pause, where the silence detector cut all three
runs, and it answers ordinary questions about half a second sooner. It
also ends turns at pauses between complete sentences that the silence
detector sleeps through. Kokoro reads "Tell me about the best things to do
in Los Angeles, with plenty of detail please" as two sentences with a pause
between them, and the `barge_in` lines were cut six times in three runs;
"Great, thank you. I will think about it and call back." was cut three
times. Those cutoffs were merged, never heard, but each one costs the
time of a reply that was started and thrown away, which is where the worse
p95 comes from. Which detector is better depends on how your callers talk:
`smart` for people who hesitate mid-sentence, `silence` for people who
speak in several complete sentences.

The caller voices are synthetic (Kokoro), which matters here. Kokoro ends
a phrase with falling intonation even when the words are unfinished: Smart
Turn scores "I want to fly from Los Angeles to" (no filler) at 0.76 to 0.99
across six voices, so it would cut that, while "I want to fly to, um" scores
0.01 to 0.02. Real hesitation sounds different and should do better; record
some and put them in `evals/recordings/` to check. A hosted alternative is
Deepgram Flux (`STT_BACKEND=deepgram-flux`, see [Backends](#backends)),
which decides end of turn from the transcript and the audio together.

## Backends

No endpoint, model name or voice is written in the Python code: every
default lives in `.env.example`, and the app starts with those values plus
one API key. A test parses the package and fails if one creeps back in.

### Speech to text

| `STT_BACKEND` | Runs | Turn taking |
| --- | --- | --- |
| `transformers` (default) | Whisper on this machine through Hugging Face transformers | VAD plus `TURN_DETECTOR` |
| `faster-whisper` | Whisper on this machine through CTranslate2 (`STT_COMPUTE_TYPE=int8` on a CPU) | VAD plus `TURN_DETECTOR` |
| `deepgram-flux` | [Deepgram Flux](https://developers.deepgram.com/docs/flux/quickstart), hosted, streaming | Flux decides (`DEEPGRAM_EOT_THRESHOLD`) |

`STT_MODEL=openai/whisper-tiny` runs the whole app on a CPU. The local
backends are free and keep audio on the machine, but they transcribe once
per turn, after the turn has ended, so Whisper's time adds straight to the
reply delay (about 0.3 to 0.6 s for tiny on a laptop CPU). Flux transcribes
while you talk and ends the turn itself from the words and the audio, so
the transcript is ready the moment the turn ends. It costs money per minute
of audio and sends the caller's voice to Deepgram. With Flux the browser
audio streams to Deepgram in 80 ms chunks and `TURN_DETECTOR` is not used.
Barge-in starts at Flux's `StartOfTurn` event, once you have talked for
`BARGE_IN_MIN_SPEECH_MS`.

### Speech to speech

`PIPELINE=s2s` replaces the whole cascade with one realtime model:
`S2S_PROVIDER=openai` ([Realtime API](https://platform.openai.com/docs/guides/realtime-websocket))
or `gemini` ([Live API](https://ai.google.dev/gemini-api/docs/live)). The
browser audio goes straight to the provider over a WebSocket
(`voice_agent/realtime/`), the provider decides turns, and its audio comes
back to the browser through the same FastRTC page. The same three travel
tools are offered as provider function calls and run here, so
`get_weather` works the same way. Talking over the reply stops it, and only
what you heard is kept (OpenAI is told with `conversation.item.truncate`).
The provider stops on its own while it is still generating; once it has
sent the whole reply (it sends faster than real time), the handler's own
VAD catches you talking over the rest of the playback. If the connection
drops, the next audio frame reconnects with the conversation so far in the
instructions. Each browser connection gets its own provider session.

Why the cascade stays the default: a speech-to-speech model has lower
latency and more natural turn taking, because there is no separate
transcribe, think and synthesize hop. But it is billed per minute of audio,
ties you to that provider's voices, and hides the stages: there is no
separate transcript time or first-token time to look at, and Gemini does
not even report when it decided you were done (its `turn_end` is recorded
as null). The cascade lets you change any one part, run it all locally, and
see where the time goes.

LiveKit Agents and Pipecat both support these providers too (see
[Why a custom FastRTC handler](#why-a-custom-fastrtc-handler)).

### Comparison

`python -m evals.run --matrix` runs every scenario once per row of
`evals/matrix.yaml`. Rows whose key is not set are listed as skipped. The
hosted rows feed the caller at real time, since the services decide turns
on the wall clock.

Measured on 2026-10-31 with `python -m evals.run --matrix --llm stub` on an
Apple M5 laptop with 16 GB, CPU only, with other heavy jobs running at the
same time. The stub LLM answers in milliseconds, so TTFA here is the speech
pipeline alone. No Deepgram, OpenAI or Gemini key was set, so those rows did
not run:

| config | passed | pass rate | TTFA p50 ms | TTFA p95 ms | cutoffs (audible) |
| --- | --- | --- | --- | --- | --- |
| whisper-tiny/silence | 7/7 | 100% | 1939 | 3182 | 1 (0) |
| whisper-tiny/smart-turn | 7/7 | 100% | 1511 | 2384 | 3 (0) |
| faster-whisper-tiny/silence | 7/7 | 100% | 1622 | 2486 | 1 (0) |
| deepgram-flux | skipped | - | - | - | needs DEEPGRAM_API_KEY |
| s2s/openai | skipped | - | - | - | needs OPENAI_API_KEY |
| s2s/gemini | skipped | - | - | - | needs GEMINI_API_KEY |

In this run faster-whisper tiny (int8) took about 0.3 s off the median
compared with the same model through transformers. Set the keys and rerun to fill in the
hosted rows. `python -m evals.run --matrix --backends fake` runs the same
matrix fully offline against local fakes of all three services
(`voice_agent/testing/fake_realtime.py`); it checks the wiring, not the
providers' speed.

### Not done: phone calls

Answering a phone number (FastRTC's telephone mode, or Twilio Media
Streams) is a possible follow-up. Both handlers already resample whatever
rate the transport delivers, so the work is in the transport and in testing
8 kHz phone audio against each backend. It is not wired up here.

## Voice evals

```bash
python -m evals.run --llm stub    # offline: stub LLM, Whisper tiny, Kokoro, Silero
python -m evals.run --llm live    # your configured LLM endpoint
python -m evals.run --llm live --judge
python -m evals.run --backends fake   # no model files at all
python -m evals.run --llm stub --turn-detector both   # compare silence and smart
python -m evals.run --matrix      # every backend in evals/matrix.yaml, side by side
python -m evals.run --matrix --backends fake   # the same matrix offline, against local fakes
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

`--turn-detector silence|smart|both` runs every scenario with that
detector (the default is whatever `TURN_DETECTOR` says); `both` runs each
scenario twice and prints a line per detector with its early cutoff rate and
TTFA. A caller line can hold a pause, written `[pause 1.0]` in seconds.
An early cutoff is a reply the agent started before the scripted turn was
over. It is counted, and it fails the scenario only when you would have
heard it (the reply was not merged away), unless the scenario lists that
detector under `allow_early_cutoff`. `mid_sentence_pause` tolerates it for
`silence`, which is the point of that scenario, and for Gemini Live, whose
automatic activity detection is pause-based too.

`--matrix` runs the scenarios once per row of `evals/matrix.yaml` and adds a
comparison table (passed, pass rate, TTFA p50 and p95, cutoffs). The `real`
section is used with the default `--backends real`, the `fake` section with
`--backends fake`. A row lists the settings it changes, the keys it
`requires` (skipped when unset), `realtime: true` to feed caller audio at
real time, and for the fake section a `fake_server` to start. `--stt-model`
overrides `STT_MODEL` for rows that do not set their own.

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

Tests marked `realmodels` use real Whisper (tiny), Kokoro, Silero and the
Smart Turn model and are skipped when the files are not on disk. They include a short stub-mode run
of the eval suite on the real speech models. Run only them with
`pytest -m realmodels`.

`tests/test_backends.py` runs the spoken weather question through every
local speech-to-text backend, and drives Deepgram Flux and both
speech-to-speech providers against local fakes of their WebSocket APIs
(`voice_agent/testing/fake_realtime.py`) that speak the providers' event
formats and check their credentials: turn taking, tool calls, barge-in,
reconnects, wrong keys and the offline `--matrix`. The same tests against the
real services run only when `DEEPGRAM_API_KEY`, `OPENAI_API_KEY` or
`GEMINI_API_KEY` is set. `tests/test_tools_live.py` checks that the real
configured model picks `get_weather` for a weather question; it runs only
with `RUN_LIVE_LLM=1` and LLM credentials set.

## Screenshot

![Screenshot](images/image.png)

## License

MIT. See [LICENSE](LICENSE).
