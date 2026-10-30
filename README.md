# Travel Voice Assistant

A voice assistant for a Southern California travel agency. Open the page,
press the microphone and talk; pause when you are done and it answers out
loud in your browser.

Audio travels between the browser and the server over WebRTC
([FastRTC](https://fastrtc.org)). Each turn runs speech to text, an LLM and
text to speech. The defaults are free and run locally: Whisper for speech
recognition and Kokoro for the voice. The LLM is any OpenAI-compatible chat
completions endpoint.

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
[FastRTC deployment guide](https://fastrtc.org/deployment/)).

## Configuration

Everything is read from the environment, then `.env`, then `.env.example`.
See `.env.example` for every setting and its default.

| Setting | What it does |
| --- | --- |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | The OpenAI-compatible endpoint, model and key |
| `LLM_EXTRA_QUERY` | Extra query string for every LLM call, for example `agent=true` |
| `STT_BACKEND`, `STT_MODEL` | `transformers` with a Whisper model id |
| `TTS_BACKEND`, `TTS_MODEL_PATH`, `TTS_VOICES_PATH`, `TTS_VOICE` | Kokoro files and voice |
| `VAD_BACKEND`, `SILENCE_MS`, `MIN_SPEECH_MS` | Turn taking |
| `SERVER_NAME`, `SERVER_PORT`, `SHARE`, `TURN_SERVER` | Web server |

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
skipped when the files are not on disk. Run only them with
`pytest -m realmodels`.

## Screenshot

![Screenshot](images/image.png)

## License

MIT. See [LICENSE](LICENSE).
