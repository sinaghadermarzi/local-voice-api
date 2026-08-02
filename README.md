# local-voice-api

A Docker-based text-to-speech API server with two engines:

- **kokoro** (default) — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): fast (~0.15 s/sentence on GPU), 28 named English voices
- **chatterbox** (optional) — [Chatterbox](https://github.com/resemble-ai/chatterbox) 0.5B: more natural prosody, emotion control, single voice

## Quick start

```bash
# kokoro only
docker compose up --build -d

# with the chatterbox engine too
docker compose --profile chatterbox up --build -d
```

Model weights download from Hugging Face on first use and are cached in a
named volume, so restarts are fast.

## API

### `POST /v1/audio/speech`

Request body:

| Field             | Type   | Default      | Notes                                            |
|-------------------|--------|--------------|--------------------------------------------------|
| `input`           | string | (required)   | Text to synthesize                               |
| `engine`          | string | `kokoro`     | `kokoro` or `chatterbox`                         |
| `voice`           | string | `af_heart`   | Kokoro only; see `/v1/voices`                    |
| `speed`           | float  | `1.0`        | 0.5–2.0 (applied via ffmpeg for chatterbox)      |
| `response_format` | string | `wav`        | `wav`, `mp3`, or `opus`                          |
| `exaggeration`    | float  | `0.5`        | Chatterbox only: emotion intensity (0.5 neutral) |
| `cfg_weight`      | float  | `0.5`        | Chatterbox only: pacing/adherence                |

If the chatterbox engine isn't running, `engine: "chatterbox"` returns a JSON
503 explaining how to start it.

Returns raw audio bytes (`audio/wav`, `audio/mpeg`, or `audio/ogg`). The `opus`
format is Opus in an OGG container (48 kbps VBR) — suitable for Telegram voice
messages as-is. All errors are JSON: `{"detail": "<reason>"}`.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello! This is my local voice API.", "voice": "af_heart"}' \
  -o hello.wav
```

### `GET /v1/voices`

Lists available voices: 28 English voices (20 American, 8 British). Non-English
Kokoro voices are intentionally excluded.

### `GET /health`

Returns `{"status": "ok", "loaded_languages": [...]}`.

Interactive docs at http://localhost:8000/docs.

## Notes

- Runs on CPU (torch CPU build); Kokoro-82M is small enough that short texts synthesize in a few seconds.
- Long texts are chunked by Kokoro internally and concatenated into one audio file.
