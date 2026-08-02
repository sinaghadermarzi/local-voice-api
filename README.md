# local-voice-api

A Docker-based text-to-speech API server powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Send text, get back audio (WAV or MP3).

## Quick start

```bash
docker compose up --build -d
```

The first request per language downloads the model weights (~330 MB) from Hugging Face; they're cached in a named volume so restarts are fast.

## API

### `POST /v1/audio/speech`

Request body:

| Field             | Type   | Default    | Notes                        |
|-------------------|--------|------------|------------------------------|
| `input`           | string | (required) | Text to synthesize           |
| `voice`           | string | `af_heart` | See `/v1/voices`             |
| `speed`           | float  | `1.0`      | 0.5–2.0                      |
| `response_format` | string | `wav`      | `wav` or `mp3`               |

Returns raw audio bytes (`audio/wav` or `audio/mpeg`).

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello! This is my local voice API.", "voice": "af_heart"}' \
  -o hello.wav
```

### `GET /v1/voices`

Lists available voices grouped by language (American/British English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin).

### `GET /health`

Returns `{"status": "ok", "loaded_languages": [...]}`.

Interactive docs at http://localhost:8000/docs.

## Notes

- Runs on CPU (torch CPU build); Kokoro-82M is small enough that short texts synthesize in a few seconds.
- Long texts are chunked by Kokoro internally and concatenated into one audio file.
