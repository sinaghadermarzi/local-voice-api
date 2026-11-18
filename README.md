# local-voice-api

A Docker-based text-to-speech API server with two engines:

- **kokoro** (default) — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M): fast (~0.15 s/sentence on GPU), 28 named English voices
- **chatterbox-turbo** (optional) — [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) 350M:
  more natural prosody, built-in voice plus zero-shot cloned voices from clips in `voices/`.
  (`chatterbox` is accepted as an alias. Set `CHATTERBOX_VARIANT=standard` in compose to run the
  slower original 0.5B model instead.)

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
| `engine`          | string | `kokoro`     | `kokoro` or `chatterbox-turbo`                   |
| `voice`           | string | per engine   | kokoro: `af_heart` default; chatterbox-turbo: `default` or a clip name in `voices/` |
| `speed`           | float  | `1.0`        | 0.5–2.0 (applied via ffmpeg for chatterbox)      |
| `response_format` | string | `wav`        | `wav`, `mp3`, or `opus`                          |
| `temperature`     | float  | `0.8`        | Chatterbox Turbo: sampling temperature           |
| `exaggeration`    | float  | `0.5`        | Standard Chatterbox only (Turbo ignores it)      |
| `cfg_weight`      | float  | `0.5`        | Standard Chatterbox only (Turbo ignores it)      |

If the Chatterbox Turbo engine isn't running, `engine: "chatterbox-turbo"` returns a JSON
503 explaining how to start it. Turbo understands paralinguistic tags in the text such as
`[laugh]`, `[chuckle]` and `[cough]`.

Chatterbox handles one generation at a time (a lock serializes requests). Turbo uses about
2.9 GB of GPU memory when loaded, and working memory is released after each request.

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

Lists available voices: 28 Kokoro English voices (20 American, 8 British; non-English
excluded) and the Chatterbox Turbo voices (`default` plus clips in `voices/`).

### `GET /health`

Returns `{"status": "ok", "loaded_languages": [...]}`.

Interactive docs at http://localhost:8000/docs.

## Notes

- Runs on CPU (torch CPU build); Kokoro-82M is small enough that short texts synthesize in a few seconds.
- Long texts are chunked by Kokoro internally and concatenated into one audio file.
