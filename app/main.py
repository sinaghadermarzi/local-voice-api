"""Local voice API — text-to-speech server backed by Kokoro."""

import io
import logging
import subprocess
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from kokoro import KPipeline
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-api")

SAMPLE_RATE = 24000

# English-only voice set. The first letter of a voice name selects the
# pipeline language: a=American English, b=British English.
VOICES = {
    "American English": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "British English": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
}
ALL_VOICES = {v for group in VOICES.values() for v in group}

# (media type, extra ffmpeg args) per response format; wav needs no transcode
FORMATS = {
    "wav": ("audio/wav", None),
    "mp3": ("audio/mpeg", ["-f", "mp3", "-b:a", "128k"]),
    "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus", "-b:a", "32k", "-vbr", "on"]),
}

app = FastAPI(title="Local Voice API", version="0.2.0")

_pipelines: dict[str, KPipeline] = {}
_lock = threading.Lock()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Synthesis failed: {type(exc).__name__}: {exc}"},
    )


def get_pipeline(lang_code: str) -> KPipeline:
    with _lock:
        if lang_code not in _pipelines:
            logger.info("Loading Kokoro pipeline for lang_code=%s", lang_code)
            _pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        return _pipelines[lang_code]


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=20000, description="Text to synthesize")
    voice: str = Field("af_heart", description="Voice name, e.g. af_heart")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    response_format: str = Field("wav", description="Audio format: wav, mp3, or opus (Opus in an OGG container)")


def synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    try:
        pipeline = get_pipeline(voice[0])
        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
    except Exception as exc:
        logger.exception("Synthesis failed for voice=%s", voice)
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed for voice '{voice}': {type(exc).__name__}: {exc}",
        )
    if not chunks:
        raise HTTPException(status_code=422, detail="No audio produced for the given input")
    return np.concatenate(chunks)


def to_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcode(wav_bytes: bytes, ffmpeg_args: list[str]) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"],
        input=wav_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


@app.get("/health")
def health():
    return {"status": "ok", "loaded_languages": sorted(_pipelines)}


@app.get("/v1/voices")
def list_voices():
    return {"voices": VOICES}


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    if req.voice not in ALL_VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice '{req.voice}'. See /v1/voices.")
    if req.response_format not in FORMATS:
        raise HTTPException(status_code=400, detail="response_format must be 'wav', 'mp3', or 'opus'")

    media_type, ffmpeg_args = FORMATS[req.response_format]
    audio = synthesize(req.input, req.voice, req.speed)
    wav = to_wav(audio)
    body = wav if ffmpeg_args is None else transcode(wav, ffmpeg_args)
    return Response(content=body, media_type=media_type)
