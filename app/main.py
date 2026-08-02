"""Local voice API — text-to-speech server backed by Kokoro."""

import io
import logging
import os
import subprocess
import threading

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from kokoro import KPipeline
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-api")

SAMPLE_RATE = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHATTERBOX_URL = os.environ.get("CHATTERBOX_URL", "http://chatterbox:8001")
ENGINES = ("kokoro", "chatterbox")

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
    "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-compression_level", "5"]),
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
            logger.info("Loading Kokoro pipeline for lang_code=%s on %s", lang_code, DEVICE)
            _pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=DEVICE)
        return _pipelines[lang_code]


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=20000, description="Text to synthesize")
    engine: str = Field("kokoro", description="TTS engine: kokoro (fast, named voices) or chatterbox (more natural prosody; ignores 'voice')")
    voice: str = Field("af_heart", description="Voice name, e.g. af_heart (kokoro engine only)")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    response_format: str = Field("wav", description="Audio format: wav, mp3, or opus (Opus in an OGG container)")
    exaggeration: float = Field(0.5, ge=0.0, le=2.0, description="Chatterbox only: emotion intensity (0.5 = neutral)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="Chatterbox only: pacing/adherence control")


def synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    try:
        pipeline = get_pipeline(voice[0])
        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio))
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


def transcode(wav_bytes: bytes, ffmpeg_args: list[str] | None, speed: float = 1.0) -> bytes:
    args = list(ffmpeg_args) if ffmpeg_args else ["-f", "wav"]
    if speed != 1.0:
        args = ["-af", f"atempo={speed}", *args]
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"],
        input=wav_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def synthesize_chatterbox(req: "SpeechRequest") -> bytes:
    try:
        resp = httpx.post(
            f"{CHATTERBOX_URL}/synthesize",
            json={"input": req.input, "exaggeration": req.exaggeration, "cfg_weight": req.cfg_weight},
            timeout=600,
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Chatterbox engine is not running. Start it with: docker compose --profile chatterbox up -d",
        )
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise HTTPException(status_code=502, detail=f"Chatterbox engine error: {detail}")
    return resp.content


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "loaded_languages": sorted(_pipelines)}


@app.get("/v1/voices")
def list_voices():
    return {"voices": VOICES}


@app.post("/v1/audio/speech")
def create_speech(req: SpeechRequest):
    if req.engine not in ENGINES:
        raise HTTPException(status_code=400, detail=f"Unknown engine '{req.engine}'. Available: {', '.join(ENGINES)}")
    if req.response_format not in FORMATS:
        raise HTTPException(status_code=400, detail="response_format must be 'wav', 'mp3', or 'opus'")

    media_type, ffmpeg_args = FORMATS[req.response_format]
    if req.engine == "chatterbox":
        wav = synthesize_chatterbox(req)
        # chatterbox has no native speed control; apply it in the transcode
        if ffmpeg_args is None and req.speed == 1.0:
            body = wav
        else:
            body = transcode(wav, ffmpeg_args, speed=req.speed)
    else:
        if req.voice not in ALL_VOICES:
            raise HTTPException(status_code=400, detail=f"Unknown voice '{req.voice}'. See /v1/voices.")
        wav = to_wav(synthesize(req.input, req.voice, req.speed))
        body = wav if ffmpeg_args is None else transcode(wav, ffmpeg_args)
    return Response(content=body, media_type=media_type)
