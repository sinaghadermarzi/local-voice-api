"""Local voice API — text-to-speech server backed by Kokoro."""

import io
import logging
import subprocess
import threading

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from kokoro import KPipeline
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-api")

SAMPLE_RATE = 24000

# Known Kokoro v1.0 voices, grouped by language. The first letter of a voice
# name selects the pipeline language: a=American, b=British English, etc.
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
    "Spanish": ["ef_dora", "em_alex", "em_santa"],
    "French": ["ff_siwis"],
    "Hindi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "Italian": ["if_sara", "im_nicola"],
    "Japanese": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tenkoro", "jm_kumo"],
    "Brazilian Portuguese": ["pf_dora", "pm_alex", "pm_santa"],
    "Mandarin Chinese": [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
}
ALL_VOICES = {v for group in VOICES.values() for v in group}

app = FastAPI(title="Local Voice API", version="0.1.0")

_pipelines: dict[str, KPipeline] = {}
_lock = threading.Lock()


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
    response_format: str = Field("wav", description="Audio format: wav or mp3")


def synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    pipeline = get_pipeline(voice[0])
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio))
    if not chunks:
        raise HTTPException(status_code=422, detail="No audio produced for the given input")
    return np.concatenate(chunks)


def to_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "-b:a", "128k", "pipe:1"],
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
    if req.response_format not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'mp3'")

    audio = synthesize(req.input, req.voice, req.speed)
    wav = to_wav(audio)
    if req.response_format == "mp3":
        return Response(content=wav_to_mp3(wav), media_type="audio/mpeg")
    return Response(content=wav, media_type="audio/wav")
