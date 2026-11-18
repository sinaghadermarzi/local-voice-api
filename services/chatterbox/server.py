"""Internal Chatterbox TTS engine service. Returns raw WAV; the gateway handles formats.

Runs Chatterbox Turbo (350M) by default; set CHATTERBOX_VARIANT=standard for the
original 0.5B English model. Voices are reference clips in VOICES_DIR (zero-shot cloning).
"""

import io
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatterbox-engine")

MAX_CHUNK_CHARS = 400  # Chatterbox degrades on very long single generations
CHUNK_GAP_SECONDS = 0.2

# "turbo" (350M, faster) or "standard" (0.5B original English model)
VARIANT = os.environ.get("CHATTERBOX_VARIANT", "turbo").strip().lower()
# Reference clips for voice cloning: <VOICES_DIR>/<name>.wav (or .mp3/.flac), longer than 5 s.
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/voices"))
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")
DEFAULT_VOICE = "default"  # the model's built-in voice

_model = None
_default_conds = None
_voice_conds: dict[str, tuple[float, object]] = {}  # name -> (file mtime, Conditionals)
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _default_conds
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Chatterbox variant=%s on %s ...", VARIANT, device)
    if VARIANT == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        _model = ChatterboxTurboTTS.from_pretrained(device=device)
    elif VARIANT == "standard":
        from chatterbox.tts import ChatterboxTTS
        _model = ChatterboxTTS.from_pretrained(device=device)
    else:
        raise RuntimeError(f"Unknown CHATTERBOX_VARIANT {VARIANT!r} (use turbo or standard)")
    _default_conds = _model.conds
    logger.info("Chatterbox %s ready (sr=%s), voices dir %s", VARIANT, _model.sr, VOICES_DIR)
    yield


app = FastAPI(title="Chatterbox engine", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=20000)
    voice: str = Field(DEFAULT_VOICE, description="'default' or the name of a clip in the voices dir")
    # Only the standard variant uses these; Turbo ignores them.
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    # Turbo sampling temperature (lower = steadier delivery).
    temperature: float = Field(0.8, gt=0.0, le=2.0)


def list_voices() -> list[str]:
    names = set()
    if VOICES_DIR.is_dir():
        names = {p.stem for p in VOICES_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS}
    return [DEFAULT_VOICE, *sorted(names)]


def voice_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise HTTPException(status_code=400, detail=f"Invalid voice name {name!r}")
    for ext in AUDIO_EXTS:
        p = VOICES_DIR / f"{name}{ext}"
        if p.is_file():
            return p
    raise HTTPException(status_code=400, detail=f"Unknown voice {name!r}. Available: {', '.join(list_voices())}")


def use_voice(name: str, exaggeration: float) -> None:
    """Point the model at a voice's conditioning; caller holds _lock."""
    if name == DEFAULT_VOICE:
        _model.conds = _default_conds
        return
    path = voice_path(name)
    mtime = path.stat().st_mtime
    cached = _voice_conds.get(name)
    if cached and cached[0] == mtime:
        _model.conds = cached[1]
        return
    logger.info("Preparing voice %s from %s", name, path.name)
    try:
        _model.prepare_conditionals(str(path), exaggeration=exaggeration)
    except AssertionError as exc:  # e.g. "Audio prompt must be longer than 5 seconds!"
        raise HTTPException(status_code=400, detail=f"Voice {name!r} unusable: {exc}")
    _voice_conds[name] = (mtime, _model.conds)


def split_text(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks or [text]


@app.get("/health")
def health():
    return {"status": "ok", "variant": VARIANT, "model_loaded": _model is not None}


@app.get("/voices")
def voices():
    return {"variant": VARIANT, "voices": list_voices()}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    import torch

    if VARIANT == "turbo":
        kwargs = {"temperature": req.temperature}
    else:
        kwargs = {"exaggeration": req.exaggeration, "cfg_weight": req.cfg_weight}
    pieces = []
    gap = np.zeros(int(_model.sr * CHUNK_GAP_SECONDS), dtype=np.float32)
    with _lock:
        try:
            use_voice(req.voice, req.exaggeration)
            for i, chunk in enumerate(split_text(req.input)):
                wav = _model.generate(chunk, **kwargs)
                if i > 0:
                    pieces.append(gap)
                pieces.append(wav.detach().cpu().numpy().squeeze())
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # give working memory back to the shared 8 GB card
    audio = np.concatenate(pieces)
    buf = io.BytesIO()
    sf.write(buf, audio, _model.sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
