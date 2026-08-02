"""Internal Chatterbox TTS engine service. Returns raw WAV; the gateway handles formats."""

import io
import logging
import re
import threading
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatterbox-engine")

MAX_CHUNK_CHARS = 400  # Chatterbox degrades on very long single generations
CHUNK_GAP_SECONDS = 0.2

_model = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    import torch
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading ChatterboxTTS on %s ...", device)
    _model = ChatterboxTTS.from_pretrained(device=device)
    logger.info("ChatterboxTTS ready (sr=%s)", _model.sr)
    yield


app = FastAPI(title="Chatterbox engine", lifespan=lifespan)


class SynthesizeRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=20000)
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)


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
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    pieces = []
    gap = np.zeros(int(_model.sr * CHUNK_GAP_SECONDS), dtype=np.float32)
    with _lock:
        for i, chunk in enumerate(split_text(req.input)):
            wav = _model.generate(chunk, exaggeration=req.exaggeration, cfg_weight=req.cfg_weight)
            if i > 0:
                pieces.append(gap)
            pieces.append(wav.detach().cpu().numpy().squeeze())
    audio = np.concatenate(pieces)
    buf = io.BytesIO()
    sf.write(buf, audio, _model.sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
