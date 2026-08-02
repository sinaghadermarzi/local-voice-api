FROM python:3.12-slim

# espeak-ng: phonemizer backend used by Kokoro's G2P (misaki)
# libsndfile1: soundfile WAV encoding
# ffmpeg: mp3 conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image several GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Model weights are downloaded from Hugging Face on first use; persist them via a volume
ENV HF_HOME=/cache/huggingface
RUN mkdir -p /cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
