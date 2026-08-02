FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# chatterbox-tts pins its own torch/transformers versions; CUDA wheels come
# from PyPI by default, the driver from Docker GPU passthrough
RUN pip install --no-cache-dir chatterbox-tts fastapi "uvicorn[standard]" soundfile

COPY server.py .

ENV HF_HOME=/cache/huggingface
RUN mkdir -p /cache/huggingface

EXPOSE 8001

# generous start period: first boot downloads ~3 GB of weights
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
