FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /app/requirements.txt

# Pre-download XTTS v2 at build time (reduces cold start + avoids runtime prompt/download)
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# App
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
