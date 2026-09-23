FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ensure pkg_resources exists (setuptools)
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Auto-accept Coqui TOS prompt (non-interactive)
COPY patches/auto_accept_coqui_tos.py /app/patches/auto_accept_coqui_tos.py
RUN python /app/patches/auto_accept_coqui_tos.py

# (Optional) Pre-warm caches at build time.
# Comment these out if you want faster builds.
# 1) XTTS v2 download
RUN python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
# 2) NLLB model download
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; m='facebook/nllb-200-distilled-600M'; AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
