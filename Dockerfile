FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev ffmpeg libsndfile1 git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install F5-TTS, Faster-Whisper, CTranslate2, dan Dep terkait
RUN pip3 install runpod faster-whisper ctranslate2 transformers sentencepiece boto3 pydub
RUN pip3 install git+https://github.com/SWIRL-AI/F5-TTS.git

# Bake Model Whisper & NLLB saat Docker Build
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-1.3B')"

COPY handler.py /app/handler.py

CMD [ "python3", "-u", "/app/handler.py" ]