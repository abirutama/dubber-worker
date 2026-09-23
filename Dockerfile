FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

# 1. Install System Dependencies & Audio tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Upgrade pip
RUN python3 -m pip install --upgrade pip

# 3. Install PyTorch dengan CUDA 12.1 (Stabil di GPU RunPod)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install All Python Packages dari PyPI
RUN pip3 install \
    runpod \
    faster-whisper \
    ctranslate2 \
    transformers \
    sentencepiece \
    boto3 \
    pydub \
    f5-tts

# 5. Bake Weights Model ke Image saat Build (Bebas Cold-Start saat Runtime)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-1.3B'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-1.3B')"

# 6. Copy skrip handler
COPY handler.py /app/handler.py

CMD [ "python3", "-u", "/app/handler.py" ]