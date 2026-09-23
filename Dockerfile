FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    pkg-config \
    python3-dev \
    libsndfile1 \
    libsox-dev \
    sox \
    libffi-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv to avoid conflicts with distutils-installed packages in base image
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade build tooling
RUN pip install -U pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
