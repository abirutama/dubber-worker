# AGENT.md — Build & Deploy Runpod Serverless: ASR (Whisper medium) + NLLB-200 + XTTS-v2 voice cloning → MP3 24kHz

## Goal
Create a Runpod Serverless endpoint that accepts an `audio_url` (MP3) and a list of target languages, then:
1) downloads audio
2) converts to WAV 24kHz
3) transcribes with Whisper **medium**
4) translates with **NLLB-200**
5) generates cloned-voice speech with **XTTS-v2**
6) encodes MP3 **24kHz**
7) uploads outputs to Cloudflare R2 (S3-compatible)
8) returns URLs per language

The endpoint must be **async** (Runpod `/run` + `/status/{job_id}`).

## High-level design
- Input is URL-based (no multipart upload) to fit serverless constraints.
- Each job can request N targets.
- Output: 1 MP3 file per target language.

## Requirements
- A container registry: GHCR (recommended) or Docker Hub
- Runpod API key (`RUNPOD_API_KEY`)
- Cloudflare R2 credentials (S3-compatible):
  - `R2_ENDPOINT_URL` (e.g. `https://<accountid>.r2.cloudflarestorage.com`)
  - `R2_ACCESS_KEY_ID`
  - `R2_SECRET_ACCESS_KEY`
  - `R2_BUCKET_DEFAULT` (optional)
  - `R2_PUBLIC_BASE_URL` (optional but recommended; public https base for returned URLs)
- A Cloudflare R2 bucket and a way to generate **signed GET URLs** for input audio.

## Repo to create (files)
Create a repo folder `dubber-worker/` containing:

### `requirements.txt`
Pin versions (adjust only if necessary to fix compatibility):
- runpod
- boto3, botocore
- faster-whisper
- transformers, sentencepiece, sacremoses
- TTS (Coqui)
- pydub, requests

### `Dockerfile`
- Base: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- Install `ffmpeg`
- `pip install -r requirements.txt`
- run `handler.py`

### `handler.py`
Implement Runpod serverless handler with steps:
1) download `audio_url` → `/tmp/work-<job_id>/in.mp3`
2) `ffmpeg` decode/resample to mono WAV 24kHz
3) extract `reference_seconds` (default 6s) from the resampled wav → `ref.wav`
4) `faster-whisper` model `medium` → `detected_language`, `transcript`
5) NLLB-200 translation:
   - default model: `facebook/nllb-200-distilled-600M`
   - accept `nllb_src_lang` from input (optional)
6) XTTS-v2 voice cloning:
   - load `tts_models/multilingual/multi-dataset/xtts_v2`
   - generate WAV per target using `speaker_wav=ref.wav`
   - IMPORTANT: XTTS language codes differ from NLLB codes. Input must include:
     - `tts_lang_map` mapping `nllb_code -> xtts_code` (e.g. `fra_Latn -> fr`)
7) encode WAV → MP3 24kHz using ffmpeg (`libmp3lame`, 128k)
8) upload each MP3 to R2 using boto3 client:
   - key format: `{prefix}/{job_id}/{target}.mp3`
9) return JSON:
   - job_id
   - detected_language
   - transcript
   - translations: map target -> translated text
   - outputs: map target -> {audio_url, r2_key}

Add `runpod.serverless.progress_update` messages for observability.

## Build & push image
Prefer GHCR:

### GHCR
- Ensure `docker login ghcr.io` is configured.
- Image name: `ghcr.io/<GITHUB_USER>/dubber-worker:<TAG>`

Commands:
```bash
docker build -t ghcr.io/<GITHUB_USER>/dubber-worker:0.1 .
docker push ghcr.io/<GITHUB_USER>/dubber-worker:0.1
