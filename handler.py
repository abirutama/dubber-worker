import os
import json
import uuid
import time
import shutil
import subprocess
from urllib.parse import urlparse

import requests
import boto3
import runpod

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Coqui TTS
from TTS.api import TTS


# -----------------------------
# Utilities
# -----------------------------

def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")

def download_file(url: str, out_path: str) -> None:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def ffmpeg_to_wav_24k(in_path: str, out_path: str) -> None:
    # Mono 24kHz WAV for consistency
    run([
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", "24000",
        "-vn",
        out_path
    ])

def ffmpeg_trim(in_wav: str, out_wav: str, seconds: int) -> None:
    run([
        "ffmpeg", "-y",
        "-i", in_wav,
        "-t", str(seconds),
        out_wav
    ])

def ffmpeg_wav_to_mp3_24k(in_wav: str, out_mp3: str) -> None:
    # Encode MP3; keep 24k sample rate
    run([
        "ffmpeg", "-y",
        "-i", in_wav,
        "-ar", "24000",
        "-codec:a", "libmp3lame",
        "-b:a", "128k",
        out_mp3
    ])


# -----------------------------
# R2 (S3-compatible) client
# -----------------------------

def get_s3_client():
    endpoint = os.environ["R2_ENDPOINT_URL"]
    key_id = os.environ["R2_ACCESS_KEY_ID"]
    secret = os.environ["R2_SECRET_ACCESS_KEY"]

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=os.environ.get("R2_REGION", "auto"),
    )

def upload_to_r2(local_path: str, bucket: str, key: str) -> str:
    s3 = get_s3_client()
    extra = {"ContentType": "audio/mpeg"} if local_path.lower().endswith(".mp3") else None
    if extra:
        s3.upload_file(local_path, bucket, key, ExtraArgs=extra)
    else:
        s3.upload_file(local_path, bucket, key)

    # If you have a public base URL (recommended), return it:
    public_base = os.environ.get("R2_PUBLIC_BASE_URL")
    if public_base:
        return f"{public_base.rstrip('/')}/{key}"

    # Otherwise return s3-style URL (may not be directly downloadable unless configured)
    return f"s3://{bucket}/{key}"


# -----------------------------
# Lazy-loaded models (cached in process)
# -----------------------------

_WHISPER = None
_NLLB_TOKENIZER = None
_NLLB_MODEL = None
_XTTS = None

def get_whisper():
    global _WHISPER
    if _WHISPER is None:
        # Use CUDA if available
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        _WHISPER = WhisperModel("medium", device=device, compute_type=compute_type)
    return _WHISPER

def get_nllb():
    global _NLLB_TOKENIZER, _NLLB_MODEL
    if _NLLB_MODEL is None:
        model_id = os.environ.get("NLLB_MODEL_ID", "facebook/nllb-200-distilled-600M")
        _NLLB_TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            _NLLB_MODEL = _NLLB_MODEL.to("cuda")
        _NLLB_MODEL.eval()
    return _NLLB_TOKENIZER, _NLLB_MODEL

def get_xtts():
    global _XTTS
    if _XTTS is None:
        # XTTS v2 model name as supported by Coqui TTS
        # This will download weights on first load.
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            _XTTS = _XTTS.to("cuda")
    return _XTTS


# -----------------------------
# Core pipeline
# -----------------------------

def transcribe(wav_path: str):
    whisper = get_whisper()
    segments, info = whisper.transcribe(wav_path, beam_size=5)
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    detected = getattr(info, "language", None)
    return detected, text

def translate_nllb(text: str, src_lang: str | None, tgt_lang: str):
    tokenizer, model = get_nllb()

    # NLLB expects language codes; for best results set tokenizer.src_lang.
    # If src_lang is unknown, you may map detected lang -> nllb code in your client,
    # or default to eng_Latn/ind_Latn based on your use case.
    if src_lang:
        tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    out = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

def tts_xtts_clone(text: str, speaker_wav: str, lang: str, out_wav: str):
    tts = get_xtts()
    # XTTS uses 'language' param as a short code in many examples (e.g., "en", "id").
    # Your NLLB lang codes are different, so you should pass a TTS language code here.
    # For now, we expect caller to provide tts_lang like "en", "id", "fr", etc.
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=lang, file_path=out_wav)


def handler(event):
    job = event.get("input", {}) or {}

    audio_url = job["audio_url"]
    targets = job.get("targets", [])
    if not targets:
        raise ValueError("targets is required (list of languages)")

    reference_seconds = int(job.get("reference_seconds", 6))

    out_cfg = job.get("output", {}) or {}
    bucket = out_cfg.get("bucket") or os.environ.get("R2_BUCKET_DEFAULT")
    if not bucket:
        raise ValueError("output.bucket is required (or set R2_BUCKET_DEFAULT)")
    prefix = out_cfg.get("prefix", "dubber-out/")
    out_format = out_cfg.get("format", "mp3").lower()
    if out_format not in ["mp3"]:
        raise ValueError("Only mp3 output is supported in this worker")

    # IMPORTANT: XTTS language codes differ from NLLB codes.
    # We'll accept a mapping dict in the request: nllb_code -> xtts_code
    # Example: {"fra_Latn":"fr","deu_Latn":"de","ind_Latn":"id"}
    tts_lang_map = job.get("tts_lang_map", {})

    # For NLLB src_lang, ideally pass something like "eng_Latn" / "ind_Latn".
    # We'll accept it from request; otherwise leave unset.
    nllb_src_lang = job.get("nllb_src_lang")

    job_id = event.get("id") or str(uuid.uuid4())
    workdir = f"/tmp/work-{job_id}"
    ensure_dir(workdir)

    in_mp3 = os.path.join(workdir, "in.mp3")
    in_wav = os.path.join(workdir, "in_24k.wav")
    ref_wav = os.path.join(workdir, "ref.wav")

    runpod.serverless.progress_update(event, "Downloading audio...")
    download_file(audio_url, in_mp3)

    runpod.serverless.progress_update(event, "Decoding + resampling audio (24kHz)...")
    ffmpeg_to_wav_24k(in_mp3, in_wav)

    runpod.serverless.progress_update(event, f"Extracting reference audio ({reference_seconds}s)...")
    ffmpeg_trim(in_wav, ref_wav, reference_seconds)

    runpod.serverless.progress_update(event, "Transcribing (Whisper medium)...")
    detected_lang, transcript = transcribe(in_wav)

    outputs = {}
    translations = {}

    for tgt in targets:
        runpod.serverless.progress_update(event, f"Translating to {tgt} (NLLB-200)...")
        translated = translate_nllb(transcript, nllb_src_lang, tgt)
        translations[tgt] = translated

        tts_lang = tts_lang_map.get(tgt)
        if not tts_lang:
            raise ValueError(f"Missing tts_lang_map for target {tgt}. Provide e.g. {{'{tgt}':'fr'}}")

        out_wav = os.path.join(workdir, f"out_{tgt}.wav")
        out_mp3 = os.path.join(workdir, f"out_{tgt}.mp3")

        runpod.serverless.progress_update(event, f"TTS XTTS-v2 for {tgt} (voice cloning)...")
        tts_xtts_clone(translated, speaker_wav=ref_wav, lang=tts_lang, out_wav=out_wav)

        runpod.serverless.progress_update(event, f"Encoding MP3 24kHz for {tgt}...")
        ffmpeg_wav_to_mp3_24k(out_wav, out_mp3)

        key = f"{prefix.rstrip('/')}/{job_id}/{tgt}.mp3"
        runpod.serverless.progress_update(event, f"Uploading to R2: {key}")
        url = upload_to_r2(out_mp3, bucket=bucket, key=key)

        outputs[tgt] = {"audio_url": url, "r2_key": key}

    # Cleanup
    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "job_id": job_id,
        "detected_language": detected_lang,
        "transcript": transcript,
        "translations": translations,
        "outputs": outputs
    }


runpod.serverless.start({"handler": handler})
