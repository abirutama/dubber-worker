import os
import uuid
import traceback
from typing import Dict, Any, List

import runpod
import requests
from pydub import AudioSegment

import torch
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from TTS.api import TTS

_WHISPER = None
_NLLB_TOKENIZER = None
_NLLB_MODEL = None
_XTTS = None


def _download(url: str, out_path: str) -> str:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def _ensure_whisper() -> WhisperModel:
    global _WHISPER
    if _WHISPER is None:
        model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")
        _WHISPER = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _WHISPER


def _ensure_nllb():
    global _NLLB_TOKENIZER, _NLLB_MODEL
    if _NLLB_TOKENIZER is None or _NLLB_MODEL is None:
        model_name = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
        _NLLB_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _NLLB_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _NLLB_MODEL.to(device)
        _NLLB_MODEL.eval()
    return _NLLB_TOKENIZER, _NLLB_MODEL


def _ensure_xtts():
    global _XTTS
    if _XTTS is None:
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _XTTS


def _make_reference_wav(input_audio_path: str, reference_seconds: int) -> str:
    seg = AudioSegment.from_file(input_audio_path)
    if reference_seconds and reference_seconds > 0:
        seg = seg[: reference_seconds * 1000]
    seg = seg.set_channels(1).set_frame_rate(24000)
    out_ref = f"/tmp/ref_{uuid.uuid4().hex}.wav"
    seg.export(out_ref, format="wav")
    return out_ref


def _asr_transcribe(audio_path: str) -> Dict[str, Any]:
    whisper = _ensure_whisper()
    segments, info = whisper.transcribe(audio_path)

    parts = []
    for s in segments:
        if s.text:
            parts.append(s.text.strip())

    return {
        "text": " ".join(parts).strip(),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
    }


def _nllb_translate(text: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer, model = _ensure_nllb()
    device = next(model.parameters()).device

    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    out_tokens = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=1024,
        num_beams=4,
    )

    return tokenizer.batch_decode(out_tokens, skip_special_tokens=True)[0]


def _export_mp3(any_audio_path: str, mp3_path: str) -> str:
    seg = AudioSegment.from_file(any_audio_path)
    seg.export(mp3_path, format="mp3")
    return mp3_path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input", {}) if isinstance(job, dict) else {}

        audio_url = inp.get("audio_url")
        if not audio_url:
            return {"error": "Missing input.audio_url"}

        targets: List[str] = inp.get("targets") or []
        if not isinstance(targets, list) or not targets:
            return {"error": "Missing/invalid input.targets (must be a non-empty list)"}

        reference_seconds = int(inp.get("reference_seconds", 6) or 6)

        nllb_src_lang = inp.get("nllb_src_lang")
        if not nllb_src_lang:
            return {"error": "Missing input.nllb_src_lang (e.g. zho_Hans, eng_Latn, ind_Latn)"}

        tts_lang_map = inp.get("tts_lang_map") or {}
        if not isinstance(tts_lang_map, dict) or not tts_lang_map:
            return {"error": "Missing/invalid input.tts_lang_map"}

        out_cfg = inp.get("output") or {}
        out_prefix = out_cfg.get("prefix", "output/")
        out_format = (out_cfg.get("format", "mp3") or "mp3").lower()
        if out_format not in ("mp3", "wav"):
            return {"error": "output.format must be 'mp3' or 'wav'"}

        # 1) Download source mp3
        in_audio_path = f"/tmp/in_{uuid.uuid4().hex}.mp3"
        _download(audio_url, in_audio_path)

        # 2) Speaker reference
        ref_wav = _make_reference_wav(in_audio_path, reference_seconds)

        # 3) ASR
        asr = _asr_transcribe(in_audio_path)
        transcript = asr["text"]
        if not transcript:
            return {"error": "ASR produced empty transcript", "asr": asr}

        # 4) Translate + XTTS per target
        xtts = _ensure_xtts()
        outputs = {}

        for tgt in targets:
            if tgt not in tts_lang_map:
                return {"error": f"Missing tts_lang_map entry for target '{tgt}'"}

            translated = _nllb_translate(transcript, src_lang=nllb_src_lang, tgt_lang=tgt)
            tts_lang = tts_lang_map[tgt]

            out_wav = f"/tmp/out_{tgt}_{uuid.uuid4().hex}.wav"
            xtts.tts_to_file(
                text=translated,
                speaker_wav=ref_wav,
                language=tts_lang,
                file_path=out_wav
            )

            if out_format == "mp3":
                out_file = f"/tmp/out_{tgt}_{uuid.uuid4().hex}.mp3"
                _export_mp3(out_wav, out_file)
            else:
                out_file = out_wav

            outputs[tgt] = {
                "key": f"{out_prefix}{os.path.basename(out_file)}",
                "path": out_file,
                "tts_lang": tts_lang,
            }

        return {
            "status": "ok",
            "source_audio_url": audio_url,
            "reference_seconds": reference_seconds,
            "asr": asr,
            "transcript": transcript,
            "outputs": outputs,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
