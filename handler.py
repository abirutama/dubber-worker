import os
import uuid
import runpod
from TTS.api import TTS

_XTTS = None

def get_xtts():
    global _XTTS
    if _XTTS is None:
        # NOTE: first run may trigger TOS prompt inside TTS download path.
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _XTTS


def handler(job):
    job_input = job.get("input", {}) if isinstance(job, dict) else {}

    text = job_input.get("text")
    if not text:
        return {"error": "Missing required field: input.text"}

    speaker_wav = job_input.get("speaker_wav")
    language = job_input.get("language", "id")
    output_dir = job_input.get("output_dir", "/tmp")

    os.makedirs(output_dir, exist_ok=True)
    out_wav = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.wav")

    tts = get_xtts()
    if speaker_wav:
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=out_wav)
    else:
        tts.tts_to_file(text=text, language=language, file_path=out_wav)

    return {"output_wav": out_wav}


runpod.serverless.start({"handler": handler})
