import os
import uuid
import traceback
import runpod
from TTS.api import TTS

_XTTS = None

def get_xtts():
    global _XTTS
    if _XTTS is None:
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _XTTS

def handler(job):
    try:
        inp = job.get("input", {}) if isinstance(job, dict) else {}
        text = inp.get("text")
        if not text or not isinstance(text, str):
            return {"error": "Missing/invalid input.text"}

        language = inp.get("language", "id")
        speaker_wav = inp.get("speaker_wav")

        out_wav = f"/tmp/tts_{uuid.uuid4().hex}.wav"
        tts = get_xtts()

        if speaker_wav:
            if not os.path.exists(speaker_wav):
                return {"error": f"speaker_wav not found: {speaker_wav}"}
            tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=out_wav)
        else:
            tts.tts_to_file(text=text, language=language, file_path=out_wav)

        return {"output_wav": out_wav}

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})