import os
import uuid
import runpod
from TTS.api import TTS

# Cache model once per worker
_XTTS = None

def get_xtts():
    global _XTTS
    if _XTTS is None:
        # This is where your previous crash happened.
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _XTTS


def handler(job):
    """
    Expected input example:
    {
      "text": "Halo dunia",
      "speaker_wav": "/path/to/ref.wav",   # optional
      "language": "id",                    # optional; e.g. "id", "en", "zh"
      "output_dir": "/tmp"                 # optional
    }
    """
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    text = job_input.get("text")
    if not text:
        return {"error": "Missing required field: input.text"}

    speaker_wav = job_input.get("speaker_wav", None)
    language = job_input.get("language", "id")
    output_dir = job_input.get("output_dir", "/tmp")

    os.makedirs(output_dir, exist_ok=True)
    out_wav = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.wav")

    tts = get_xtts()

    # XTTS v2 API: use tts_to_file with speaker_wav for voice cloning
    # (If you don't pass speaker_wav, it will use a default voice)
    if speaker_wav:
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=out_wav)
    else:
        tts.tts_to_file(text=text, language=language, file_path=out_wav)

    return {"output_wav": out_wav}


runpod.serverless.start({"handler": handler})
