import os
import uuid
import traceback
import runpod

from TTS.api import TTS

# Cache once per worker process (important for serverless performance)
_XTTS = None


def get_xtts():
    """
    Loads XTTS v2 once per worker and caches it globally.
    Requires the XTTS model to be available in cache (ideally pre-downloaded at build time).
    """
    global _XTTS
    if _XTTS is None:
        # This will download/load the model if not present in cache
        _XTTS = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _XTTS


def handler(job):
    """
    Runpod Serverless handler.

    Input schema (job["input"]):
    {
      "text": "Halo dunia",
      "language": "id",                 # optional (default "id")
      "speaker_wav": "/path/ref.wav",   # optional for voice cloning
      "output_format": "wav",           # optional (only wav here)
      "output_dir": "/tmp"              # optional (default /tmp)
    }

    Output:
    {
      "output_wav": "/tmp/tts_xxx.wav"
    }
    """
    try:
        job_input = job.get("input", {}) if isinstance(job, dict) else {}

        text = job_input.get("text")
        if not text or not isinstance(text, str):
            return {"error": "Missing/invalid input.text (must be a non-empty string)"}

        language = job_input.get("language", "id")
        speaker_wav = job_input.get("speaker_wav")

        output_dir = job_input.get("output_dir", "/tmp")
        os.makedirs(output_dir, exist_ok=True)

        # Only WAV output in this handler (simple + compatible)
        out_wav = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.wav")

        tts = get_xtts()

        # XTTS v2 voice cloning: pass speaker_wav if provided
        if speaker_wav:
            if not isinstance(speaker_wav, str):
                return {"error": "input.speaker_wav must be a string path"}
            if not os.path.exists(speaker_wav):
                return {"error": f"speaker_wav not found: {speaker_wav}"}

            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=out_wav
            )
        else:
            tts.tts_to_file(
                text=text,
                language=language,
                file_path=out_wav
            )

        return {
            "output_wav": out_wav,
            "language": language,
            "used_speaker_wav": bool(speaker_wav),
        }

    except Exception as e:
        # Return structured error so the job response is informative
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})