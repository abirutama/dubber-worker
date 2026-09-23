import runpod
import torch
import boto3
import os
import gc
import transformers
from faster_whisper import WhisperModel
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_vocoder, load_model, infer_process

# ==========================================
# 1. LOAD MODEL KE GPU (Global Scope)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("[INIT] Loading Whisper Large-v3...")
whisper_model = WhisperModel("large-v3", device=device, compute_type="float16")

print("[INIT] Loading NLLB-1.3B...")
tokenizer_nllb = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model_nllb = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)

print("[INIT] Loading F5-TTS...")
vocoder = load_vocoder(vocoder_name="vocos", device=device)
f5_model = load_model(DiT, {"dim": 1024, "depth": 22, "heads": 16}, checkpoint_path=None, device=device)

# ==========================================
# 2. CLIENT CLOUDFLARE R2
# ==========================================
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY")
)

# ==========================================
# 3. HANDLER UTAMA RUNPOD
# ==========================================
def handler(job):
    job_input = job['input']
    
    # Input parameter dari Client
    bucket_name = job_input.get('bucket_name')
    file_key = job_input.get('file_key')                        # Contoh: "inputs/suara_awal.mp3"
    target_lang_nllb = job_input.get('target_lang', 'ind_Latn')  # Default: Bahasa Indonesia ('ind_Latn')
    
    local_input = "/tmp/input_audio.mp3"
    local_output_dir = "/tmp"
    local_output_filename = "output_cloned.wav"
    local_output_path = os.path.join(local_output_dir, local_output_filename)
    
    try:
        # A. Download audio sumber dari Cloudflare R2
        print(f"[R2] Downloading {file_key} from bucket {bucket_name}...")
        s3_client.download_file(bucket_name, file_key, local_input)
        
        # B. Transcribe menggunakan Faster-Whisper
        print("[Whisper] Transcribing audio...")
        segments, _ = whisper_model.transcribe(local_input, beam_size=5)
        source_text = " ".join([segment.text for segment in segments]).strip()
        print(f"[Whisper] Result: {source_text}")
        
        # C. Translate menggunakan NLLB
        print(f"[NLLB] Translating text to language code: {target_lang_nllb}...")
        inputs = tokenizer_nllb(source_text, return_tensors="pt").to(device)
        translated_tokens = model_nllb.generate(
            **inputs, 
            forced_bos_token_id=tokenizer_nllb.lang_code_to_id[target_lang_nllb], 
            max_length=512
        )
        translated_text = tokenizer_nllb.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(f"[NLLB] Translated Result: {translated_text}")
        
        # D. Voice Cloning & Synthesize menggunakan F5-TTS
        print("[F5-TTS] Synthesizing cloned audio...")
        infer_process(
            ref_audio=local_input,
            ref_text=source_text,        # Teks asli sebagai acuan aksen/nada
            gen_text=translated_text,    # Teks terjemahan yang dibacakan
            model_obj=f5_model,
            vocoder=vocoder,
            output_dir=local_output_dir,
            file_name=local_output_filename,
            device=device
        )
        
        # E. Upload hasil audio ke Cloudflare R2
        output_key = f"outputs/translated_{os.path.basename(file_key)}.wav"
        print(f"[R2] Uploading generated audio to {output_key}...")
        s3_client.upload_file(local_output_path, bucket_name, output_key)
        
        return {
            "status": "success",
            "source_text": source_text,
            "translated_text": translated_text,
            "output_r2_key": output_key
        }

    except Exception as e:
        print(f"[ERROR] Process failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        # Bersihkan file temporer & VRAM GPU
        if os.path.exists(local_input):
            os.remove(local_input)
        if os.path.exists(local_output_path):
            os.remove(local_output_path)
        gc.collect()
        torch.cuda.empty_cache()

# Start Serverless Loop
runpod.serverless.start({"handler": handler})