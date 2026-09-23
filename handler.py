import runpod
import torch
import boto3
import os
import gc
import ctranslate2
import transformers
from faster_whisper import WhisperModel
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_vocoder, load_model, infer_process

# 1. INISIALISASI MODEL LOKAL (Global Scope)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Whisper Large-v3...")
whisper_model = WhisperModel("large-v3", device=device, compute_type="float16")

print("Loading NLLB-1.3B via CTranslate2...")
# Pastikan model NLLB sudah dikonversi ke format ct2 atau gunakan huggingface transformer biasa jika VRAM muat
tokenizer_nllb = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
translator_nllb = ctranslate2.Translator("facebook/nllb-200-distilled-1.3B", device=device)

print("Loading F5-TTS...")
vocoder = load_vocoder(vocoder_name="vocos", device=device)
f5_model = load_model(DiT, {"dim": 1024, "depth": 22, "heads": 16}, checkpoint_path=None, device=device)

# Cloudflare R2 Client
s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY")
)

def handler(job):
    job_input = job['input']
    
    bucket_name = job_input.get('bucket_name')
    file_key = job_input.get('file_key') # contoh: "inputs/sample.mp3"
    target_lang_code = job_input.get('target_lang', 'ind_Latn') # Default ke Indonesia (ind_Latn)
    
    local_input = "/tmp/input.mp3"
    local_output = "/tmp/output.wav"
    
    try:
        # A. Download mp3 sumber dari R2
        s3_client.download_file(bucket_name, file_key, local_input)
        
        # B. Transcribe dengan Whisper
        segments, _ = whisper_model.transcribe(local_input, beam_size=5)
        source_text = " ".join([segment.text for segment in segments])
        
        # C. Translate menggunakan NLLB
        source_tokens = tokenizer_nllb.convert_ids_to_tokens(tokenizer_nllb.encode(source_text))
        results = translator_nllb.translate_batch([source_tokens], target_prefix=[[target_lang_code]])
        target_tokens = results[0].hypothesis[0][1:] # Skip prefix
        translated_text = tokenizer_nllb.decode(tokenizer_nllb.convert_tokens_to_ids(target_tokens))
        
        # D. Synthesize & Voice Clone dengan F5-TTS
        # F5-TTS menggunakan audio sampel acuan (local_input) dan teks hasil terjemahan
        infer_process(
            ref_audio=local_input,
            ref_text=source_text, # Teks asli sebagai acuan fonem acuan audio
            gen_text=translated_text, # Teks hasil terjemahan yang mau diucapkan
            model_obj=f5_model,
            vocoder=vocoder,
            output_dir="/tmp",
            file_name="output.wav",
            device=device
        )
        
        # E. Upload hasil kembali ke Cloudflare R2
        output_key = f"outputs/translated_{os.path.basename(file_key)}.wav"
        s3_client.upload_file(local_output, bucket_name, output_key)
        
        return {
            "status": "success",
            "source_text": source_text,
            "translated_text": translated_text,
            "output_key": output_key
        }

    finally:
        # Bersihkan Memory GPU
        gc.collect()
        torch.cuda.empty_cache()

runpod.serverless.start({"handler": handler})