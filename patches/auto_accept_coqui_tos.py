from pathlib import Path

manage_files = list(Path("/opt/venv/lib").glob("python*/site-packages/TTS/utils/manage.py"))
if not manage_files:
    raise SystemExit("Could not find TTS/utils/manage.py under /opt/venv/lib/python*/site-packages")

manage_path = manage_files[0]
src = manage_path.read_text(encoding="utf-8")

marker = "# --- RUNPOD_AUTO_ACCEPT_TOS_PATCH ---"
if marker in src:
    print(f"Patch already applied: {manage_path}")
    raise SystemExit(0)

patch = f"""
\n{marker}
# Injected to disable interactive Coqui TOS prompts in non-interactive environments (Docker build / serverless).
# By proceeding, you assert you have a commercial license or agree to CPML terms:
# https://coqui.ai/cpml

def _runpod_always_true(*args, **kwargs):
    return True

try:
    ModelManager.ask_tos = _runpod_always_true
    ModelManager.tos_agreed = _runpod_always_true
except Exception as e:
    print("Runpod TOS patch failed to apply:", e)
"""

manage_path.write_text(src + patch, encoding="utf-8")
print(f"Injected Runpod TOS auto-accept patch into: {manage_path}")
