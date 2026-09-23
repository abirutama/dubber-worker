import re
from pathlib import Path

# Find the manage.py inside the venv
manage_files = list(Path("/opt/venv/lib").glob("python*/site-packages/TTS/utils/manage.py"))
if not manage_files:
    raise SystemExit("Could not find TTS/utils/manage.py under /opt/venv/lib/python*/site-packages")

manage_path = manage_files[0]
text = manage_path.read_text(encoding="utf-8")

# Replace the entire ask_tos method with a non-interactive version
pattern = r"def ask_tos\(self, output_path\):[\s\S]*?\n\s*return\s+answer\s+in\s+\[.*?\]\s*\n"
m = re.search(pattern, text)
if not m:
    raise SystemExit(f"Could not locate ask_tos() function in {manage_path}. File format may differ.")

replacement = """def ask_tos(self, output_path):
        \"""
        Non-interactive acceptance of Coqui TOS prompt to allow automated builds.
        By proceeding, you assert you have a commercial license or agree to CPML terms:
        https://coqui.ai/cpml
        \"""
        return True
"""

new_text = text[:m.start()] + replacement + text[m.end():]
manage_path.write_text(new_text, encoding="utf-8")

print(f"Patched: {manage_path} (ask_tos() now auto-accepts).")
