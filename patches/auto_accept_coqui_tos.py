from pathlib import Path
import re

manage_files = list(Path("/opt/venv/lib").glob("python*/site-packages/TTS/utils/manage.py"))
if not manage_files:
    raise SystemExit("Could not find TTS/utils/manage.py under /opt/venv/lib/python*/site-packages")

manage_path = manage_files[0]
src = manage_path.read_text(encoding="utf-8")

# Find: def ask_tos(self, output_path):
m_start = re.search(
    r"^(\s*)def\s+ask_tos\s*\(\s*self\s*,\s*output_path\s*\)\s*:\s*$",
    src,
    re.MULTILINE
)
if not m_start:
    raise SystemExit(f"Could not find def ask_tos(self, output_path): in {manage_path}")

indent = m_start.group(1)

# Replace until the next "def ..." at same indent (end of function)
m_next = re.search(
    rf"^(?:{re.escape(indent)})def\s+\w+\s*\(.*\)\s*:\s*$",
    src[m_start.end():],
    re.MULTILINE
)
end_idx = m_start.end() + (m_next.start() if m_next else len(src) - m_start.end())

replacement = (
f"""{indent}def ask_tos(self, output_path):
{indent}    \"""
{indent}    Non-interactive acceptance of Coqui TOS prompt to allow automated builds.
{indent}    By proceeding, you assert you have a commercial license or agree to CPML terms:
{indent}    https://coqui.ai/cpml
{indent}    \"""
{indent}    return True

"""
)

patched = src[:m_start.start()] + replacement + src[end_idx:]
manage_path.write_text(patched, encoding="utf-8")

print(f"Patched ask_tos() in: {manage_path}")