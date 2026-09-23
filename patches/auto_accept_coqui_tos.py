from pathlib import Path
import re
import sys

manage_files = list(Path("/opt/venv/lib").glob("python*/site-packages/TTS/utils/manage.py"))
if not manage_files:
    raise SystemExit("Could not find TTS/utils/manage.py under /opt/venv/lib/python*/site-packages")

manage_path = manage_files[0]
src = manage_path.read_text(encoding="utf-8")

# 1) Find any def ask_tos(...) line (don't lock parameter names)
m = re.search(r"^(\s*)def\s+ask_tos\s*\(.*\)\s*:\s*$", src, re.MULTILINE)
if not m:
    print("ask_tos not found. Printing context around 'tos'/'input' keywords for debugging:\n", file=sys.stderr)

    for kw in ["ask_tos", "TOS", "tos", "cpml", "input("]:
        idx = src.find(kw)
        if idx != -1:
            start = max(0, idx - 300)
            end = min(len(src), idx + 300)
            print(f"\n--- context around '{kw}' ---\n{src[start:end]}\n", file=sys.stderr)

    raise SystemExit(f"Could not locate ask_tos() in {manage_path}")

indent = m.group(1)
start_idx = m.start()

# 2) Replace function body until next def at same indent
m_next = re.search(rf"^(?:{re.escape(indent)})def\s+\w+\s*\(.*\)\s*:\s*$", src[m.end():], re.MULTILINE)
end_idx = m.end() + (m_next.start() if m_next else len(src) - m.end())

replacement = (
f"""{indent}def ask_tos(self, output_path):
{indent}    \"\"\"Auto-accept Coqui TOS prompt for non-interactive environments.
{indent}    By proceeding, you assert you have a commercial license or agree to CPML terms:
{indent}    https://coqui.ai/cpml
{indent}    \"\"\"
{indent}    return True

"""
)

patched = src[:start_idx] + replacement + src[end_idx:]
manage_path.write_text(patched, encoding="utf-8")

print(f"Patched ask_tos() in: {manage_path}")