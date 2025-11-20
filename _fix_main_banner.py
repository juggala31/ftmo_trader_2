import re, pathlib

p = pathlib.Path(r"C:\ftmo_trader_2\run_live.py")
t = p.read_text(encoding="utf-8")

# 1) Ensure the import exists just once
if "from settings import load_settings" not in t:
    t = t.replace("import os, sys, time", "import os, sys, time\nfrom settings import load_settings", 1)

# 2) Normalize the first lines INSIDE main() so s exists before we use it.
banner = (
    "def main():\n"
    "    s = load_settings()\n"
    "    print(\"[OK] Settings loaded\")\n"
    "    print(f\"  mode={s.app_mode}  log={s.log_level}  server={s.mt5_server}\")\n"
    "    print(\"  Symbols: \" + \", \".join(_to_names(getattr(s, \\\"symbols\\\", []))))\n"
    "    print()\n"
)

# Replace the start of main() (first ~15 lines after def main:) with our banner.
t = re.sub(
    r'(?ms)^def\s+main\s*\(\)\s*:\s*\n(?:.*?\n){0,15}',
    banner,
    t,
    count=1
)

p.write_text(t, encoding="utf-8")
print("[OK] main() banner normalized")
