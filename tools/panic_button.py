import json
import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
PROFILE = AI_DIR / "ai_profile.json"
BACKUP_DIR = AI_DIR / "profile_backups"


def main() -> None:
    print(f"[panic] Root: {ROOT}")
    if not PROFILE.exists():
        print(f"[panic][error] ai_profile.json not found at {PROFILE}")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # Backup current profile
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"panic_backup_{ts}.json"
    try:
        original = PROFILE.read_text(encoding="utf-8")
        backup_path.write_text(original, encoding="utf-8")
        print(f"[panic] Backed up current profile to {backup_path}")
    except Exception as e:
        print(f"[panic][warn] failed to backup profile: {e!r}")

    # Load and modify
    try:
        data = json.loads(original)
    except Exception as e:
        print(f"[panic][error] failed to parse ai_profile.json: {e!r}")
        return

    if not isinstance(data, dict):
        print("[panic][error] ai_profile.json is not a dict at top-level")
        return

    disabled_count = 0
    for sym, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        cfg["enabled"] = False
        cfg["trade_mode"] = "off"
        disabled_count += 1

    # Write back updated profile
    try:
        PROFILE.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[panic] Disabled {disabled_count} symbols (enabled=false, trade_mode='off').")
        print(f"[panic] Updated profile written to {PROFILE}")
    except Exception as e:
        print(f"[panic][error] failed to write updated profile: {e!r}")


if __name__ == "__main__":
    main()