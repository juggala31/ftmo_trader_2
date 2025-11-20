import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
PROFILE_PATH = AI_DIR / "ai_profile.json"
BACKUP_DIR = AI_DIR / "profile_backups"

DEFAULT_RISK = 0.01  # 1% per trade


def load_profile() -> Dict:
    if not PROFILE_PATH.exists():
        raise SystemExit(f"[fix_sim_risk] profile not found: {PROFILE_PATH}")
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"ai_profile.json.bak_{ts}"
    backup_path.write_bytes(PROFILE_PATH.read_bytes())
    return backup_path


def iter_symbols(profile: Dict) -> List[str]:
    symbols: List[str] = []

    sym_block = profile.get("symbols")
    if isinstance(sym_block, dict):
        symbols.extend(sym_block.keys())

    for k, v in profile.items():
        if k in ("global", "symbols"):
            continue
        if isinstance(v, dict):
            symbols.append(k)

    return sorted(sorted(set(symbols)))


def get_sym_cfg(profile: Dict, symbol: str) -> Dict:
    sym_block = profile.get("symbols")
    if isinstance(sym_block, dict) and isinstance(sym_block.get(symbol), dict):
        return sym_block[symbol]

    if isinstance(profile.get(symbol), dict):
        return profile[symbol]

    # If missing, create in symbols block
    if not isinstance(sym_block, dict):
        profile["symbols"] = {}
        sym_block = profile["symbols"]
    sym_block[symbol] = {}
    return sym_block[symbol]


def fix_sim_risk():
    profile = load_profile()
    backup_path = save_backup()
    print(f"[fix_sim_risk] Backup written -> {backup_path}")

    symbols = iter_symbols(profile)
    if not symbols:
        print("[fix_sim_risk] No symbols found in profile.")
        return

    total_changes = 0

    for sym in symbols:
        if not sym.endswith(".sim"):
            continue

        cfg = get_sym_cfg(profile, sym)

        cur = cfg.get("atr_risk_pct", cfg.get("risk_pct", 0.0))
        if not isinstance(cur, (int, float)):
            cur = 0.0

        new_val = cur
        if new_val == 0.0:
            new_val = DEFAULT_RISK

        if new_val != cur:
            print(f"[fix_sim_risk] {sym}: atr_risk_pct {cur} -> {new_val}")
            cfg["atr_risk_pct"] = new_val
            total_changes += 1
        else:
            cfg.setdefault("atr_risk_pct", new_val)

        # H1 block
        tf_cfg = cfg.setdefault("H1", {})
        tf_cur = tf_cfg.get("atr_risk_pct", 0.0)
        if not isinstance(tf_cur, (int, float)):
            tf_cur = 0.0
        tf_new = tf_cur if tf_cur != 0.0 else new_val
        if tf_new != tf_cur:
            print(f"[fix_sim_risk] {sym}.H1: atr_risk_pct {tf_cur} -> {tf_new}")
            tf_cfg["atr_risk_pct"] = tf_new
            total_changes += 1

    PROFILE_PATH.write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[fix_sim_risk] Updated profile -> {PROFILE_PATH}")
    print(f"[fix_sim_risk] Changes applied: {total_changes}")


if __name__ == "__main__":
    fix_sim_risk()