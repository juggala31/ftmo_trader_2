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
        raise SystemExit(f"[set_default_risk] profile not found: {PROFILE_PATH}")
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


def normalize_risk():
    profile = load_profile()
    backup_path = save_backup()
    print(f"[set_default_risk] Backup written -> {backup_path}")

    symbols = iter_symbols(profile)
    if not symbols:
        print("[set_default_risk] No symbols found in profile.")
        return

    total_changed = 0

    for sym in symbols:
        cfg = get_sym_cfg(profile, sym)

        # Symbol-level atr_risk_pct
        cur = cfg.get("atr_risk_pct", cfg.get("risk_pct", 0.0))
        new_val = cur

        # Normalize obvious percent-style values (1 -> 0.01, 2 -> 0.02, etc.)
        if isinstance(cur, (int, float)):
            if cur > 0.2 and cur <= 20.0:
                # Treat as percent, convert to fraction
                new_val = cur / 100.0

        # If missing or zero, set to default
        if not isinstance(new_val, (int, float)) or new_val == 0.0:
            new_val = DEFAULT_RISK

        if new_val != cur:
            print(f"[set_default_risk] {sym}: atr_risk_pct {cur} -> {new_val}")
            cfg["atr_risk_pct"] = new_val
            total_changed += 1
        else:
            # Ensure atr_risk_pct key exists
            cfg.setdefault("atr_risk_pct", new_val)

        # Per-TF (H1 for now)
        for tf in ("H1",):
            tf_cfg = cfg.setdefault(tf, {})
            tf_cur = tf_cfg.get("atr_risk_pct", 0.0)
            tf_new = tf_cur
            if not isinstance(tf_cur, (int, float)) or tf_cur == 0.0:
                tf_new = new_val
            if tf_new != tf_cur:
                print(f"[set_default_risk] {sym}.{tf}: atr_risk_pct {tf_cur} -> {tf_new}")
                tf_cfg["atr_risk_pct"] = tf_new
                total_changed += 1

    PROFILE_PATH.write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[set_default_risk] Updated profile -> {PROFILE_PATH}")
    print(f"[set_default_risk] Changes applied: {total_changed}")


if __name__ == "__main__":
    normalize_risk()