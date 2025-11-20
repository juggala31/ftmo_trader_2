import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "ai" / "ai_profile.json"


def backup_profile(path: Path) -> None:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = path.parent / "profile_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.name}.bak_{ts}"
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[fix_ai_profile] Backup written -> {backup_path}")


def main() -> None:
    if not PROFILE_PATH.exists():
        print(f"[fix_ai_profile][error] not found: {PROFILE_PATH}")
        return

    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    changed = False

    # 1) Clamp US100Z25.sim and XAUZ25.sim risk to 1% ATR_RISK
    for sym in ("US100Z25.sim", "XAUZ25.sim"):
        node = data.get(sym)
        if not isinstance(node, dict):
            print(f"[fix_ai_profile][warn] {sym} missing top-level block; skipping risk fix.")
            continue

        risk = node.setdefault("risk", {})
        old = risk.get("atr_risk_pct")
        if old != 0.01:
            print(f"[fix_ai_profile] {sym}: atr_risk_pct {old} -> 0.01")
            risk["atr_risk_pct"] = 0.01
            changed = True

    # 2) Ensure USOILZ25.sim has a top-level AI block so tuner can see it
    if "USOILZ25.sim" not in data:
        print("[fix_ai_profile] Adding AI block for USOILZ25.sim (paper, 1% ATR_RISK).")
        data["USOILZ25.sim"] = {
            "enabled": False,
            "risk": {
                "atr_risk_pct": 0.01,
                "mode": "ATR_RISK"
            },
            "timeframes": {
                "H1": {
                    "enabled": True,
                    "long_threshold": 0.65,
                    "short_threshold": 0.35
                }
            },
            "trade_mode": "paper"
        }
        changed = True
    else:
        print("[fix_ai_profile] USOILZ25.sim AI block already present; no add needed.")

    if not changed:
        print("[fix_ai_profile] No changes needed; profile already in safe state.")
        return

    # 3) Backup and write new profile
    backup_profile(PROFILE_PATH)
    PROFILE_PATH.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    print(f"[fix_ai_profile] Updated profile -> {PROFILE_PATH}")


if __name__ == "__main__":
    main()