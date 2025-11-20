import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "ai" / "ai_profile.json"


def backup_profile(path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = path.parent / "profile_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.name}.bak_{ts}"
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[set_trading_mode] Backup written -> {backup_path}")
    return backup_path


def clamp_risk_for_mode(data: Dict[str, Any], mode: str) -> List[Tuple[str, float, float]]:
    """
    For each symbol-level block with risk.mode == 'ATR_RISK',
    enforce an upper bound on atr_risk_pct depending on mode:

      eval -> max 0.01 (1%)
      live -> max 0.02 (2%)

    Only *reduces* risk if it's above the cap; never increases.
    Returns list of (symbol, old, new) for changed entries.
    """
    if mode == "eval":
        cap = 0.01
    elif mode == "live":
        cap = 0.02
    else:
        raise ValueError(f"Unknown mode: {mode}")

    changes: List[Tuple[str, float, float]] = []

    for sym, node in data.items():
        if sym in ("global", "symbols", "_mode"):
            continue
        if not isinstance(node, dict):
            continue

        risk = node.get("risk")
        if not isinstance(risk, dict):
            continue

        mode_str = (risk.get("mode") or "").upper()
        if mode_str != "ATR_RISK":
            continue

        old = risk.get("atr_risk_pct")
        if not isinstance(old, (int, float)):
            continue

        if old > cap:
            new = cap
            risk["atr_risk_pct"] = new
            changes.append((sym, old, new))

    return changes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Set trading mode (eval/live) and enforce ATR_RISK caps in ai_profile.json."
    )
    ap.add_argument("--mode", required=True, choices=["eval", "live"], help="Trading mode.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing profile.",
    )
    args = ap.parse_args(argv)

    if not PROFILE_PATH.exists():
        print(f"[set_trading_mode][error] Profile not found: {PROFILE_PATH}")
        raise SystemExit(1)

    data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    old_mode = data.get("_mode", "(none)")
    print(f"[set_trading_mode] Current mode: {old_mode}")
    print(f"[set_trading_mode] New mode:     {args.mode}")

    # Apply risk caps
    changes = clamp_risk_for_mode(data, args.mode)
    if changes:
        print("[set_trading_mode] Risk changes (ATR_RISK caps):")
        for sym, old, new in changes:
            print(f"  {sym}: atr_risk_pct {old*100:.2f}% -> {new*100:.2f}%")
    else:
        print("[set_trading_mode] No ATR_RISK changes needed (all within caps).")

    # Update mode flag
    data["_mode"] = args.mode

    if args.dry_run:
        print("[set_trading_mode] DRY-RUN: not writing profile.")
        return

    backup_profile(PROFILE_PATH)
    PROFILE_PATH.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    print(f"[set_trading_mode] Updated profile -> {PROFILE_PATH}")


if __name__ == "__main__":
    main()