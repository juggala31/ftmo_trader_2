#!/usr/bin/env python
# Phase State Diagnostic — print, add trades, reset, set targets

import argparse, json, sys
from pathlib import Path

ROOT = Path(r"C:\ftmo_trader_2")
PHASES = ROOT / "phases"
sys.path.insert(0, str(PHASES))

from phase_manager import PhaseManager, get_phase_snapshot

def main():
    ap = argparse.ArgumentParser(description="Phase state diagnostic tool")
    ap.add_argument("--add", type=float, help="Record a trade PnL (e.g., --add 25.0 or --add -15.5)")
    ap.add_argument("--reset", action="store_true", help="Reset phase state (clears history)")
    ap.add_argument("--targets", type=str, help="Override phase trade targets, e.g. --targets 60,120")
    args = ap.parse_args()

    pm = PhaseManager()

    if args.reset:
        st = pm.reset()
        print("[OK] Reset phase state.")

    if args.targets:
        try:
            parts = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
            pm.set_targets(parts)
            print(f"[OK] Targets set -> {parts}")
        except Exception as e:
            print(f"[ERR] Invalid --targets: {e}")

    if args.add is not None:
        st = pm.add_trade(args.add)
        print(f"[OK] Recorded trade PnL={args.add:+.2f}")

    snap = get_phase_snapshot()
    print(json.dumps(snap, indent=2))

if __name__ == "__main__":
    main()