#!/usr/bin/env python
# Preview the effective runtime computed from the current phase state.
# This DOES NOT modify your live runner. It just prints and writes data\phase_runtime.json.

import sys, json
from pathlib import Path

ROOT = Path(r"C:\ftmo_trader_2")
PHASES = ROOT / "phases"
sys.path.insert(0, str(PHASES))

from phase_bind import compute_effective_runtime, write_runtime_json

def main():
    # Example GUI overrides (None by default). Leave all as None to use pure phase values.
    gui_overrides = {
        "min_conf": None,      # e.g., 0.03 to enforce a higher GUI floor
        "cooldown_s": None,    # e.g., 30 to request faster entries (phase will raise if needed)
        "max_positions": None, # e.g., 2 to be stricter than phase
    }
    eff = write_runtime_json(gui_overrides=gui_overrides)
    print(json.dumps(eff, indent=2))
    print(f"\n[OK] Wrote: {str(ROOT / 'data' / 'phase_runtime.json')}")

if __name__ == "__main__":
    main()