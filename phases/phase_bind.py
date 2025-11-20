#!/usr/bin/env python
"""
Phase -> Runtime policy binding (non-invasive).
Computes effective MIN_CONF, COOLDOWN, MAX_POS from current PhaseState.
Can be used by the live controller or any tool.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(r"C:\ftmo_trader_2")
PHASES_DIR = ROOT / "phases"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import the phase engine
import sys
sys.path.insert(0, str(PHASES_DIR))
from phase_manager import PhaseManager, get_phase_snapshot

RUNTIME_PATH = DATA_DIR / "phase_runtime.json"

def compute_effective_runtime(gui_overrides=None):
    """
    Return a dict with effective runtime settings:
      - min_conf
      - cooldown_s
      - max_positions
    If gui_overrides is provided, it should be a dict like:
      { "min_conf": float|None, "cooldown_s": int|None, "max_positions": int|None }
    We take the 'safest' of both (e.g., max(min_conf_phase, min_conf_gui)).
    """
    snap = get_phase_snapshot()
    # Phase-derived
    p_min = float(snap["min_conf"])
    p_cd  = int(snap["cooldown_s"])
    p_mp  = int(snap["max_positions"])

    g_min = g_cd = g_mp = None
    if isinstance(gui_overrides, dict):
        g_min = gui_overrides.get("min_conf")
        g_cd  = gui_overrides.get("cooldown_s")
        g_mp  = gui_overrides.get("max_positions")

    # Combine (prefer safer). If GUI leaves None, use phase value.
    eff_min = max(p_min, float(g_min)) if g_min is not None else p_min
    eff_cd  = max(p_cd, int(g_cd))     if g_cd  is not None else p_cd
    eff_mp  = min(p_mp, int(g_mp))     if g_mp  is not None else p_mp  # cap by the stricter max

    return {
        "phase_name": snap["phase_name"],
        "total_trades": snap["total_trades"],
        "progress_pct": snap["progress_pct"],
        "trades_remaining": snap["trades_remaining"],
        "wr_recent": snap["wr_recent"],
        "pf_recent": snap["pf_recent"],
        "min_conf": eff_min,
        "cooldown_s": eff_cd,
        "max_positions": eff_mp,
    }

def write_runtime_json(gui_overrides=None, path=RUNTIME_PATH):
    """Compute and persist the effective runtime so other processes can read it."""
    eff = compute_effective_runtime(gui_overrides=gui_overrides)
    path.write_text(json.dumps(eff, indent=2), encoding="utf-8")
    return eff

if __name__ == "__main__":
    eff = write_runtime_json()
    print(json.dumps(eff, indent=2))