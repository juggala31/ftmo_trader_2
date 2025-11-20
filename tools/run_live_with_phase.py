#!/usr/bin/env python
import os, sys, subprocess, json
from pathlib import Path

ROOT   = Path(r"C:\ftmo_trader_2")
PHASES = ROOT / "phases"
DATA   = ROOT / "data"
sys.path.insert(0, str(PHASES))

from phase_bind import write_runtime_json

def export_env(eff):
    # Phase-driven values
    min_conf = str(eff.get("min_conf", 0.0))
    cooldown = str(int(eff.get("cooldown_s", 0)))
    maxpos   = str(int(eff.get("max_positions", 1)))

    # Disable force-once helper (kills the threading error)
    os.environ["AI_FORCE_ONCE"] = "0"
    os.environ["AI_FORCE_SIDE"] = ""
    os.environ["AI_FORCE_VOL"]  = ""

    # Min-conf (cover many aliases)
    for k in [
        "MIN_CONF","MIN_CONF_FLOOR","AI_MIN_CONF","SIGNAL_MIN_CONF",
        "CONFIDENCE_FLOOR","PRED_CONF_MIN"
    ]:
        os.environ[k] = min_conf

    # Cooldown seconds (cover many aliases)
    for k in [
        "COOLDOWN_S","COOLDOWN","ORDER_COOLDOWN_S","HEARTBEAT_COOLDOWN_S",
        "HB_COOLDOWN_S","GUI_COOLDOWN_S","CONTROLLER_COOLDOWN_S"
    ]:
        os.environ[k] = cooldown

    # Max concurrent positions (cover many aliases)
    for k in [
        "MAX_POSITIONS","MAX_POS","MAX_CONCURRENT","MAX_CONCURRENT_TRADES",
        "MAX_OPEN_TRADES","MAX_INFLIGHT"
    ]:
        os.environ[k] = maxpos

    # For logging/telemetry
    os.environ["PHASE_NAME"]          = eff.get("phase_name","")
    os.environ["PHASE_TOTAL_TRADES"]  = str(eff.get("total_trades",0))
    os.environ["PHASE_PROGRESS_PCT"]  = str(eff.get("progress_pct",0.0))

def main():
    eff = write_runtime_json()
    print("[phase] effective runtime:")
    print(json.dumps(eff, indent=2))
    export_env(eff)

    run_live = str(ROOT / "run_live.py")
    if not Path(run_live).exists():
        print(f"[ERR] not found: {run_live}")
        sys.exit(1)

    cmd = [sys.executable, run_live]
    print(f"[run] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=os.environ.copy())
    proc.wait()
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()