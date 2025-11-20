import json
from pathlib import Path
from typing import Dict, Any, List, Optional


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "ai" / "ai_profile.json"
STATE = ROOT / "ai" / "auto_tuner_state.json"


def load_json(path: Path) -> Any:
    try:
        if not path.exists():
            print(f"[dump_trade_modes][warn] {path} does not exist.")
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[dump_trade_modes][error] failed to load {path}: {e!r}")
        return None


def main() -> None:
    profile = load_json(PROFILE)
    state = load_json(STATE) or {}

    if not isinstance(profile, dict):
        print(f"[dump_trade_modes][error] ai_profile.json is not a dict at top level.")
        return

    weak_streak = state.get("weak_streak", {}) if isinstance(state, dict) else {}
    status = state.get("status", {}) if isinstance(state, dict) else {}

    symbols: List[str] = sorted(profile.keys())

    print(f"[dump_trade_modes] Profile: {PROFILE}")
    print(f"[dump_trade_modes] State:   {STATE}")
    print()
    print(f"{'SYMBOL':<15} {'EN':<3} {'MODE':<6} {'RISK%':>6} {'TF':<6} {'LONG':>6} {'SHORT':>6} {'BUCKET':<8} {'W_STREAK':>9}")
    print("-" * 80)

    for sym in symbols:
        cfg = profile.get(sym, {})
        if not isinstance(cfg, dict):
            continue

        enabled = bool(cfg.get("enabled", True))
        mode = str(cfg.get("trade_mode", "live")).strip().lower() or "live"

        risk = cfg.get("risk", {})
        atr_pct = None
        if isinstance(risk, dict):
            if risk.get("mode") == "ATR_RISK":
                atr_pct = risk.get("atr_risk_pct")
        if atr_pct is None:
            atr_pct_str = "  -"
        else:
            atr_pct_str = f"{float(atr_pct):4.2f}"

        # Assume main TF is H1 if present, else any
        timeframes = cfg.get("timeframes", {})
        tf_name = ""
        long_thr = ""
        short_thr = ""
        if isinstance(timeframes, dict) and timeframes:
            if "H1" in timeframes:
                tf_name = "H1"
                tf_cfg = timeframes["H1"]
            else:
                # pick first
                tf_name, tf_cfg = next(iter(timeframes.items()))
            if isinstance(tf_cfg, dict):
                lt = tf_cfg.get("long_threshold")
                st = tf_cfg.get("short_threshold")
                long_thr = f"{float(lt):4.2f}" if lt is not None else ""
                short_thr = f"{float(st):4.2f}" if st is not None else ""
        tf_name = tf_name or ""

        bucket = status.get(sym, "")
        ws_val = weak_streak.get(sym, 0)
        try:
            ws_int = int(ws_val)
        except Exception:
            ws_int = 0

        print(
            f"{sym:<15} "
            f"{('Y' if enabled else 'N'):>1}  "
            f"{mode[:6]:<6} "
            f"{atr_pct_str:>6} "
            f"{tf_name:<6} "
            f"{long_thr:>6} "
            f"{short_thr:>6} "
            f"{str(bucket)[:8]:<8} "
            f"{ws_int:>9}"
        )


if __name__ == "__main__":
    main()