#!/usr/bin/env python
import os, sys, math, atexit
from pathlib import Path

_DIAG = os.getenv("BT_DIAG","0") not in ("0","false","False","")

def _clamp01(x):
    try: return max(0.0, min(1.0, float(x)))
    except Exception: return 0.0

def _side_from_score(score):
    try:
        s = float(score)
        return ("BUY", _clamp01(1/(1+math.exp(-s)))) if s >= 0 else ("SELL", _clamp01(1/(1+math.exp(s))))
    except Exception:
        return (None, 0.0)

# usage counters
_used = {"get_signal":0,"live_signal":0,"decide":0,"decide_from_close":0,"predict":0,"fallback_demo":0}
_used_name = None

def _print_summary():
    if not _DIAG: return
    total = sum(_used.values())
    parts = [f"{k}={v}" for k,v in _used.items()]
    if total == 0 or (_used["fallback_demo"] == total):
        print("[bt/live] summary: FELL BACK TO DEMO  " + "  ".join(parts))
    else:
        print("[bt/live] summary: LIVE PATH USED     " + "  ".join(parts))

# Try to import live module
_live = None
try:
    ROOT = Path(r"C:\ftmo_trader_2")
    sys.path.insert(0, str(ROOT / "ai"))
    import alpha_loop as _live
    if _DIAG: print("[bt/live] loaded ai.alpha_loop")
except Exception as e:
    if _DIAG: print(f"[bt/live] failed to load ai.alpha_loop: {e}")
    _live = None

# Fallback demo
try:
    sys.path.insert(0, str(ROOT / "strategy"))
    from signal_entry import get_signal as _demo_get_signal
except Exception:
    _demo_get_signal = lambda row, st: (None, 0.0)

def get_signal(row, st):
    global _used_name
    if _live is None:
        _used["fallback_demo"] += 1; _used_name = _used_name or "fallback_demo"
        return _demo_get_signal(row, st)

    o,h,l,c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

    # 1) get_signal(row, state)
    try:
        if hasattr(_live, "get_signal"):
            side, conf = _live.get_signal(row, st)  # type: ignore
            if side in ("BUY","SELL"):
                _used["get_signal"] += 1; _used_name = _used_name or "get_signal"
                return side, _clamp01(conf)
    except Exception: pass

    # 2) live_signal(row)
    try:
        if hasattr(_live, "live_signal"):
            side, conf = _live.live_signal(row)  # type: ignore
            if side in ("BUY","SELL"):
                _used["live_signal"] += 1; _used_name = _used_name or "live_signal"
                return side, _clamp01(conf)
    except Exception: pass

    # 3) decide(o,h,l,c, state)
    try:
        if hasattr(_live, "decide"):
            side, conf = _live.decide(o,h,l,c, st)  # type: ignore
            if side in ("BUY","SELL"):
                _used["decide"] += 1; _used_name = _used_name or "decide"
                return side, _clamp01(conf)
    except Exception: pass

    # 4) decide_from_close(c, state)
    try:
        if hasattr(_live, "decide_from_close"):
            side, conf = _live.decide_from_close(c, st)  # type: ignore
            if side in ("BUY","SELL"):
                _used["decide_from_close"] += 1; _used_name = _used_name or "decide_from_close"
                return side, _clamp01(conf)
    except Exception: pass

    # 5) predict(c) -> score
    try:
        if hasattr(_live, "predict"):
            score = _live.predict(c)  # type: ignore
            _used["predict"] += 1; _used_name = _used_name or "predict"
            return _side_from_score(score)
    except Exception: pass

    _used["fallback_demo"] += 1; _used_name = _used_name or "fallback_demo"
    return _demo_get_signal(row, st)

if _DIAG:
    print("[bt/live] adapter ready. Summary will print at exit.")
    atexit.register(_print_summary)