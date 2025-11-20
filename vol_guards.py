from __future__ import annotations
import time, pathlib, sys, datetime as dt
from typing import Dict, Any, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[0].parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    if yaml:
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    out: Dict[str, Any] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def _point_size(settings, symbol: str) -> float:
    try:
        return float((settings.sizing.get("points") or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

_TF_MAP = {
    "M1": getattr(mt5, "TIMEFRAME_M1", None) if mt5 else None,
    "M5": getattr(mt5, "TIMEFRAME_M5", None) if mt5 else None,
    "M15": getattr(mt5, "TIMEFRAME_M15", None) if mt5 else None,
    "M30": getattr(mt5, "TIMEFRAME_M30", None) if mt5 else None,
    "H1": getattr(mt5, "TIMEFRAME_H1", None) if mt5 else None,
    "H4": getattr(mt5, "TIMEFRAME_H4", None) if mt5 else None,
    "D1": getattr(mt5, "TIMEFRAME_D1", None) if mt5 else None,
}

_cache: Dict[Tuple[str,str], Tuple[float,float,float]] = {}
_cache_ts: Dict[Tuple[str,str], float] = {}

def _rolling_atr(highs, lows, closes, period: int = 14):
    import numpy as np
    import pandas as pd
    sH = pd.Series(highs, dtype="float64")
    sL = pd.Series(lows, dtype="float64")
    sC = pd.Series(closes, dtype="float64")
    prev = sC.shift(1)
    tr = pd.concat([(sH - sL).abs(), (sH - prev).abs(), (sL - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _get_adr_points(symbol: str, pt: float, days: int) -> float:
    """ADR in points from D1 data."""
    key = (symbol, f"ADR{days}")
    now = time.time()
    if key in _cache and now - _cache_ts.get(key, 0) < 300:
        return _cache[key][0]
    rates = mt5.copy_rates_from_pos(symbol, _TF_MAP["D1"], 0, max(days, days+2))
    if not rates:
        return 0.0
    rngs = [(float(r["high"]) - float(r["low"])) / (pt if pt > 0 else 1.0) for r in rates[-days:]]
    adr_pts = sum(rngs) / max(1, len(rngs))
    _cache[key] = (adr_pts, 0.0, 0.0); _cache_ts[key] = now
    return adr_pts

def _get_atr_points(symbol: str, pt: float, tf: str, period: int) -> float:
    """Current ATR (last value) in points on intraday TF."""
    key = (symbol, f"ATR{tf}{period}")
    now = time.time()
    if key in _cache and now - _cache_ts.get(key, 0) < 120:
        return _cache[key][1]
    tfc = _TF_MAP.get(tf.upper())
    if tfc is None:
        return 0.0
    bars = mt5.copy_rates_from_pos(symbol, tfc, 0, max(period*5, period+20))
    if not bars:
        return 0.0
    highs = [float(b["high"]) for b in bars]
    lows  = [float(b["low"]) for b in bars]
    closes= [float(b["close"]) for b in bars]
    atr_vals = _rolling_atr(highs, lows, closes, period=period)
    atr_last = float(atr_vals.iloc[-1]) if len(atr_vals) else 0.0
    atr_pts  = atr_last / (pt if pt > 0 else 1.0)
    # cache adr placeholder too (slot 0) to keep tuple shape
    prev_adr = _cache.get((symbol, f"ADR{period}"), (0.0,0.0,0.0))[0]
    _cache[key] = (prev_adr, atr_pts, 0.0); _cache_ts[key] = now
    return atr_pts

def _today_range_points(symbol: str, pt: float) -> float:
    """Current day's high-low in points."""
    key = (symbol, "TODR")
    now = time.time()
    if key in _cache and now - _cache_ts.get(key, 0) < 60:
        return _cache[key][2]
    # get last ~3 daily bars to be safe
    d1 = mt5.copy_rates_from_pos(symbol, _TF_MAP["D1"], 0, 3) or []
    if not d1:
        return 0.0
    today = d1[-1]
    rng = (float(today["high"]) - float(today["low"])) / (pt if pt > 0 else 1.0)
    _cache[key] = (0.0, 0.0, rng); _cache_ts[key] = now
    return rng

def wrap_adapter_with_vol_guards(adapter, settings, cfg_path: str, reason_prefix: str = "VOL"):
    """
    Enforce:
      - today_range_pts <= adr_max_multiple * ADR_pts
      - atr_pts <= atr_max_multiple * ADR_pts
    """
    if mt5 is None:
        return  # no MT5 = cannot evaluate, do nothing

    cfg = _load_yaml(pathlib.Path(cfg_path))
    adr_days = int(cfg.get("adr_days", 14))
    adr_mult = float(cfg.get("adr_max_multiple", 1.8))
    atr_tf   = str(cfg.get("atr_tf", "M15")).upper()
    atr_p    = int(cfg.get("atr_period", 14))
    atr_mult = float(cfg.get("atr_max_multiple", 2.5))

    base_place = adapter.place_order

    def guarded_place(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        pt = _point_size(settings, symbol)

        # compute ADR & today's range (points)
        try:
            adr_pts = _get_adr_points(symbol, pt, adr_days)
            today_pts = _today_range_points(symbol, pt)
        except Exception:
            adr_pts = 0.0; today_pts = 0.0

        if adr_pts > 0 and today_pts > adr_mult * adr_pts:
            return {"ok": False, "message": f"{reason_prefix}: dayRange {today_pts:.1f}pts>{adr_mult:.1f}×ADR({adr_pts:.1f}) on {symbol}"}

        # ATR check normalized to points vs ADR
        try:
            atr_pts = _get_atr_points(symbol, pt, atr_tf, atr_p)
        except Exception:
            atr_pts = 0.0

        if adr_pts > 0 and atr_pts > atr_mult * adr_pts:
            return {"ok": False, "message": f"{reason_prefix}: ATR({atr_tf}) {atr_pts:.1f}pts>{atr_mult:.1f}×ADR({adr_pts:.1f}) on {symbol}"}

        # pass through
        return base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)

    adapter.place_order = guarded_place  # type: ignore