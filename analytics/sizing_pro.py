from __future__ import annotations
import pathlib, sys
from typing import Dict, Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
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
    if not p.exists(): return {}
    if yaml:
        try: return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception: return {}
    return {}

def _point_size(settings, symbol: str) -> float:
    try:
        return float((settings.sizing.get("points") or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

def _vpp(settings, symbol: str) -> float:
    try:
        return float((settings.vpp or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

def _tf_const(tf: str):
    if not mt5: return None
    m = {
        "M1":"TIMEFRAME_M1","M5":"TIMEFRAME_M5","M15":"TIMEFRAME_M15",
        "M30":"TIMEFRAME_M30","H1":"TIMEFRAME_H1","H4":"TIMEFRAME_H4","D1":"TIMEFRAME_D1"
    }
    return getattr(mt5, m.get(tf.upper(),"TIMEFRAME_M15"), None)

def _atr_points(symbol: str, tf: str, period: int, pt: float) -> float:
    if not mt5: return 0.0
    c = _tf_const(tf)
    if c is None: return 0.0
    bars = mt5.copy_rates_from_pos(symbol, c, 0, max(period*5, period+20)) or []
    if not bars: return 0.0
    import pandas as pd, numpy as np
    h = pd.Series([b["high"] for b in bars], dtype="float64")
    l = pd.Series([b["low"] for b in bars], dtype="float64")
    c_ = pd.Series([b["close"] for b in bars], dtype="float64")
    prev = c_.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
    return float(atr) / (pt if pt>0 else 1.0)

def _round_step(x: float, step: float) -> float:
    return round(x/step)*step if step>0 else x

def wrap_adapter_with_sizing_pro(adapter, settings, cfg_path: str):
    cfg = _load_yaml(pathlib.Path(cfg_path))
    if not cfg.get("enabled", True): return
    rp = str(cfg.get("reason_prefix","SIZE"))
    vt = cfg.get("vol_target") or {}
    vt_on = bool(vt.get("enabled", False))
    vt_tf = str(vt.get("tf","M15"))
    vt_p  = int(vt.get("atr_period",14))
    vt_r  = float(vt.get("risk_per_trade_ccy",50.0))

    vmin = float(cfg.get("min_volume", 0.01))
    vmax = float(cfg.get("max_volume", 10.0))
    step = float(cfg.get("step", 0.01))

    base_place = adapter.place_order

    def place_sized(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        pt  = _point_size(settings, symbol)
        vpp = _vpp(settings, symbol)

        vol = float(volume)
        if vt_on and mt5:
            atr_pts = _atr_points(symbol, vt_tf, vt_p, pt)
            # naive risk proxy: atr_pts * vpp * vol ≈ risk_ccy -> solve vol
            if atr_pts > 0 and vpp > 0:
                vol = max(vmin, min(vmax, vt_r / (atr_pts * vpp)))
        # clamp + round
        vol = max(vmin, min(vmax, _round_step(vol, step)))
        return base_place(side=side, symbol=symbol, volume=vol, sl=sl, tp=tp, comment=comment)

    adapter.place_order = place_sized  # type: ignore