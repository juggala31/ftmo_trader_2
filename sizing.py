from __future__ import annotations
from typing import Dict, Tuple

def compute_sl_tp_points(*, side: str, bid: float, ask: float, use_atr: bool,
                         atr_points: float, atr_mult_sl: float, atr_mult_tp: float,
                         fixed_sl_points: float|None, fixed_tp_points: float|None) -> Tuple[float|None, float|None]:
    if use_atr and atr_points and atr_points > 0:
        slp = float(atr_points) * float(atr_mult_sl)
        tpp = float(atr_points) * float(atr_mult_tp)
        return (slp, tpp)
    # fallback to fixed points
    return (float(fixed_sl_points) if fixed_sl_points else None,
            float(fixed_tp_points) if fixed_tp_points else None)

def compute_volume_from_risk(*, equity_ccy: float, risk_pct: float,
                             stop_points: float, vpp_per_unit: float,
                             min_v: float, max_v: float, step: float) -> float:
    """
    Risk model:
      position value at 1.0 volume moves vpp_per_unit (account ccy) per 1.0 price point.
      If stop distance = stop_points, risk_ccy at volume=V is stop_points * vpp_per_unit * V
      => V = (equity * risk_pct) / (stop_points * vpp_per_unit)
    """
    if stop_points <= 0 or vpp_per_unit <= 0:
        return min_v
    raw = (equity_ccy * (risk_pct/100.0)) / (stop_points * vpp_per_unit)
    # snap to step and clamp
    if step <= 0: step = 0.01
    snapped = round(round(raw / step) * step, 8)
    return max(min_v, min(snapped, max_v))