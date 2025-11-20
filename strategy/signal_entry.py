#!/usr/bin/env python
"""
Shared strategy signal for both live and backtest.

Default: simple EMA/RSI hybrid with confidence in [0..1].
You can replace the logic here to mirror your live decision function.
"""

from math import isnan

def _ema(prev, price, k):
    return price if prev is None else (prev + k * (price - prev))

def _rsi_update(state, close, period=14):
    """
    Wilder's RSI (incremental). Keeps state in-place.
    state keys: rsi, avg_up, avg_dn, prev_close, init_cnt
    """
    pc = state.get("prev_close")
    if pc is None:
        state["prev_close"] = close
        state["init_cnt"] = 0
        return None
    chg = close - pc
    up = max(chg, 0.0)
    dn = max(-chg, 0.0)
    if state.get("init_cnt", 0) < period:
        state["avg_up"] = (state.get("avg_up", 0.0) * state.get("init_cnt", 0) + up) / (state.get("init_cnt", 0) + 1)
        state["avg_dn"] = (state.get("avg_dn", 0.0) * state.get("init_cnt", 0) + dn) / (state.get("init_cnt", 0) + 1)
        state["init_cnt"] = state.get("init_cnt", 0) + 1
        state["rsi"] = None
    else:
        au = state.get("avg_up", 0.0) * (period - 1) / period + up / period
        ad = state.get("avg_dn", 0.0) * (period - 1) / period + dn / period
        state["avg_up"], state["avg_dn"] = au, ad
        rs = (au / ad) if ad > 1e-12 else 0.0
        state["rsi"] = 100.0 - (100.0 / (1.0 + rs)) if ad > 1e-12 else 100.0
    state["prev_close"] = close
    return state.get("rsi")

def get_signal(row, st):
    """
    Args:
      row: dict with keys ['time','open','high','low','close']
      st:  dict (per-symbol persistent state) that we may read/write

    Returns:
      (side:str|None, confidence:float)
      side in {"BUY","SELL",None}
    """
    c = float(row["close"])

    # --- EMA cross (12/26) ---
    ema12_k = 2.0/(12+1)
    ema26_k = 2.0/(26+1)
    st["ema12"] = _ema(st.get("ema12"), c, ema12_k)
    st["ema26"] = _ema(st.get("ema26"), c, ema26_k)
    ema_sig = 0
    if st["ema12"] is not None and st["ema26"] is not None:
        ema_sig = 1 if st["ema12"] > st["ema26"] else -1

    # --- RSI (14) ---
    rsi = _rsi_update(st.setdefault("rsi_state", {}), c, 14)
    rsi_sig = 0
    if rsi is not None:
        if rsi < 35: rsi_sig = 1
        elif rsi > 65: rsi_sig = -1

    # Combine
    raw = ema_sig + rsi_sig
    if raw > 0:
        side = "BUY"
        confidence = min(1.0, 0.5 + 0.25*(ema_sig>0) + 0.25*(rsi_sig>0))
    elif raw < 0:
        side = "SELL"
        confidence = min(1.0, 0.5 + 0.25*(ema_sig<0) + 0.25*(rsi_sig<0))
    else:
        side, confidence = None, 0.0

    return side, confidence