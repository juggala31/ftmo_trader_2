#!/usr/bin/env python3
"""
tools/atr_helper.py
- Computes Wilder ATR from MT5 history (no extra packages).
- Exposes get_atr(symbol, period, timeframe, bars) for reuse.
- Small CLI for quick checks.

Usage examples:
  python tools/atr_helper.py --symbol XAUZ25.sim --timeframe M5 --period 14 --bars 300
"""

import os, argparse, math
from typing import Tuple, Optional

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# timeframe map (string -> MT5 const name)
_TF = {
    "M1":"TIMEFRAME_M1","M2":"TIMEFRAME_M2","M3":"TIMEFRAME_M3","M4":"TIMEFRAME_M4","M5":"TIMEFRAME_M5",
    "M10":"TIMEFRAME_M10","M12":"TIMEFRAME_M12","M15":"TIMEFRAME_M15","M20":"TIMEFRAME_M20","M30":"TIMEFRAME_M30",
    "H1":"TIMEFRAME_H1","H2":"TIMEFRAME_H2","H3":"TIMEFRAME_H3","H4":"TIMEFRAME_H4","H6":"TIMEFRAME_H6",
    "H8":"TIMEFRAME_H8","H12":"TIMEFRAME_H12","D1":"TIMEFRAME_D1","W1":"TIMEFRAME_W1","MN1":"TIMEFRAME_MN1"
}

def _mt5_tf(name: str):
    if not mt5:
        return None
    key = _TF.get(name.upper(), "TIMEFRAME_M5")
    return getattr(mt5, key, mt5.TIMEFRAME_M5)

def _true_range(h, l, c_prev) -> float:
    return max(h - l, abs(h - c_prev), abs(l - c_prev))

def _wilder_atr(tr_list, period: int) -> float:
    """
    Wilder RMA on True Range:
      ATR_0 = SMA(TR, period)
      ATR_t = ( (ATR_{t-1}*(period-1)) + TR_t ) / period
    """
    if len(tr_list) < period:
        return float("nan")
    atr = sum(tr_list[:period]) / float(period)  # seed SMA
    for tr in tr_list[period:]:
        atr = ((atr * (period - 1)) + tr) / float(period)
    return float(atr)

def get_atr(symbol: str, period: int = 14, timeframe: str = "M5", bars: int = 300) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (atr_value, last_close). ATR is in absolute price units (not pips).
    """
    if not mt5:
        return None, None

    tf = _mt5_tf(timeframe)
    if tf is None:
        return None, None

    # Ensure terminal is ready
    try:
        if mt5.terminal_info() is None:
            kw = {}
            term = os.environ.get("MT5_TERMINAL_PATH") or os.environ.get("TERMINAL_PATH")
            if term: kw["path"] = term
            login  = os.environ.get("MT5_LOGIN")
            pwd    = os.environ.get("MT5_PASSWORD")
            server = os.environ.get("MT5_SERVER")
            if login and pwd and server:
                mt5.initialize(login=int(login), password=pwd, server=server, **kw)
            else:
                mt5.initialize(**kw)
    except Exception:
        pass

    try:
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) < (period + 1):
            return None, None
        trs = []
        for i in range(1, len(rates)):
            h = float(rates[i]["high"])
            l = float(rates[i]["low"])
            c_prev = float(rates[i-1]["close"])
            trs.append(_true_range(h, l, c_prev))
        atr = _wilder_atr(trs, int(period))
        last_close = float(rates[-1]["close"])
        if atr is None or math.isnan(atr):
            return None, last_close
        return float(atr), last_close
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--period", type=int, default=14)
    ap.add_argument("--timeframe", type=str, default="M5", help="M1,M5,M15,M30,H1,H4,D1,...")
    ap.add_argument("--bars", type=int, default=300)
    args = ap.parse_args()

    atr, last_close = get_atr(args.symbol, period=args.period, timeframe=args.timeframe, bars=args.bars)
    print(f"[atr] symbol={args.symbol} tf={args.timeframe} period={args.period} bars={args.bars} -> atr={atr} last_close={last_close}")

if __name__ == "__main__":
    main()