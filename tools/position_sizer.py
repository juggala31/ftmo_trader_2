#!/usr/bin/env python3
"""
tools/position_sizer.py

Compute position size (lots) from:
- risk % of account balance, and
- stop distance in price units (absolute, e.g., ATR-based SL distance)

Formula (for 1 lot):
  monetary_per_point = tick_value * (point / tick_size)
  points_to_sl = stop_distance / point
  PnL_at_SL_for_1_lot ≈ points_to_sl * monetary_per_point
  lots = risk_amount / PnL_at_SL_for_1_lot
"""

from __future__ import annotations
import os, argparse
from typing import Optional, Tuple

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

def _ensure_mt5():
    if not mt5:
        return False
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
    return True

def _account_balance_fallback() -> Optional[float]:
    try:
        acc = mt5.account_info()
        if acc and hasattr(acc, "balance"):
            return float(acc.balance)
    except Exception:
        pass
    try:
        v = os.environ.get("ACCOUNT_BALANCE")
        return float(v) if v else None
    except Exception:
        return None

def _normalize_volume(symbol: str, lots: float) -> float:
    try:
        info = mt5.symbol_info(symbol)
        if not info:
            return max(0.01, round(lots, 2))
        vol_min = float(getattr(info, "volume_min", 0.0) or 0.0)
        vol_max = float(getattr(info, "volume_max", 0.0) or 1000000.0)
        vol_step = float(getattr(info, "volume_step", 0.01) or 0.01)
        lots = max(vol_min, min(vol_max, lots))
        if vol_step > 0:
            steps = round(lots / vol_step)
            lots = steps * vol_step
        return float(lots)
    except Exception:
        return max(0.01, round(lots, 2))

def compute_lot(symbol: str, stop_distance: float, risk_pct: float,
                default_lot: float = 0.10) -> Tuple[float, dict]:
    diag = {"mode":"fallback", "balance": None, "point": None, "tick_value": None, "tick_size": None,
            "monetary_per_point": None, "points_to_sl": None, "risk_amount": None}

    if stop_distance is None or stop_distance <= 0:
        return default_lot, {**diag, "reason":"invalid_stop_distance"}

    if not _ensure_mt5():
        return default_lot, {**diag, "reason":"no_mt5"}

    balance = _account_balance_fallback()
    if balance is None or balance <= 0:
        return default_lot, {**diag, "reason":"no_balance"}

    try:
        info = mt5.symbol_info(symbol)
    except Exception:
        info = None
    if not info:
        return default_lot, {**diag, "reason":"no_symbol"}

    point = float(getattr(info, "point", 0.0) or 0.0)
    tick_value = float(getattr(info, "trade_tick_value", getattr(info, "tick_value", 0.0)) or 0.0)
    tick_size  = float(getattr(info, "trade_tick_size", getattr(info, "tick_size", 0.0)) or 0.0)

    diag.update({"balance": balance, "point": point, "tick_value": tick_value, "tick_size": tick_size})

    if point <= 0 or tick_value <= 0 or tick_size <= 0:
        return _normalize_volume(symbol, default_lot), {**diag, "reason":"bad_symbol_params"}

    risk_amount = balance * (float(risk_pct) / 100.0)
    diag["risk_amount"] = risk_amount

    monetary_per_point = tick_value * (point / tick_size)
    diag["monetary_per_point"] = monetary_per_point
    if monetary_per_point <= 0:
        return _normalize_volume(symbol, default_lot), {**diag, "reason":"zero_monetary_per_point"}

    points_to_sl = stop_distance / point
    diag["points_to_sl"] = points_to_sl

    pnl_at_sl_for_1_lot = points_to_sl * monetary_per_point
    if pnl_at_sl_for_1_lot <= 0:
        return _normalize_volume(symbol, default_lot), {**diag, "reason":"zero_pnl_per_lot"}

    lots = risk_amount / pnl_at_sl_for_1_lot
    lots_n = _normalize_volume(symbol, lots)
    return lots_n, {**diag, "mode":"sized", "raw_lots": lots, "lots_norm": lots_n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--risk-pct", type=float, required=True)
    ap.add_argument("--stop-dist", type=float, required=True)
    ap.add_argument("--default-lot", type=float, default=0.10)
    args = ap.parse_args()

    lots, diag = compute_lot(args.symbol, stop_distance=args.stop_dist, risk_pct=args.risk_pct, default_lot=args.default_lot)
    print(f"[size] symbol={args.symbol} stop={args.stop_dist} risk%={args.risk_pct} -> lots={lots}")
    print(f"[diag] {diag}")

if __name__ == "__main__":
    main()