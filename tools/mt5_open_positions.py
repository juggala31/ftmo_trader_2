#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/mt5_open_positions.py
List current MT5 open positions (ticket, symbol, side, volume, price, profit).
- Pure stdlib + MetaTrader5 (no extra packages)
- Uses your already-logged-in MT5 terminal session if available
"""

from __future__ import annotations
import sys, time

def mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except Exception as e:
        print(f"[ERR] MetaTrader5 module not available: {e}")
        sys.exit(1)

def ensure_session(mt5):
    # Reuse existing terminal session if possible
    if mt5.initialize():
        return True
    try:
        # Last-resort: try init without args (will bind to running terminal)
        return bool(mt5.initialize())
    except Exception:
        return False

def fmt_side(typ):
    # MT5 position_type: 0=BUY, 1=SELL
    try:
        import MetaTrader5 as m
        if typ == m.POSITION_TYPE_BUY:  return "BUY"
        if typ == m.POSITION_TYPE_SELL: return "SELL"
    except Exception:
        pass
    return str(typ)

def main():
    m = mt5()
    if not ensure_session(m):
        code, msg = m.last_error()
        print(f"[ERR] Could not initialize MT5: {code} {msg}")
        sys.exit(1)

    acc = m.account_info()
    if acc:
        print(f"[INFO] MT5 account: {getattr(acc,'login','?')} @ {getattr(acc,'server','?')}  (name={getattr(acc,'name','?')})")
    else:
        print("[WARN] No account_info() — is the terminal logged in?")

    positions = m.positions_get()
    if positions is None:
        code, msg = m.last_error()
        print(f"[ERR] positions_get() failed: {code} {msg}")
        sys.exit(2)

    if len(positions) == 0:
        print("[OK] No open positions.")
        sys.exit(0)

    # Header
    print("TICKET       SYMBOL           SIDE   VOL       PRICE        PROFIT      COMMENT")
    print("-----------  ---------------  -----  -------   ----------   ----------  ---------------------")
    total_profit = 0.0
    for p in positions:
        ticket = getattr(p, "ticket", "")
        symbol = getattr(p, "symbol", "")
        side   = fmt_side(getattr(p, "type", ""))
        vol    = getattr(p, "volume", 0.0)
        price  = getattr(p, "price_open", 0.0)
        profit = float(getattr(p, "profit", 0.0))
        comment= getattr(p, "comment", "")
        total_profit += profit
        print(f"{ticket:<11}  {symbol:<15}  {side:<5}  {vol:<7.2f}   {price:<10.2f}   {profit:<10.2f}  {comment}")

    print("-----------  ---------------  -----  -------   ----------   ----------  ---------------------")
    print(f"TOTAL PROFIT: {total_profit:.2f}")

if __name__ == "__main__":
    main()
