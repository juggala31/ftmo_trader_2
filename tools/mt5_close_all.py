#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/mt5_close_all.py
Safely close ALL open MT5 positions (netting or hedging) using the Python MetaTrader5 API.

Safety:
  • Requires EXECUTE=1 in your .env or environment to actually close.
  • Otherwise prints a dry-run summary and exits.

Usage:
  python C:\ftmo_trader_2\tools\mt5_close_all.py
"""

from __future__ import annotations
import os, sys

# ---------- tiny .env loader (same style as runner) ----------
def load_env(dotenv_path: str):
    if os.path.isfile(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass

def mt5_mod():
    try:
        import MetaTrader5 as mt5
        return mt5
    except Exception as e:
        print(f"[ERR] MetaTrader5 module not available: {e}")
        sys.exit(1)

def ensure_session(mt5):
    # Try to bind to a running terminal session (preferred)
    try:
        if mt5.initialize():
            return True
    except Exception:
        pass
    # Last error report
    try:
        code, msg = mt5.last_error()
        print(f"[WARN] mt5.initialize() failed: {code} {msg}")
    except Exception:
        pass
    return False

def fmt_side(mt5, pos_type):
    try:
        return "BUY" if pos_type == mt5.POSITION_TYPE_BUY else "SELL"
    except Exception:
        return str(pos_type)

def opposite_order_type(mt5, pos_type):
    # For BUY position we send a SELL order, and vice versa.
    try:
        return mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    except Exception:
        return None

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_env(os.path.join(root, ".env"))
    execute = os.environ.get("EXECUTE", "0") in ("1","true","TRUE","yes","YES")

    mt5 = mt5_mod()
    if not ensure_session(mt5):
        code, msg = mt5.last_error()
        print(f"[ERR] Could not initialize MT5 session: {code} {msg}")
        sys.exit(2)

    acc = mt5.account_info()
    if acc:
        print(f"[INFO] MT5 account: {getattr(acc,'login','?')} @ {getattr(acc,'server','?')} (mode=NETTING)")
    else:
        print("[WARN] account_info() unavailable — is terminal logged in?")

    positions = mt5.positions_get()
    if positions is None:
        code, msg = mt5.last_error()
        print(f"[ERR] positions_get() failed: {code} {msg}")
        sys.exit(3)

    if len(positions) == 0:
        print("[OK] No open positions to close.")
        return

    print("TICKET       SYMBOL           SIDE   VOL       PRICE_OPEN   PROFIT")
    print("-----------  ---------------  -----  -------   ----------   ----------")
    for p in positions:
        print(f"{getattr(p,'ticket',''):<11}  {getattr(p,'symbol',''):<15}  "
              f"{fmt_side(mt5, getattr(p,'type','')):<5}  {getattr(p,'volume',0.0):<7.2f}   "
              f"{getattr(p,'price_open',0.0):<10.2f}   {float(getattr(p,'profit',0.0)):<10.2f}")

    if not execute:
        print("[DRY] EXECUTE!=1 — would close all positions. Set EXECUTE=1 to enable.")
        return

    print("[INFO] EXECUTE=1 — sending close orders...")
    ok_all = True

    for p in positions:
        symbol   = getattr(p, "symbol", "")
        volume   = float(getattr(p, "volume", 0.0))
        pos_type = getattr(p, "type", None)
        ticket   = getattr(p, "ticket", None)

        order_type = opposite_order_type(mt5, pos_type)
        if order_type is None:
            print(f"[ERR] Unknown position type for ticket {ticket}; skip.")
            ok_all = False
            continue

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "deviation": 20,
            "magic": 0,
            "comment": "close_all",
            "type_filling": mt5.ORDER_FILLING_IOC,
            # For hedging accounts, position field is required; for netting it is ignored.
            "position": ticket,
        }

        result = mt5.order_send(req)
        retcode = getattr(result, "retcode", None)
        order   = getattr(result, "order", None)
        price   = getattr(result, "price", None)
        comment = getattr(result, "comment", "")

        if retcode == mt5.TRADE_RETCODE_DONE or retcode == mt5.TRADE_RETCODE_PLACED:
            print(f"[CLOSED] ticket={ticket} {symbol} vol={volume} via order={order} price={price}")
        else:
            print(f"[FAIL ] ticket={ticket} {symbol} vol={volume} retcode={retcode} comment={comment}")
            ok_all = False

    if ok_all:
        print("[OK] All positions closed (or orders placed to close).")
        sys.exit(0)
    else:
        print("[WARN] Some positions failed to close; see messages above.")
        sys.exit(4)

if __name__ == "__main__":
    main()
