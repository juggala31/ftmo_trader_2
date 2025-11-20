#!/usr/bin/env python3
r"""
tools/send_order.py — minimal manual order sender using your MT5 adapter.
Version tag: SO-OK-10009
- Treats MT5 retcode==10009 ("Request executed") as success even if 'ok' missing.
- If no ticket returned, finds the position by symbol and applies SL/TP anyway.
"""

import os, sys, argparse, json, traceback
from pathlib import Path

# --- ensure project root on sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

Adapter = None
try:
    from brokers.mt5_adapter import MT5Adapter as _Adapter
    Adapter = _Adapter
except Exception as e:
    print(f"[brokers] adapter import failed: {e}")

def getenv_float(name, default):
    try: return float(os.environ.get(name, default))
    except Exception: return float(default)

def getenv_bool(name, default):
    v = os.environ.get(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _symbol_point(symbol: str) -> float:
    try:
        info = mt5.symbol_info(symbol) if mt5 else None
        if info and info.point > 0:
            return float(info.point)
    except Exception:
        pass
    return 0.01

def _compute_sltp_fixed(side: str, entry_price: float, symbol: str, pips: float, tp_mult: float):
    p = _symbol_point(symbol)
    sl_dist = pips * p
    tp_dist = sl_dist * tp_mult
    if side.upper() == "BUY":
        return round(entry_price - sl_dist, 5), round(entry_price + tp_dist, 5)
    else:
        return round(entry_price + sl_dist, 5), round(entry_price - tp_dist, 5)

def _find_position(symbol: str, ticket):
    if not mt5:
        return None
    try:
        pos_list = mt5.positions_get(symbol=symbol)
        if not pos_list:
            return None
        if ticket is not None:
            for p in pos_list:
                try:
                    if int(p.ticket) == int(ticket):
                        return p
                except Exception:
                    pass
        try:
            return max(pos_list, key=lambda p: getattr(p, "time_msc", 0))
        except Exception:
            return pos_list[0]
    except Exception:
        return None

def _apply_sltp(ticket, symbol: str, side: str, entry_price, pips: float, tp_mult: float):
    if not mt5:
        return
    pos = _find_position(symbol, ticket)
    if not pos:
        print("[sltp] no position found for SL/TP")
        return
    base_price = float(entry_price) if entry_price else float(getattr(pos, "price_open", 0.0))
    if base_price <= 0:
        print("[sltp] invalid base price")
        return
    sl, tp = _compute_sltp_fixed(side, base_price, symbol, pips, tp_mult)
    try:
        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(pos.ticket),
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
            "magic": 777123,
            "comment": "send_order_sltp",
        }
        res = mt5.order_send(req)
        print(f"[sltp] ret={getattr(res,'retcode',None)} sl={sl} tp={tp} comment={getattr(res,'comment',None)}")
    except Exception as e:
        print(f"[sltp] error: {e}")

def _retcode_ok(retcode, comment):
    try:
        if retcode == 10009:  # TRADE_RETCODE_DONE
            return True
    except Exception:
        pass
    if comment and "Request executed" in str(comment):
        return True
    return False

def main():
    print("[version] send_order.py SO-OK-10009")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", required=True, choices=["BUY","SELL","buy","sell"])
    ap.add_argument("--volume", required=True, type=float)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exec_gate = getenv_bool("EXECUTE", False)
    sltp_mode = os.environ.get("SLTP_MODE", "FIXED").upper()
    fixed_pips = getenv_float("FIXED_PIPS", 100.0)
    tp_mult   = getenv_float("TP_MULT", 2.0)

    # MT5 ready for price/SLTP
    if mt5:
        try:
            if mt5.terminal_info() is None:
                kw = {}
                term = os.environ.get("MT5_TERMINAL_PATH") or os.environ.get("TERMINAL_PATH")
                if term: kw["path"] = term
                login  = os.environ.get("MT5_LOGIN"); pwd = os.environ.get("MT5_PASSWORD"); server = os.environ.get("MT5_SERVER")
                ok = mt5.initialize(login=int(login), password=pwd, server=server, **kw) if (login and pwd and server) else mt5.initialize(**kw)
                print(f"[mt5] initialize -> {ok}")
            mt5.symbol_select(args.symbol, True)
        except Exception as e:
            print(f"[mt5] init error: {e}")

    # snapshot price
    price = None
    if mt5:
        try:
            tick = mt5.symbol_info_tick(args.symbol)
            price = (tick.ask if args.side.upper()=="BUY" else tick.bid) if tick else None
        except Exception:
            price = None

    print(f"[attempt] EXECUTE={int(exec_gate)} dry_run={args.dry_run} {args.side.upper()} {args.symbol} vol={args.volume} price={price}")
    if args.dry_run or not exec_gate:
        print("[result] SIMULATED (EXECUTE=0 or --dry-run)")
        return

    if not Adapter:
        print("[result] NOADAPTER (brokers.mt5_adapter missing)")
        return

    try:
        adapter = Adapter()
        login  = os.environ.get("MT5_LOGIN"); pwd = os.environ.get("MT5_PASSWORD"); server = os.environ.get("MT5_SERVER")
        ok, msg = adapter.connect(login, pwd, server)
        print(f"[adapter] connect -> ok={ok} msg={msg}")
        if not ok:
            print("[result] connect failed"); return

        res = adapter.order_market(args.symbol, args.side.upper(), float(args.volume))
        print(f"[adapter] order_market -> {json.dumps(res, ensure_ascii=False)}")

        retcode = res.get("retcode") if isinstance(res, dict) else None
        comment = res.get("comment") if isinstance(res, dict) else None
        ok_exec = (bool(res.get("ok", False)) if isinstance(res, dict) else bool(res)) or _retcode_ok(retcode, comment)
        ticket = int(res.get("ticket")) if (isinstance(res, dict) and res.get("ticket")) else None

        if ok_exec and sltp_mode == "FIXED":
            _apply_sltp(ticket, args.symbol, args.side.upper(), price, fixed_pips, tp_mult)

        print(f"[summary] ok={ok_exec} ticket={ticket} retcode={retcode} comment={comment}")
    except Exception as e:
        print(f"[error] {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()