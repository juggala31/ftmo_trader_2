#!/usr/bin/env python3
"""
Flatten all MT5 positions with safe filling-mode fallback.

CLI examples:
  python tools\\mt5_flatten_all.py --dry-run
  python tools\\mt5_flatten_all.py
  python tools\\mt5_flatten_all.py --symbol US30USD --symbol XAUUSD

Import from GUI:
  from tools.mt5_flatten_all import mt5_flatten_all
  closed = mt5_flatten_all()              # close all
  closed = mt5_flatten_all(["US30USD"])   # close only these symbols
"""

import os, time, argparse
from typing import Iterable, Optional, List

# Optional adapter (your project)
_HAS_ADAPTER = False
try:
    from brokers.mt5_adapter import MT5Adapter  # optional
    _HAS_ADAPTER = True
except Exception:
    pass

# MetaTrader5 module
_HAS_MT5 = False
try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except Exception:
    pass


def _is_initialized() -> bool:
    """Return True if MT5 terminal is initialized (via terminal_info)."""
    if not _HAS_MT5:
        return False
    try:
        info = mt5.terminal_info()
        return info is not None
    except Exception:
        return False


def _connect_via_adapter() -> bool:
    if not _HAS_ADAPTER:
        return False
    login  = os.environ.get("MT5_LOGIN")
    pwd    = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    try:
        adapter = MT5Adapter()
        ok, msg = adapter.connect(login, pwd, server)
        print(f"[adapter] connect(login, server) -> ok={ok} msg={msg}")
        return bool(ok)
    except Exception as e:
        print(f"[adapter] connect failed: {e}")
        return False


def _connect_direct() -> bool:
    if not _HAS_MT5:
        return False
    term = os.environ.get("MT5_TERMINAL_PATH") or os.environ.get("TERMINAL_PATH")
    login  = os.environ.get("MT5_LOGIN")
    pwd    = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    kw = {}
    if term:
        kw["path"] = term

    ok = False
    try:
        if login and pwd and server:
            ok = mt5.initialize(login=int(login), password=pwd, server=server, **kw)
        else:
            ok = mt5.initialize(**kw)
    except Exception:
        ok = False

    print(f"[mt5] initialize -> {ok}")
    return bool(ok)


def _ensure_connection() -> bool:
    # Already initialized?
    if _is_initialized():
        return True
    # Try adapter first (project-native), then raw MT5
    if _connect_via_adapter():
        return True
    return _connect_direct()


def _positions_list(symbols: Optional[Iterable[str]]) -> List:
    if not _HAS_MT5:
        return []
    if symbols:
        out = []
        for s in symbols:
            got = mt5.positions_get(symbol=s)
            if got:
                out.extend(list(got))
        return out
    got = mt5.positions_get()
    return list(got) if got else []


def _close_one(position, timeout_sec: float = 15.0) -> bool:
    """
    Close a single position using safe fill-mode fallback:
      ORDER_FILLING_RETURN -> ORDER_FILLING_IOC -> ORDER_FILLING_FOK
    """
    typ = position.type
    symbol = position.symbol
    volume = position.volume
    ticket = position.ticket

    # Opposite side to close a position
    close_type = mt5.ORDER_TYPE_SELL if typ == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"[warn] no tick for {symbol} (ticket {ticket})")
        return False
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
    if not price or price <= 0:
        print(f"[warn] invalid price for {symbol} (ticket {ticket})")
        return False

    # Ensure symbol is selected
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass

    base = dict(
        action=mt5.TRADE_ACTION_DEAL,
        symbol=symbol,
        volume=volume,
        type=close_type,
        position=ticket,
        price=price,
        deviation=25,
        magic=777123,
        comment="flatten_all",
        type_time=mt5.ORDER_TIME_GTC,
    )

    modes = [mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
    end_ts = time.time() + timeout_sec
    last = None

    for m in modes:
        req = dict(base); req["type_filling"] = m
        res = mt5.order_send(req)
        ret = getattr(res, "retcode", None)
        if ret == mt5.TRADE_RETCODE_DONE:
            print(f"[OK] ticket={ticket} {symbol} vol={volume} closed via mode={m}")
            return True
        print(f"[retry] ticket={ticket} mode={m} ret={ret} comment={getattr(res,'comment',None)}")
        last = ret
        if time.time() > end_ts:
            break
        time.sleep(0.25)

    print(f"[FAIL] ticket={ticket} {symbol} last_ret={last}")
    return False


def mt5_flatten_all(symbols: Optional[Iterable[str]] = None, dry_run: bool = False, timeout: float = 15.0) -> int:
    """
    Returns number of successfully closed positions.
    """
    if not _HAS_MT5:
        print("[ERR] MetaTrader5 module not available.")
        return 0

    if not _ensure_connection():
        print("[ERR] Unable to initialize/connect MT5.")
        return 0

    # Normalize symbols (strip spaces, ignore empties)
    if symbols:
        symbols = [s for s in (s.strip() for s in symbols) if s]

    pos = _positions_list(symbols)
    if not pos:
        print("[info] no open positions.")
        return 0

    print(f"[info] {len(pos)} open position(s) found.")
    if dry_run:
        # Strict dry-run: list, do NOT send closes
        for p in pos:
            side = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"
            print(f"[DRY] ticket={p.ticket} {p.symbol} {side} vol={p.volume} open={p.price_open}")
        print("[DRY] no positions were closed.")
        return 0

    closed = 0
    for p in pos:
        if _close_one(p, timeout_sec=timeout):
            closed += 1

    return closed


def main():
    ap = argparse.ArgumentParser(description="Flatten all MT5 positions with filling-mode fallback.")
    ap.add_argument("--symbol", action="append", help="Limit to this symbol. Can be repeated.")
    ap.add_argument("--dry-run", action="store_true", help="List positions without closing.")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-position timeout seconds.")
    args = ap.parse_args()

    count = mt5_flatten_all(args.symbol, dry_run=args.dry_run, timeout=args.timeout)
    print(f"[summary] closed_positions={count}")


if __name__ == "__main__":
    main()