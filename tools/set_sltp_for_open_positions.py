import os, math
from typing import Optional
import numpy as np
import MetaTrader5 as mt5

SYMS = ["XAUZ25.sim","US30Z25.sim","US100Z25.sim","US500Z25.sim","USOILZ25.sim","BTCX25.sim"]

def _retcode_map():
    d = {}
    for k in dir(mt5):
        if k.startswith("TRADE_RETCODE_"):
            d[getattr(mt5, k)] = k
    return d
RET = _retcode_map()

def cfg_int(name: str, default: int) -> int:
    try: return int(os.environ.get(name, str(default)))
    except: return default

def _symkey(symbol: str) -> str:
    return symbol.replace(".", "_").replace("-", "_")

def cfg_int_sym(name: str, symbol: str, default: int) -> int:
    v = os.environ.get(f"{name}_{_symkey(symbol)}")
    if v is None: v = os.environ.get(name)
    try: return int(v) if v is not None else default
    except: return default

def cfg_float_sym(name: str, symbol: str, default: float) -> float:
    v = os.environ.get(f"{name}_{_symkey(symbol)}")
    if v is None: v = os.environ.get(name)
    try: return float(v) if v is not None else default
    except: return default

def atr_from_rates(rates, period: int = 14) -> Optional[float]:
    if rates is None or len(rates) < period + 1: return None
    high = np.array(rates["high"], dtype=float)
    low  = np.array(rates["low"],  dtype=float)
    close= np.array(rates["close"],dtype=float)
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return float(tr[-period:].mean())

def _broker_min_points(si) -> int:
    base_pts = 0
    for attr in ("trade_stops_distance","trade_stops_level","stops_level"):
        v = getattr(si, attr, 0) or 0
        if v > base_pts: base_pts = v
    return int(base_pts)

def _snap_to_tick(p: float, si) -> float:
    # respect tick size and digits
    tick_sz = getattr(si, "trade_tick_size", None)
    step = tick_sz if (isinstance(tick_sz, (int,float)) and tick_sz>0) else si.point
    steps = round(p/step)
    return round(steps*step, si.digits)

def _calc_targets_from_current(symbol: str, is_buy: bool, atr: float, si, tick, extra_points: int, atr_mult: float):
    sl_mult = cfg_float_sym("AI_SL_MULT", symbol, 2.0)
    tp_mult = cfg_float_sym("AI_TP_MULT", symbol, 2.0)

    bid = tick.bid if tick else None
    ask = tick.ask if tick else None
    if bid is None or ask is None: return None, None

    # distance guards from *current* price
    min_pts = _broker_min_points(si) + max(0, extra_points) + cfg_int_sym("AI_STOPS_BUFFER_POINTS", symbol, 0)
    min_px  = min_pts * si.point
    min_atr = (atr_mult * atr) if atr_mult > 0 else 0.0
    guard   = max(min_px, min_atr)

    if is_buy:
        sl_raw = bid - sl_mult*atr
        tp_raw = ask + tp_mult*atr
        sl_raw = min(sl_raw, bid - guard)
        tp_raw = max(tp_raw, ask + guard)
        sl = min(_snap_to_tick(sl_raw, si), bid - si.point)
        tp = max(_snap_to_tick(tp_raw, si), ask + si.point)
    else:
        sl_raw = ask + sl_mult*atr
        tp_raw = bid - tp_mult*atr
        sl_raw = max(sl_raw, ask + guard)
        tp_raw = min(tp_raw, bid - guard)
        sl = max(_snap_to_tick(sl_raw, si), ask + si.point)
        tp = min(_snap_to_tick(tp_raw, si), bid - si.point)

    return round(sl, si.digits), round(tp, si.digits)

def _modify_sltp(symbol: str, ticket: int, sl, tp):
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": float(sl) if sl is not None else 0.0,
        "tp": float(tp) if tp is not None else 0.0,
        "deviation": cfg_int("AI_DEVIATION", 100),
        "magic": cfg_int("AI_MAGIC", 902010),
        "comment": "attach sltp"
    }
    res = mt5.order_send(req)
    return req, res

def _modify_only(symbol: str, ticket: int, sl, tp, only: str):
    if only == "SL":
        return _modify_sltp(symbol, ticket, sl, None)
    else:
        return _modify_sltp(symbol, ticket, None, tp)

def main():
    if not mt5.initialize(login=int(os.getenv("MT5_LOGIN","0") or 0),
                          password=os.getenv("MT5_PASSWORD") or "",
                          server=os.getenv("MT5_SERVER") or ""):
        print("[fix] mt5 init failed:", mt5.last_error()); return
    try:
        poss_all = mt5.positions_get()
        if not poss_all or len(poss_all) == 0:
            print("[fix] no open positions"); return
        poss = [p for p in poss_all if p.symbol in SYMS]
        print(f"[fix] scanning {len(poss)} open positions")

        for p in poss:
            s  = p.symbol
            si = mt5.symbol_info(s)
            if not si:
                print(f"  {s}: no symbol_info"); continue

            tick = mt5.symbol_info_tick(s)
            if not tick:
                print(f"  {s}: no tick"); continue

            rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_H1, 0, 200)
            if rates is None or len(rates) < 20:
                print(f"  {s}: insufficient H1 bars for ATR"); continue
            atr = atr_from_rates(rates)
            if atr is None or not math.isfinite(atr):
                print(f"  {s}: bad ATR"); continue

            has_sl = (p.sl is not None and p.sl > 0.0)
            has_tp = (p.tp is not None and p.tp > 0.0)
            if has_sl and has_tp:
                print(f"  {s} #{p.ticket}: SL/TP already set"); continue

            is_buy = (p.type == mt5.POSITION_TYPE_BUY)

            point_buffers = [0, 10, 25, 50, 100, 250, 500, 1000]
            atr_multipliers = [2.0, 3.0, 5.0, 8.0]

            def try_set(buf_pts: int, atrx: float, only: Optional[str]=None):
                sl, tp = _calc_targets_from_current(s, is_buy, atr, si, tick, buf_pts, atrx)
                if sl is None or tp is None:
                    return None
                req, res = _modify_only(s, p.ticket, sl, tp, only) if only else _modify_sltp(s, p.ticket, sl, tp)
                rc = getattr(res, "retcode", None)
                name = RET.get(rc, str(rc))
                cmt = getattr(res, "comment", "")
                extra = f" {only}-only" if only else ""
                print(f"    -> buf={buf_pts} atrx={atrx} try{extra}: rc={name} cmt='{cmt}' sl={sl} tp={tp}")
                return rc

            done = False
            for buf in point_buffers:
                rc = try_set(buf, 0.0, None)
                if rc == mt5.TRADE_RETCODE_DONE:
                    done = True; print(f"  {s} #{p.ticket}: SLTP set (bufPts={buf})"); break
                rc_sl = try_set(buf, 0.0, "SL")
                if rc_sl == mt5.TRADE_RETCODE_DONE:
                    done = True; print(f"  {s} #{p.ticket}: SL set only (bufPts={buf})"); break
                rc_tp = try_set(buf, 0.0, "TP")
                if rc_tp == mt5.TRADE_RETCODE_DONE:
                    done = True; print(f"  {s} #{p.ticket}: TP set only (bufPts={buf})"); break
            if done:
                continue

            for k in atr_multipliers:
                rc = try_set(0, k, None)
                if rc == mt5.TRADE_RETCODE_DONE:
                    done = True; print(f"  {s} #{p.ticket}: SLTP set (atrx={k})"); break
                rc_sl = try_set(0, k, "SL")
                if rc_sl == mt5.TRADE_RETCODE_DONE:
                    done = True; print(f"  {s} #{p.ticket}: SL set only (atrx={k})"); break
                rc_tp = try_set(0, k, "TP")
                if rc_tp == mt5.TRADE_RETCODE_DONE:
                    done = True; print(f"  {s} #{p.ticket}: TP set only (atrx={k})"); break

            if not done:
                print(f"  {s} #{p.ticket}: still no SL/TP (kept open)")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()