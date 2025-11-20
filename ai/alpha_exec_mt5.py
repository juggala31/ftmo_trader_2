import os, time, math
from typing import Optional, Tuple
import numpy as np

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise RuntimeError(f"[ai][exec] MetaTrader5 module unavailable: {e}")

def _resolve_filling_mode(filling_mode: int) -> int:
    """
    Convert symbol_info.filling_mode bitmask into an ORDER_FILLING_* constant.
    Falls back to ORDER_FILLING_RETURN for safety.
    """
    try:
        fm = int(filling_mode or 0)
    except Exception:
        fm = 0

    # Use SYMBOL_FILLING_* flags when available
    try:
        if hasattr(mt5, "SYMBOL_FILLING_FOK") and (fm & mt5.SYMBOL_FILLING_FOK):
            return mt5.ORDER_FILLING_FOK
        if hasattr(mt5, "SYMBOL_FILLING_IOC") and (fm & mt5.SYMBOL_FILLING_IOC):
            return mt5.ORDER_FILLING_IOC
        if hasattr(mt5, "SYMBOL_FILLING_RETURN") and (fm & mt5.SYMBOL_FILLING_RETURN):
            return mt5.ORDER_FILLING_RETURN
    except Exception:
        # fall back to simple mapping below
        pass

    # Some servers just return 1,2,3 (FOK / IOC / FOK|IOC)
    if fm == 1:
        return mt5.ORDER_FILLING_FOK
    if fm == 2:
        return mt5.ORDER_FILLING_IOC
    if fm == 3:
        # Prefer IOC when both are allowed
        return mt5.ORDER_FILLING_IOC

    # Last-resort default
    return mt5.ORDER_FILLING_RETURN

# --------------------------
# Helpers
# --------------------------
def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)

def getenv_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)

def getenv_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","on")

def _sym_key(symbol: str) -> str:
    # Match your env naming convention: letters+digits with '.' -> '_' for .sim
    return symbol.replace('.', '_')

def compute_atr(rates: np.ndarray, period: int = 14) -> float:
    """
    Expects MT5 rates array with fields: time, open, high, low, close, tick_volume, spread, real_volume
    Returns ATR in price units (same units as close).
    """
    if rates is None or len(rates) < period + 1:
        return 0.0
    high = rates['high'].astype(float)
    low  = rates['low'].astype(float)
    close= rates['close'].astype(float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close)
    ])
    # last 'period' bars average
    atr = np.mean(tr[-period:])
    return float(atr)

def _symbol_info_tuple(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"[ai][exec] symbol_info({symbol}) is None ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â add it to Market Watch.")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"[ai][exec] symbol_info_tick({symbol}) is None.")
    digits = int(info.digits)
    point  = float(info.point)
    vol_min = float(getattr(info, "volume_min", 0.01) or 0.01)
    vol_step= float(getattr(info, "volume_step", 0.01) or 0.01)
    stops_level = int(getattr(info, "stops_level", 0) or 0)
    filling_mode = int(getattr(info, "filling_mode", 0) or 0)
    return info, tick, digits, point, vol_min, vol_step, stops_level, filling_mode

def _round_volume(vol: float, step: float, vmin: float) -> float:
    if step <= 0: step = 0.01
    v = max(vmin, math.floor(vol/step + 1e-9) * step)
    return float(round(v, 2))

def _calc_sltp_for_side(side: str, price: float, atr: float, sl_mult: float, tp_mult: float,
                        point: float, stops_level: int, buffer_pts: int) -> Tuple[float,float]:
    """
    Returns (sl, tp) in absolute price terms.
    Ensures min distance of (stops_level + buffer_pts) * point.
    """
    # desired raw distances
    sl_dist = atr * sl_mult if atr > 0 else 0.0
    tp_dist = atr * tp_mult if atr > 0 else 0.0

    # enforce minimum distance by stops level + buffer
    min_dist = (stops_level + buffer_pts) * point
    if sl_dist < min_dist: sl_dist = min_dist
    if tp_dist < min_dist: tp_dist = min_dist

    if side == "BUY":
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist
    return (float(sl), float(tp))

def _ensure_sltp(symbol: str, ticket: int, side: str, price: float, atr: float,
                 point: float, stops_level: int, sl_mult: float, tp_mult: float, buffer_pts: int) -> Tuple[bool, Optional[int]]:
    sl, tp = _calc_sltp_for_side(side, price, atr, sl_mult, tp_mult, point, stops_level, buffer_pts)
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": sl,
        "tp": tp,
        "type_time": mt5.ORDER_TIME_GTC
    }
    res = mt5.order_send(req)
    ok = (res is not None and res.retcode == mt5.TRADE_RETCODE_DONE)
    return ok, (res.retcode if res is not None else None)

def _close_position(symbol: str, ticket: int, side: str, volume: float, deviation: int, filling_mode: int) -> Tuple[bool, Optional[int]]:
    # To close, we send a DEAL in the opposite direction
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, None
    if side == "BUY":
        price = float(tick.bid)
        otype = mt5.ORDER_TYPE_SELL
    else:
        price = float(tick.ask)
        otype = mt5.ORDER_TYPE_BUY
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": otype,
        "position": ticket,
        "volume": float(volume),
        "price": price,
        "deviation": int(deviation),
        "type_filling": _resolve_filling_mode(filling_mode),
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "ai.close.flip"
    }
    res = mt5.order_send(req)
    ok = (res is not None and res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED))
    return ok, (res.retcode if res is not None else None)

# --------------------------
# Public entry point
# --------------------------
def maybe_flip_position(symbol: str, signal: float, rates: np.ndarray) -> Tuple[bool, Optional[int]]:
    """
    Executes orders with SL/TP inline where possible.
    signal: +1.0 (long), -1.0 (short), 0.0 (flat/no action)
    returns (executed, retcode)
    """
    # Config (supports both global and per-symbol env via *_<SYMBOL> keys)
    sym_key     = _sym_key(symbol)

    base_lots   = getenv_float("AI_LOTS", 0.10)
    lots        = getenv_float(f"AI_LOTS_{sym_key}", base_lots)

    base_sl     = getenv_float("AI_SL_MULT", 2.0)
    sl_mult     = getenv_float(f"AI_SL_MULT_{sym_key}", base_sl)

    base_tp     = getenv_float("AI_TP_MULT", 2.0)
    tp_mult     = getenv_float(f"AI_TP_MULT_{sym_key}", base_tp)

    base_dev    = getenv_int("AI_DEVIATION", 50)
    deviation   = getenv_int(f"AI_DEVIATION_{sym_key}", base_dev)

    base_cd     = getenv_int("AI_COOLDOWN_SEC", 0)
    cool_sec    = getenv_int(f"AI_COOLDOWN_SEC_{sym_key}", base_cd)

    buf_pts     = getenv_int(f"AI_STOPS_BUFFER_POINTS_{sym_key}", 0)
    dryrun      = getenv_bool("DRY_RUN_AI", False)

    # Compute ATR from provided bars
    atr = compute_atr(rates, period=14)

    info, tick, digits, point, vmin, vstep, stops_level, filling_mode = _symbol_info_tuple(symbol)
    print(f"[ai][exec][diag] {symbol}: digits={digits} point={point} vmin={vmin} vstep={vstep} stops_level={stops_level} "
          f"dist={stops_level}+{buf_pts} freeze={getattr(info,'freeze_level',0)} filling={filling_mode}")

    # ATR-based lot sizing (optional)
    risk_mode_env = os.environ.get(f"AI_RISK_MODE_{sym_key}") or os.environ.get("AI_RISK_MODE", "FIXED_LOT")
    risk_mode = str(risk_mode_env).upper()
    risk_pct = getenv_float(f"AI_RISK_PCT_{sym_key}", getenv_float("AI_RISK_PCT", 0.0))

    if risk_mode == "ATR_RISK" and risk_pct > 0.0 and atr > 0.0 and sl_mult > 0.0:
        acct = mt5.account_info()
        eq = float(getattr(acct, "equity", 0.0) or getattr(acct, "balance", 0.0) or 0.0)
        contract_size = float(getattr(info, "trade_contract_size", 1.0) or 1.0)

        if eq > 0.0 and contract_size > 0.0:
            risk_money = eq * (risk_pct / 100.0)
            # Approx risk per 1 lot if SL ~= atr * sl_mult:
            #   risk_per_lot ≈ atr * sl_mult * contract_size
            denom = atr * sl_mult * contract_size
            if denom > 0.0:
                lots_atr = risk_money / denom
                if lots_atr > 0.0:
                    print(f"[ai][exec][atr] {symbol}: mode=ATR_RISK risk_pct={risk_pct} eq={eq:.2f} atr={atr:.5f} sl_mult={sl_mult} cs={contract_size} raw_lots={lots_atr:.4f}")
                    lots = lots_atr

    # Current position if any
    existing = mt5.positions_get(symbol=symbol)
    pos = existing[0] if existing else None
    side_now = None
    if pos is not None:
        side_now = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"

    # Decide desired side
    desired = None
    if signal > 0:
        desired = "BUY"
    elif signal < 0:
        desired = "SELL"
    else:
        # flat signal: no open/flip; could close if you want ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â we hold by default
        if pos is not None:
            # optionally ensure SL/TP if missing
            if (pos.sl == 0.0 or pos.tp == 0.0) and not dryrun:
                ok, rc = _ensure_sltp(symbol, pos.ticket, side_now, float(pos.price_open), atr, point, stops_level, sl_mult, tp_mult, buf_pts)
                if ok:
                    print(f"[ai][exec] hold {symbol} {side_now}; sltp_ret=10013 (kept open)")
                else:
                    print(f"[ai][exec] hold {symbol} {side_now}; sltp_ret={rc} (gave up)")
        return False, None

    # If we already have the desired side, just ensure SL/TP present
    if pos is not None and side_now == desired:
        if (pos.sl == 0.0 or pos.tp == 0.0) and not dryrun:
            ok, rc = _ensure_sltp(symbol, pos.ticket, desired, float(pos.price_open), atr, point, stops_level, sl_mult, tp_mult, buf_pts)
            msg = "kept open" if ok else "gave up"
            print(f"[ai][exec] hold {symbol} {desired}; sltp_ret={'10013' if ok else rc} ({msg})")
        else:
            print(f"[ai][exec] hold {symbol} {desired}")
        return True, 10009  # nothing new placed; treated as handled

    # If opposite side is open, close it first
    if pos is not None and side_now != desired and not dryrun:
        ok, rc = _close_position(symbol, pos.ticket, side_now, float(pos.volume), deviation, filling_mode)
        if not ok:
            print(f"[ai][exec] close-for-flip {symbol} {side_now} -> rc={rc}")
            return False, rc
        # tiny pause to let the terminal settle
        time.sleep(0.15)

    if dryrun:
        print(f"[ai][exec] (dry) would open {symbol} {desired}")
        return True, 10009

    # Determine entry price and inline SL/TP
    if desired == "BUY":
        price = float(tick.ask)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = float(tick.bid)
        order_type = mt5.ORDER_TYPE_SELL

    sl, tp = _calc_sltp_for_side(desired, price, atr, sl_mult, tp_mult, point, stops_level, buf_pts)

    vol = _round_volume(lots, vstep, vmin)
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": order_type,
        "volume": float(vol),
        "price": price,
        "deviation": int(deviation),
        "sl": sl,
        "tp": tp,
        "type_filling": _resolve_filling_mode(filling_mode),
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "ai.open.inline_sltp"
    }

    res = mt5.order_send(req)
    if res is None:
        print(f"[ai][exec] open {symbol} {desired}: False ret=None")
        return False, None

    if res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        print(f"[ai][exec] open {symbol} {desired}: True ret={res.retcode}")
        return True, int(res.retcode)

    # If the broker refused SL/TP inline, try open-without then attach SL/TP
    # (common with futures-like .sim if stops_level reacts weirdly)
    print(f"[ai][exec] inline SL/TP rejected rc={res.retcode}; trying open-without then attachÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦")

    req2 = dict(req)
    req2.pop("sl", None)
    req2.pop("tp", None)
    req2["comment"] = "ai.open.no_sltp_then_attach"
    res2 = mt5.order_send(req2)
    if res2 is None or res2.retcode not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        print(f"[ai][exec] open {symbol} {desired} (no-sltp) failed rc={(None if res2 is None else res2.retcode)}")
        return False, (None if res2 is None else int(res2.retcode))

    # Find the fresh position and attach SL/TP
    time.sleep(0.15)
    pos_now = mt5.positions_get(symbol=symbol)
    if pos_now:
        ticket = pos_now[0].ticket
        ok, rc3 = _ensure_sltp(symbol, ticket, desired, price, atr, point, stops_level, sl_mult, tp_mult, buf_pts)
        if ok:
            print(f"[ai][exec] attach SL/TP after open ok rc=10013")
        else:
            print(f"[ai][exec] attach SL/TP after open failed rc={rc3}")
    return True, int(res2.retcode)