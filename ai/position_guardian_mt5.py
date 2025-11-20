"""
MT5 Position Guardian for FTMO Trader 2.0

Watches open positions at a higher frequency than the H1 entry logic and
applies defensive risk rules:

- Breakeven move once trade reaches +1R.
- ATR / R-multiple based trailing stop for strong winners.
- Early "panic" exit on abnormal fast losses.
- Time-based exit for zombie trades that go nowhere.

Guardian v1 is conservative and can run in DRY (advisory-only) mode.
"""

import os
import time
from typing import Optional, Dict

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise RuntimeError(f"[guardian] MetaTrader5 module unavailable: {e!r}")

# --- Global toggles and parameters ---

# If DRY_RUN_AI != "0", we only LOG actions, we do not send MT5 orders.
DRY_RUN = os.getenv("DRY_RUN_AI", "1") != "0"

# Master enable flag for the guardian itself
ENABLED = os.getenv("AI_GUARDIAN_ENABLED", "1") != "0"

# Verbose diagnostics (print a line per position even if no rule fires)
VERBOSE = os.getenv("AI_GUARDIAN_VERBOSE", "1") != "0"

# Loop sleep in seconds
LOOP_SEC = float(os.getenv("AI_GUARD_LOOP_SEC", "5"))

# R-multiple based thresholds
BREAKEVEN_TRIGGER_R = float(os.getenv("AI_GUARD_BE_TRIGGER_R", "1.0"))    # when to move SL to BE
BREAKEVEN_OFFSET_R  = float(os.getenv("AI_GUARD_BE_OFFSET_R", "0.10"))    # small cushion above BE

TRAIL_TRIGGER_R     = float(os.getenv("AI_GUARD_TRAIL_TRIGGER_R", "1.5")) # when to start trailing
TRAIL_ATR_MULT      = float(os.getenv("AI_GUARD_TRAIL_ATR_MULT", "1.0"))  # ATR multiple for trail
TRAIL_MIN_R         = float(os.getenv("AI_GUARD_TRAIL_MIN_R", "0.75"))    # minimum locked-in R

# Panic / zombie thresholds
PANIC_EARLY_AGE_SEC = float(os.getenv("AI_GUARD_PANIC_EARLY_SEC", str(60 * 60)))       # < 1h old
PANIC_LOSS_R        = float(os.getenv("AI_GUARD_PANIC_LOSS_R", "1.25"))               # loss worse than -1.25R

ZOMBIE_AGE_SEC      = float(os.getenv("AI_GUARD_ZOMBIE_AGE_SEC", str(6 * 60 * 60)))   # >= 6h old
ZOMBIE_NEAR_R       = float(os.getenv("AI_GUARD_ZOMBIE_NEAR_R", "0.5"))               # between -0.5R and +0.5R

# ATR timeframe for "fast" volatility (for trailing distance)
GUARD_ATR_TF = os.getenv("AI_GUARD_ATR_TF", "M15").upper()
ATR_PERIOD   = int(os.getenv("AI_GUARD_ATR_PERIOD", "14"))


# --- Helpers ---

def _map_tf(tf_str: str):
    """Map simple TF code like 'M5','M15','M30','H1' to MT5 timeframe constant."""
    tf_str = tf_str.upper()
    if tf_str == "M1":
        return mt5.TIMEFRAME_M1
    if tf_str == "M5":
        return mt5.TIMEFRAME_M5
    if tf_str == "M15":
        return mt5.TIMEFRAME_M15
    if tf_str == "M30":
        return mt5.TIMEFRAME_M30
    if tf_str == "H1":
        return mt5.TIMEFRAME_H1
    if tf_str == "H4":
        return mt5.TIMEFRAME_H4
    return mt5.TIMEFRAME_M15


def _now() -> float:
    return time.time()


def _ensure_mt5_initialized() -> bool:
    """
    Ensure MetaTrader5 is initialized.

    Uses MT5_LOGIN / MT5_PASSWORD / MT5_SERVER if available, otherwise
    falls back to a plain mt5.initialize().
    """
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if login and password and server:
        try:
            login_int = int(login)
        except ValueError:
            print(f"[guardian][warn] MT5_LOGIN='{login}' is not an int, using default initialize()")
            ok = mt5.initialize()
        else:
            ok = mt5.initialize(login=login_int, password=password, server=server)
    else:
        ok = mt5.initialize()

    if not ok:
        print(f"[guardian][err] mt5.initialize() failed: last_error={mt5.last_error()}")
    else:
        print("[guardian] mt5.initialize() OK (guardian)")
    return ok


def _compute_atr_points(symbol: str, period: int = ATR_PERIOD) -> Optional[float]:
    """Compute ATR in price units on the chosen fast TF."""
    tf = _map_tf(GUARD_ATR_TF)
    count = period + 1
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) < period + 1:
        return None

    trs = []
    prev_close = None
    for r in rates:
        high = r["high"]
        low = r["low"]
        close = r["close"]
        if prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
        trs.append(tr)
        prev_close = close

    if not trs:
        return None

    atr = sum(trs[-period:]) / float(period)
    return atr


def _calc_r_multiple(pos, current_price: float) -> Optional[float]:
    """
    Calculate current R-multiple of the trade.

    1R is defined as |entry - SL| in price units.
    """
    entry = pos.price_open
    sl = pos.sl
    if sl <= 0:
        return None

    if pos.type == mt5.POSITION_TYPE_BUY:
        risk_price = entry - sl
        if risk_price <= 0:
            return None
        pnl_price = current_price - entry
        r = pnl_price / risk_price
    else:
        # SELL
        risk_price = sl - entry
        if risk_price <= 0:
            return None
        pnl_price = entry - current_price
        r = pnl_price / risk_price

    return r


def _desired_breakeven_sl(pos, risk_price: float) -> float:
    """Return target SL for breakeven+offset in price units."""
    entry = pos.price_open
    if pos.type == mt5.POSITION_TYPE_BUY:
        return entry + BREAKEVEN_OFFSET_R * risk_price
    else:
        return entry - BREAKEVEN_OFFSET_R * risk_price


def _desired_trailing_sl(
    pos,
    current_price: float,
    risk_price: float,
    atr_points: Optional[float],
) -> Optional[float]:
    """
    ATR-based trailing SL:

    trail_dist_price = max(TRAIL_ATR_MULT * atr_points, TRAIL_MIN_R * risk_price)
    """
    if atr_points is None or atr_points <= 0:
        trail_dist = TRAIL_MIN_R * risk_price
    else:
        trail_dist = max(TRAIL_ATR_MULT * atr_points, TRAIL_MIN_R * risk_price)

    if pos.type == mt5.POSITION_TYPE_BUY:
        return current_price - trail_dist
    else:
        return current_price + trail_dist


def _modify_sl(pos, new_sl: float):
    """Send MT5 SL modification request for a position."""
    if DRY_RUN:
        print(f"[guardian][dry] would_modify_sl ticket={pos.ticket} symbol={pos.symbol} sl={pos.sl:.5f} -> {new_sl:.5f}")
        return

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol": pos.symbol,
        "sl": new_sl,
        "tp": pos.tp,
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"[guardian][err] order_send returned None for SL modify ticket={pos.ticket}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[guardian][err] SL modify failed ticket={pos.ticket} retcode={result.retcode} comment={result.comment}")
    else:
        print(f"[guardian] SL modified ticket={pos.ticket} sl={pos.sl:.5f} -> {new_sl:.5f}")


def _close_position(pos):
    """Close position at market."""
    if DRY_RUN:
        print(f"[guardian][dry] would_close ticket={pos.ticket} symbol={pos.symbol} volume={pos.volume}")
        return

    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        print(f"[guardian][err] no tick data for {pos.symbol} to close position")
        return

    if pos.type == mt5.POSITION_TYPE_BUY:
        price = tick.bid
        order_type = mt5.ORDER_TYPE_SELL
    else:
        price = tick.ask
        order_type = mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": pos.volume,
        "type": order_type,
        "position": pos.ticket,
        "price": price,
        "deviation": 50,
        "comment": "guardian-close",
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result is None:
        print(f"[guardian][err] order_send returned None for close ticket={pos.ticket}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[guardian][err] close failed ticket={pos.ticket} retcode={result.retcode} comment={result.comment}")
    else:
        print(f"[guardian] closed ticket={pos.ticket} at price={price}")


def _get_mid_price(symbol: str) -> Optional[float]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    if tick.ask and tick.bid:
        return (tick.ask + tick.bid) / 2.0
    return tick.last or tick.bid or tick.ask


def _guardian_step():
    """One full scan of all open positions and apply rule decisions."""
    positions = mt5.positions_get()
    if positions is None:
        return
    if len(positions) == 0:
        if VERBOSE:
            print("[guardian][diag] no open positions")
        return

    now = _now()
    atr_cache: Dict[str, Optional[float]] = {}

    for pos in positions:
        symbol = pos.symbol
        if symbol not in atr_cache:
            atr_cache[symbol] = _compute_atr_points(symbol)
        atr_points = atr_cache[symbol]

        mid = _get_mid_price(symbol)
        if mid is None:
            continue

        sl = pos.sl
        if sl <= 0:
            if VERBOSE:
                print(f"[guardian][diag] ticket={pos.ticket} symbol={symbol} has no SL, skipping")
            continue

        entry = pos.price_open
        if pos.type == mt5.POSITION_TYPE_BUY:
            risk_price = entry - sl
        else:
            risk_price = sl - entry

        if risk_price <= 0:
            if VERBOSE:
                print(f"[guardian][diag] ticket={pos.ticket} symbol={symbol} bad risk_price={risk_price}, skipping")
            continue

        r = _calc_r_multiple(pos, mid)
        if r is None:
            if VERBOSE:
                print(f"[guardian][diag] ticket={pos.ticket} symbol={symbol} could not compute R, skipping")
            continue

        age_sec = now - pos.time

        if VERBOSE:
            print(
                f"[guardian][diag] ticket={pos.ticket} symbol={symbol} age={age_sec:.0f}s "
                f"r={r:.2f} sl={sl:.5f} mid={mid:.5f}"
            )

        # --- PANIC rule ---
        if age_sec < PANIC_EARLY_AGE_SEC and r <= -PANIC_LOSS_R:
            print(
                f"[guardian][panic] ticket={pos.ticket} symbol={symbol} age={age_sec:.0f}s r={r:.2f} "
                f"-> EARLY oversized loss, closing."
            )
            _close_position(pos)
            continue

        # --- BREAKEVEN rule ---
        if r >= BREAKEVEN_TRIGGER_R:
            be_sl = _desired_breakeven_sl(pos, risk_price)
            if pos.type == mt5.POSITION_TYPE_BUY:
                if be_sl > sl:
                    print(
                        f"[guardian][be] ticket={pos.ticket} symbol={symbol} r={r:.2f} "
                        f"sl={sl:.5f} -> be_sl={be_sl:.5f}"
                    )
                    _modify_sl(pos, be_sl)
                    sl = be_sl
            else:
                if be_sl < sl:
                    print(
                        f"[guardian][be] ticket={pos.ticket} symbol={symbol} r={r:.2f} "
                        f"sl={sl:.5f} -> be_sl={be_sl:.5f}"
                    )
                    _modify_sl(pos, be_sl)
                    sl = be_sl

        # --- TRAILING rule ---
        if r >= TRAIL_TRIGGER_R:
            trail_sl = _desired_trailing_sl(pos, mid, risk_price, atr_points)
            if trail_sl is not None:
                if pos.type == mt5.POSITION_TYPE_BUY:
                    if trail_sl > sl and trail_sl < mid:
                        print(
                            f"[guardian][trail] ticket={pos.ticket} symbol={symbol} r={r:.2f} "
                            f"sl={sl:.5f} -> trail_sl={trail_sl:.5f} (mid={mid:.5f})"
                        )
                        _modify_sl(pos, trail_sl)
                else:
                    if trail_sl < sl and trail_sl > mid:
                        print(
                            f"[guardian][trail] ticket={pos.ticket} symbol={symbol} r={r:.2f} "
                            f"sl={sl:.5f} -> trail_sl={trail_sl:.5f} (mid={mid:.5f})"
                        )
                        _modify_sl(pos, trail_sl)

        # --- ZOMBIE rule ---
        if age_sec >= ZOMBIE_AGE_SEC and -ZOMBIE_NEAR_R <= r <= ZOMBIE_NEAR_R:
            print(
                f"[guardian][zombie] ticket={pos.ticket} symbol={symbol} age={age_sec/3600:.1f}h "
                f"r={r:.2f} -> closing dead trade."
            )
            _close_position(pos)
            continue


def run_guardian_loop():
    """
    Main loop entry for the position guardian.

    This is designed to run either:
    - In its own process (via guardian_entry.py), or
    - Embedded in an existing MT5-initialized process.
    """
    if not ENABLED:
        print("[guardian] Position Guardian DISABLED via AI_GUARDIAN_ENABLED=0")
        return

    if not _ensure_mt5_initialized():
        return

    print(
        "[guardian] Position Guardian starting "
        f"(LOOP_SEC={LOOP_SEC}, DRY_RUN={'YES' if DRY_RUN else 'NO'}, VERBOSE={'YES' if VERBOSE else 'NO'})"
    )

    while True:
        try:
            _guardian_step()
        except Exception as e:
            print(f"[guardian][err] exception in guardian loop: {e!r}")
        time.sleep(LOOP_SEC)