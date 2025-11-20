#!/usr/bin/env python
"""
Backtest runner for Trader 2.0

- Loads an OHLC(+features) CSV, typically from ai/datasets/csv/{TF}/{symbol}.csv.
- Supports two signal modes:
    * demo : simple EMA/RSI rule
    * ml   : XGBoost-based probabilities using ai.xgb_loader_api
- Simulates trades with a simple 1-position-per-symbol engine:
    * ATR-based SL/TP (default SL=1 ATR, TP=2 ATR)
    * FIXED_LOT sizing (PF/WR independent of absolute lot size in backtest).
- Writes trades CSV with the same schema as data/trades.csv:
    [ts,ticket,symbol,side,volume,entry,close,realized_quote,realized_ccy,reason]
- Prints summary metrics (PF, win-rate, expectancy, max drawdown).

ML mode:
- Uses ai.xgb_loader_api:
    HAVE_XGB_INF, XGB_FEATURE_COLS, xgb_predict_proba(symbol, tf, features)
- Thresholds are pulled from ai/ai_profile.json per symbol×TF when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root & sys.path so "import ai..." works even from backtest/
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ML backend integration: shared with live alpha via ai.xgb_loader_api
# ---------------------------------------------------------------------------

try:
    from ai.xgb_loader_api import (  # type: ignore
        HAVE_XGB_INF,
        XGB_FEATURE_COLS,
        xgb_predict_proba,
    )
    _ML_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - missing module / bad import
    HAVE_XGB_INF = False
    XGB_FEATURE_COLS: List[str] = []
    _ML_IMPORT_ERROR = str(e)

    def xgb_predict_proba(symbol: str, tf: str, features: Dict[str, float]) -> Optional[float]:
        return None

# ---------------------------------------------------------------------------
# CSV / time helpers
# ---------------------------------------------------------------------------


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def compute_atr(rows: List[Dict[str, str]], period: int = 14) -> List[Optional[float]]:
    atr: List[Optional[float]] = [None] * len(rows)
    prev_close: Optional[float] = None
    tr_ema: Optional[float] = None
    k = 2.0 / (period + 1.0)

    for i, row in enumerate(rows):
        h = to_float(row.get("high"))
        l = to_float(row.get("low"))
        c = to_float(row.get("close"))
        if h is None or l is None or c is None:
            atr[i] = None
            continue

        if prev_close is None:
            prev_close = c
            tr = h - l
        else:
            base = prev_close
            tr = max(h - l, abs(h - base), abs(l - base))
            prev_close = c

        if tr_ema is None:
            tr_ema = tr
            atr[i] = None  # warmup
        else:
            tr_ema = tr_ema + k * (tr - tr_ema)
            atr[i] = tr_ema
    return atr


def parse_time_to_epoch(ts_str: str) -> Optional[float]:
    if ts_str is None:
        return None
    s = str(ts_str).strip()
    if not s:
        return None
    dt_obj: Optional[datetime] = None
    try:
        dt_obj = datetime.fromisoformat(s)
    except Exception:
        try:
            dt_obj = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.timestamp()


def parse_time_to_dt(ts_str: str) -> Optional[datetime]:
    if ts_str is None:
        return None
    s = str(ts_str).strip()
    if not s:
        return None
    try:
        dt_obj = datetime.fromisoformat(s)
    except Exception:
        try:
            dt_obj = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj


# ---------------------------------------------------------------------------
# Fee model (placeholder)
# ---------------------------------------------------------------------------


def apply_costs(pnl: float) -> float:
    return pnl  # passthrough for now


# ---------------------------------------------------------------------------
# Session filters
# ---------------------------------------------------------------------------


def classify_session(dt_obj: datetime) -> str:
    h = dt_obj.hour
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 13:
        return "LONDON"
    if 13 <= h < 20:
        return "NY"
    return "OFF"


def load_session_filters() -> Dict[str, List[str]]:
    path = ROOT / "ai" / "session_filters.json"
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            data = json.load(f)
        out: Dict[str, List[str]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                out[k] = [str(s).upper() for s in v]
        return out
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Profile thresholds (ML)
# ---------------------------------------------------------------------------


def load_profile_thresholds(symbol: str, tf: str) -> Tuple[float, float]:
    default_long = 0.60
    default_short = 0.40
    path = ROOT / "ai" / "ai_profile.json"
    if not path.exists():
        return default_long, default_short
    try:
        with path.open("r") as f:
            prof = json.load(f)
    except Exception:
        return default_long, default_short

    entry = prof.get(symbol)
    if not isinstance(entry, dict):
        return default_long, default_short

    tf_cfg = entry.get("timeframes", {}).get(tf, {})
    if not isinstance(tf_cfg, dict):
        return default_long, default_short

    long_th = tf_cfg.get("long_threshold", default_long)
    short_th = tf_cfg.get("short_threshold", default_short)

    try:
        return float(long_th), float(short_th)
    except Exception:
        return default_long, default_short


# ---------------------------------------------------------------------------
# Indicator helpers (demo mode)
# ---------------------------------------------------------------------------


def rsi(series: List[Optional[float]], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(series)
    gain: List[float] = []
    loss: List[float] = []
    for i, v in enumerate(series):
        if v is None:
            out[i] = None
            gain.append(0.0)
            loss.append(0.0)
            continue
        if i == 0:
            gain.append(0.0)
            loss.append(0.0)
            out[i] = None
            continue
        delta = v - (series[i - 1] or v)
        g = max(delta, 0.0)
        l = max(-delta, 0.0)
        gain.append(g)
        loss.append(l)
        if i < period:
            out[i] = None
        else:
            avg_g = sum(gain[i + 1 - period : i + 1]) / period
            avg_l = sum(loss[i + 1 - period : i + 1]) / period
            if avg_l == 0:
                out[i] = 100.0
            else:
                rs = avg_g / avg_l
                out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def ema(series: List[Optional[float]], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(series)
    k = 2.0 / (period + 1.0)
    ema_val: Optional[float] = None
    for i, v in enumerate(series):
        if v is None:
            out[i] = None
            continue
        if ema_val is None:
            ema_val = v
        else:
            ema_val = ema_val + k * (v - ema_val)
        out[i] = ema_val
    return out


def demo_signals(rows: List[Dict[str, str]]) -> List[int]:
    closes = [to_float(r.get("close")) for r in rows]
    ema50 = ema(closes, 50)
    rsi14 = rsi(closes, 14)
    sigs: List[int] = [0] * len(rows)
    for i in range(len(rows)):
        c = closes[i]
        e = ema50[i]
        r = rsi14[i]
        if c is None or e is None or r is None:
            sigs[i] = 0
            continue
        if c > e and r > 55.0:
            sigs[i] = 1
        elif c < e and r < 45.0:
            sigs[i] = -1
        else:
            sigs[i] = 0
    return sigs


# ---------------------------------------------------------------------------
# ML signal provider (aligned with alpha_loop + xgb_loader_api)
# ---------------------------------------------------------------------------


def ml_signals(
    rows: List[Dict[str, str]], symbol: str, tf: str, long_th: float, short_th: float
) -> Tuple[List[int], bool]:
    """
    Return (signals, used_ml).

    Signals:
        +1 = long, -1 = short, 0 = flat
    used_ml:
        True if at least one bar used XGB and produced a non-None probability.
    """
    sigs: List[int] = [0] * len(rows)
    used_ml = False

    if not HAVE_XGB_INF or not XGB_FEATURE_COLS or xgb_predict_proba is None:
        return sigs, False

    for i, row in enumerate(rows):
        features: Dict[str, float] = {}
        for col in XGB_FEATURE_COLS:
            val = row.get(col, None)
            if val is None or val == "":
                features[col] = float("nan")
            else:
                try:
                    features[col] = float(val)
                except Exception:
                    features[col] = float("nan")

        if not features:
            continue
        if any(math.isnan(features[c]) for c in XGB_FEATURE_COLS):
            continue

        try:
            p_up = xgb_predict_proba(symbol, tf, features)
        except Exception:
            p_up = None

        if p_up is None:
            continue

        used_ml = True
        if p_up >= long_th:
            sigs[i] = 1
        elif p_up <= short_th:
            sigs[i] = -1
        else:
            sigs[i] = 0

    return sigs, used_ml


# ---------------------------------------------------------------------------
# Simple simulator
# ---------------------------------------------------------------------------


@dataclass
class Position:
    symbol: str
    side: str  # BUY or SELL
    entry: float
    size: float
    sl: float
    tp: float
    open_bar: int
    closed_at: Optional[int] = None
    exit: Optional[float] = None
    pnl: float = 0.0
    reason: str = ""


@dataclass
class SimConfig:
    risk_mode: str = "FIXED_LOT"
    fixed_lot: float = 0.10
    atr_risk_pct: float = 0.5
    sl_atr_mult: float = 1.0
    tp_atr_mult: float = 2.0
    cooldown_bars: int = 0
    max_positions: int = 1


@dataclass
class SimState:
    equity: float = 100000.0
    open_positions: List[Position] = field(default_factory=list)
    last_entry_bar: int = -10**9


def update_positions(
    st: SimState,
    cfg: SimConfig,
    bar_idx: int,
    high: float,
    low: float,
) -> None:
    for p in st.open_positions:
        if p.closed_at is not None:
            continue
        exit_price: Optional[float] = None
        exit_reason: str = ""
        if p.side == "BUY":
            if low <= p.sl:
                exit_price = p.sl
                exit_reason = "bt_sl"
            elif high >= p.tp:
                exit_price = p.tp
                exit_reason = "bt_tp"
        else:
            if high >= p.sl:
                exit_price = p.sl
                exit_reason = "bt_sl"
            elif low <= p.tp:
                exit_price = p.tp
                exit_reason = "bt_tp"

        if exit_price is not None:
            p.exit = exit_price
            p.closed_at = bar_idx
            raw_pnl = (p.exit - p.entry) * p.size if p.side == "BUY" else (p.entry - p.exit) * p.size
            p.pnl = apply_costs(raw_pnl)
            p.reason = exit_reason


def enter_if_allowed(
    st: SimState,
    cfg: SimConfig,
    bar_idx: int,
    symbol: str,
    side: str,
    price: float,
    atr: Optional[float],
) -> Optional[Position]:
    if cfg.max_positions > 0:
        alive = [p for p in st.open_positions if p.closed_at is None]
        if len(alive) >= cfg.max_positions:
            return None

    if bar_idx - st.last_entry_bar < cfg.cooldown_bars:
        return None

    size = cfg.fixed_lot

    if atr is None or atr <= 0:
        sl_dist = 100.0
    else:
        sl_dist = cfg.sl_atr_mult * atr

    tp_mult = cfg.tp_atr_mult
    if side == "BUY":
        sl = price - sl_dist
        tp = price + tp_mult * sl_dist
    else:
        sl = price + sl_dist
        tp = price - tp_mult * sl_dist

    p = Position(
        symbol=symbol,
        side=side,
        entry=price,
        size=size,
        sl=sl,
        tp=tp,
        open_bar=bar_idx,
    )
    st.open_positions.append(p)
    st.last_entry_bar = bar_idx
    return p


def flatten_all(st: SimState, close: float, bar_idx: int) -> None:
    for p in st.open_positions:
        if p.closed_at is not None:
            continue
        p.exit = close
        p.closed_at = bar_idx
        raw_pnl = (p.exit - p.entry) * p.size if p.side == "BUY" else (p.entry - p.exit) * p.size
        p.pnl = apply_costs(raw_pnl)
        if not p.reason:
            p.reason = "bt_eod"


# ---------------------------------------------------------------------------
# Metrics / output
# ---------------------------------------------------------------------------


def metrics(positions: List[Position]) -> Dict[str, float]:
    closed = [p for p in positions if p.closed_at is not None]
    trades = len(closed)
    if trades == 0:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "exp": 0.0, "dd": 0.0}

    pnl_list = [p.pnl for p in closed]
    wins = [p for p in closed if p.pnl > 0]
    losses = [p for p in closed if p.pnl < 0]

    gross_win = sum(p.pnl for p in wins)
    gross_loss = sum(p.pnl for p in losses)

    pf = gross_win / abs(gross_loss) if gross_loss < 0 else 0.0
    wr = 100.0 * len(wins) / trades
    exp = sum(pnl_list) / trades

    eq = 0.0
    peak = 0.0
    dd = 0.0
    for p in closed:
        eq += p.pnl
        if eq > peak:
            peak = eq
        dd = min(dd, eq - peak)

    return {"trades": trades, "pf": pf, "wr": wr, "exp": exp, "dd": dd}


def save_trades_csv(
    symbol: str,
    out_path: Path,
    positions: List[Position],
    rows: List[Dict[str, str]],
) -> Path:
    closed = [p for p in positions if p.closed_at is not None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts",
                "ticket",
                "symbol",
                "side",
                "volume",
                "entry",
                "close",
                "realized_quote",
                "realized_ccy",
                "reason",
            ]
        )
        ticket = 1
        for p in closed:
            idx = p.closed_at if p.closed_at is not None else 0
            if idx < 0:
                idx = 0
            if idx >= len(rows):
                idx = len(rows) - 1
            ts = rows[idx].get("time") or rows[idx].get("ts") or ""
            writer.writerow(
                [
                    ts,
                    ticket,
                    symbol,
                    f"{p.side}",
                    f"{p.size:.2f}",
                    f"{p.entry:.5f}",
                    f"{(p.exit if p.exit is not None else p.entry):.5f}",
                    f"{p.pnl:.2f}",
                    f"{p.pnl:.2f}",
                    p.reason or "",
                ]
            )
            ticket += 1
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Simple backtest runner (EMA/RSI or ML).")
    parser.add_argument("--csv", required=True, help="Input OHLC(+features) CSV.")
    parser.add_argument("--symbol", required=True, help="Symbol name (e.g., US100Z25.sim).")
    parser.add_argument("--tf", required=True, help="Timeframe (e.g., H1).")
    parser.add_argument(
        "--signal",
        choices=["demo", "ml"],
        default="demo",
        help="Signal source: demo (EMA/RSI) or ml (XGBoost).",
    )
    parser.add_argument("--out", required=True, help="Output trades CSV path.")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[bt] ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(csv_path)
    if not rows:
        print("[bt] ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    atr = compute_atr(rows, period=14)

    session_filters = load_session_filters()
    allowed_sessions = [s.upper() for s in session_filters.get(args.symbol, [])]

    sigs: List[int]
    used_ml = False
    long_th = 0.60
    short_th = 0.40

    if args.signal == "ml":
        long_th, short_th = load_profile_thresholds(args.symbol, args.tf)
        sigs, used_ml = ml_signals(rows, args.symbol, args.tf, long_th, short_th)
        if used_ml:
            print(
                f"[bt] Using ML mode with XGB model for {args.symbol} {args.tf} "
                f"(P_LONG={long_th:.2f}, P_SHORT={short_th:.2f})"
            )
        else:
            if not HAVE_XGB_INF:
                reason = f"HAVE_XGB_INF=False (import error={_ML_IMPORT_ERROR})"
            elif not XGB_FEATURE_COLS:
                reason = "XGB_FEATURE_COLS is empty"
            else:
                reason = "no bar had full, non-NaN feature set"
            print(
                "[bt] WARNING: ML backend not used ("
                + reason
                + "); falling back to demo EMA/RSI signals."
            )
            sigs = demo_signals(rows)
    else:
        sigs = demo_signals(rows)

    cfg = SimConfig()
    st = SimState()

    for i, row in enumerate(rows):
        h = to_float(row.get("high"))
        l = to_float(row.get("low"))
        c = to_float(row.get("close"))
        if h is None or l is None or c is None:
            continue

        dt = parse_time_to_dt(row.get("time") or row.get("ts") or "")
        if dt is not None:
            session = classify_session(dt)
        else:
            session = "UNKNOWN"

        session_ok = True
        if allowed_sessions:
            session_ok = session in allowed_sessions

        raw_sig = sigs[i]
        sig = raw_sig if session_ok else 0

        update_positions(st, cfg, i, h, l)

        if sig != 0:
            side = "BUY" if sig > 0 else "SELL"
            enter_if_allowed(st, cfg, i, args.symbol, side, c, atr[i])

    last_close = to_float(rows[-1].get("close"))
    if last_close is not None:
        flatten_all(st, last_close, len(rows) - 1)

    out_path = Path(args.out)
    out = save_trades_csv(args.symbol, out_path, st.open_positions, rows)
    m = metrics(st.open_positions)
    print(f"[bt] saved trades -> {out}")
    print(
        "[bt] metrics:",
        f"trades={m['trades']}  PF={m['pf']:.2f}  WR={m['wr']:.2f}%  "
        f"EXP={m['exp']:.2f}  DD={m['dd']:.2f}",
    )


if __name__ == "__main__":
    main()