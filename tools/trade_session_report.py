#!/usr/bin/env python
"""
trade_session_report.py

Time-of-day / session analytics for Trader 2.0.

Works with:
- Live trades:   data/trades.csv
- Backtests:     reports/bt_*.csv

Features:
- Optional filters:
    --symbol SYMBOL
    --since-days N
- Classifies trades into sessions based on ts:
    ASIA   = 00:00–06:59
    LONDON = 07:00–12:59
    NY     = 13:00–19:59
    OFF    = 20:00–23:59
    UNKNOWN when ts cannot be parsed
- Computes metrics per:
    - Overall
    - Session
    - Symbol × session
- Metrics:
    trades, wins, losses, WR, pnl, avg_pnl, PF, max_dd

Usage examples:

    # Live trades last 30 days, all symbols
    python tools/trade_session_report.py `
        --trades data/trades.csv `
        --since-days 30

    # Backtest session stats for US100Z25.sim H1 ML
    python tools/trade_session_report.py `
        --trades reports/bt_US100Z25.sim_H1_ml.csv `
        --symbol US100Z25.sim `
        --since-days 365

    # Write CSV summary
    python tools/trade_session_report.py `
        --trades data/trades.csv `
        --since-days 60 `
        --out reports/trade_session_stats_live_60d.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Time parsing & session classification
# ---------------------------------------------------------------------------


def parse_ts(ts: str) -> Optional[datetime]:
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None

    dt_obj: Optional[datetime] = None

    # Try ISO first
    try:
        dt_obj = datetime.fromisoformat(s)
    except Exception:
        dt_obj = None

    # Fallback: "YYYY-MM-DD HH:MM:SS"
    if dt_obj is None:
        try:
            dt_obj = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj


def classify_session(dt_obj: datetime) -> str:
    h = dt_obj.hour
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 13:
        return "LONDON"
    if 13 <= h < 20:
        return "NY"
    return "OFF"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class TradeRow:
    ts: datetime
    symbol: str
    pnl: float
    session: str


@dataclass
class Metrics:
    trades: int
    wins: int
    losses: int
    flats: int
    wr: float
    pnl: float
    avg_pnl: float
    pf: float
    max_dd: float


def compute_metrics(pnls: Iterable[float]) -> Metrics:
    pnl_list = list(pnls)
    trades = len(pnl_list)
    if trades == 0:
        return Metrics(
            trades=0,
            wins=0,
            losses=0,
            flats=0,
            wr=0.0,
            pnl=0.0,
            avg_pnl=0.0,
            pf=0.0,
            max_dd=0.0,
        )

    wins = sum(1 for x in pnl_list if x > 0)
    losses = sum(1 for x in pnl_list if x < 0)
    flats = trades - wins - losses

    pnl_sum = sum(pnl_list)
    avg_pnl = pnl_sum / trades

    gross_win = sum(x for x in pnl_list if x > 0)
    gross_loss = sum(x for x in pnl_list if x < 0)
    pf = gross_win / abs(gross_loss) if gross_loss < 0 else 0.0

    wr = 100.0 * wins / trades if trades > 0 else 0.0

    # Max drawdown on an equity curve built from these pnls in order
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for x in pnl_list:
        eq += x
        if eq > peak:
            peak = eq
        dd = min(dd, eq - peak)

    return Metrics(
        trades=trades,
        wins=wins,
        losses=losses,
        flats=flats,
        wr=wr,
        pnl=pnl_sum,
        avg_pnl=avg_pnl,
        pf=pf,
        max_dd=dd,
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_trades(
    path: Path,
    symbol_filter: Optional[str],
    since_days: Optional[float],
) -> List[TradeRow]:
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {path}")

    rows_raw: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_raw.append(row)

    if not rows_raw:
        return []

    # Determine cutoff
    cutoff_dt: Optional[datetime] = None
    if since_days is not None and since_days > 0:
        # Use "now" in UTC
        now = datetime.now(timezone.utc)
        cutoff_dt = now - timedelta(days=float(since_days))

    trades: List[TradeRow] = []

    for row in rows_raw:
        sym = (row.get("symbol") or "").strip()
        if symbol_filter and sym != symbol_filter:
            continue

        ts_str = row.get("ts") or row.get("time") or ""
        dt = parse_ts(ts_str)
        if dt is None:
            # No timestamp → group as UNKNOWN with ts=epoch start (for ordering)
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
            session = "UNKNOWN"
        else:
            if cutoff_dt is not None and dt < cutoff_dt:
                continue
            session = classify_session(dt)

        # PnL: prefer realized_quote, then realized_ccy, else 0
        raw_pnl_str = row.get("realized_quote") or row.get("realized_ccy") or "0"
        try:
            pnl = float(raw_pnl_str)
            if math.isnan(pnl):
                pnl = 0.0
        except Exception:
            pnl = 0.0

        trades.append(TradeRow(ts=dt, symbol=sym, pnl=pnl, session=session))

    # Sort by time so DD is meaningful
    trades.sort(key=lambda t: t.ts)
    return trades


def analyze_sessions(trades: List[TradeRow]) -> Tuple[
    Metrics,
    Dict[str, Metrics],
    Dict[Tuple[str, str], Metrics],
]:
    # Overall metrics
    overall_metrics = compute_metrics(t.pnl for t in trades)

    # Per session
    by_session: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        by_session[t.session].append(t.pnl)

    per_session: Dict[str, Metrics] = {}
    for sess, pnls in by_session.items():
        per_session[sess] = compute_metrics(pnls)

    # Per symbol × session
    by_sym_session: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for t in trades:
        key = (t.symbol, t.session)
        by_sym_session[key].append(t.pnl)

    per_sym_session: Dict[Tuple[str, str], Metrics] = {}
    for key, pnls in by_sym_session.items():
        per_sym_session[key] = compute_metrics(pnls)

    return overall_metrics, per_session, per_sym_session


def print_metrics_table(title: str, rows: List[Tuple[str, Metrics]]) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not rows:
        print("  (no trades)")
        return

    header = f"{'GROUP':<20} {'TRD':>4} {'W':>3} {'L':>3} {'WR%':>6} {'PNL':>10} {'AVG':>8} {'PF':>6} {'MAX_DD':>10}"
    print(header)
    print("-" * len(header))
    for label, m in rows:
        print(
            f"{label:<20} "
            f"{m.trades:>4d} "
            f"{m.wins:>3d} "
            f"{m.losses:>3d} "
            f"{m.wr:>6.1f} "
            f"{m.pnl:>10.2f} "
            f"{m.avg_pnl:>8.2f} "
            f"{m.pf:>6.2f} "
            f"{m.max_dd:>10.2f}"
        )


def write_csv_out(
    out_path: Path,
    overall: Metrics,
    per_session: Dict[str, Metrics],
    per_sym_session: Dict[Tuple[str, str], Metrics],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group_type",  # overall / session / symbol_session
                "symbol",
                "session",
                "trades",
                "wins",
                "losses",
                "flats",
                "wr_pct",
                "pnl",
                "avg_pnl",
                "pf",
                "max_dd",
            ]
        )

        # Overall
        writer.writerow(
            [
                "overall",
                "",
                "",
                overall.trades,
                overall.wins,
                overall.losses,
                overall.flats,
                f"{overall.wr:.2f}",
                f"{overall.pnl:.2f}",
                f"{overall.avg_pnl:.2f}",
                f"{overall.pf:.4f}",
                f"{overall.max_dd:.2f}",
            ]
        )

        # Per session
        for sess, m in sorted(per_session.items()):
            writer.writerow(
                [
                    "session",
                    "",
                    sess,
                    m.trades,
                    m.wins,
                    m.losses,
                    m.flats,
                    f"{m.wr:.2f}",
                    f"{m.pnl:.2f}",
                    f"{m.avg_pnl:.2f}",
                    f"{m.pf:.4f}",
                    f"{m.max_dd:.2f}",
                ]
            )

        # Per symbol × session
        for (sym, sess), m in sorted(per_sym_session.items()):
            writer.writerow(
                [
                    "symbol_session",
                    sym,
                    sess,
                    m.trades,
                    m.wins,
                    m.losses,
                    m.flats,
                    f"{m.wr:.2f}",
                    f"{m.pnl:.2f}",
                    f"{m.avg_pnl:.2f}",
                    f"{m.pf:.4f}",
                    f"{m.max_dd:.2f}",
                ]
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Time-of-day / session performance report for trades CSV."
    )
    parser.add_argument(
        "--trades",
        default="data/trades.csv",
        help="Trades CSV path (default: data/trades.csv).",
    )
    parser.add_argument(
        "--symbol",
        help="Optional symbol filter (e.g., US100Z25.sim).",
    )
    parser.add_argument(
        "--since-days",
        type=float,
        help="Only include trades with ts >= now - since_days.",
    )
    parser.add_argument(
        "--out",
        help="Optional output CSV path for aggregated stats.",
    )

    args = parser.parse_args(argv)

    trades_path = Path(args.trades)

    print(f"[session_report] Trades: {trades_path}")
    if args.symbol:
        print(f"[session_report] Symbol filter: {args.symbol}")
    if args.since_days is not None:
        print(f"[session_report] Since days: {args.since_days}")

    trades = load_trades(trades_path, args.symbol, args.since_days)
    print(f"[session_report] Loaded trades (after filters): {len(trades)}")

    overall, per_session, per_sym_session = analyze_sessions(trades)

    # Print overall
    print_metrics_table("Overall performance", [("ALL", overall)])

    # Per-session
    sess_rows = sorted(per_session.items(), key=lambda kv: kv[0])
    print_metrics_table(
        "Performance by session (ASIA / LONDON / NY / OFF / UNKNOWN)",
        [(sess, m) for sess, m in sess_rows],
    )

    # Per symbol × session
    sym_sess_rows = sorted(per_sym_session.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    pretty_rows: List[Tuple[str, Metrics]] = []
    for (sym, sess), m in sym_sess_rows:
        label = f"{sym}:{sess}"
        pretty_rows.append((label, m))

    print_metrics_table("Performance by symbol × session", pretty_rows)

    # Optional CSV output
    if args.out:
        out_path = Path(args.out)
        write_csv_out(out_path, overall, per_session, per_sym_session)
        print(f"\n[session_report] Wrote CSV summary → {out_path}")


if __name__ == "__main__":
    main()