#!/usr/bin/env python
"""
bt_session_report.py

Analyze backtest trades (bt_*.csv) by session.

- Reads one or more trades CSVs that look like your backtest outputs:
    ts,ticket,symbol,side,volume,entry,close,realized_quote,realized_ccy,reason

- Assumes:
    * ts is epoch seconds (float) in UTC (as written by backtest/runner.py)
    * Filenames are like: bt_<symbol>_<tf>_<signal>.csv
      e.g. bt_US100Z25.sim_H1_ml.csv

- Outputs a CSV with one row per (symbol, tf, signal, session):

    symbol,tf,signal,session,trades,wr,pf,avg_pnl,gross_pnl,max_dd

Session buckets (UTC):
    ASIA   = 00:00 - 06:59
    LONDON = 07:00 - 12:59
    NY     = 13:00 - 20:59
    OFF    = 21:00 - 23:59
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def parse_fname(path: Path) -> Tuple[str, str, str]:
    """
    Parse backtest filename of the form bt_<symbol>_<tf>_<signal>.csv
    Returns (symbol, tf, signal). Falls back to ('?', '?', '?') on failure.
    """
    name = path.name
    if not name.startswith("bt_") or not name.endswith(".csv"):
        return "?", "?", "?"
    core = name[3:-4]  # strip "bt_" and ".csv"
    parts = core.split("_")
    if len(parts) < 3:
        return "?", "?", "?"
    symbol = parts[0]
    tf = parts[1]
    signal = parts[2]
    return symbol, tf, signal


def session_from_hour(h: int) -> str:
    """
    Map a UTC hour to a session label.

      ASIA   = 00:00 - 06:59
      LONDON = 07:00 - 12:59
      NY     = 13:00 - 20:59
      OFF    = 21:00 - 23:59
    """
    if 0 <= h <= 6:
        return "ASIA"
    if 7 <= h <= 12:
        return "LONDON"
    if 13 <= h <= 20:
        return "NY"
    return "OFF"


def compute_dd(pnls: Iterable[float]) -> float:
    """
    Compute max drawdown over a sequence of trade PnLs.
    """
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for p in pnls:
        eq += float(p)
        if eq > peak:
            peak = eq
        dd = min(dd, eq - peak)
    return float(dd)


@dataclass
class SessionStat:
    symbol: str
    tf: str
    signal: str
    session: str
    trades: int
    wr: float
    pf: float
    avg_pnl: float
    gross_pnl: float
    max_dd: float


def analyze_file(path: Path) -> List[SessionStat]:
    """
    Analyze a single trades CSV. If it doesn't look like a trades file,
    return [] and log a warning instead of raising.
    """
    symbol, tf, signal = parse_fname(path)

    df = pd.read_csv(path)
    if df.empty:
        print(f"[bt_session] {path}: empty, skipping.")
        return []

    if "realized_quote" not in df.columns:
        # This is probably an aggregate (like bt_session_stats_*) or some other non-trades CSV
        print(f"[bt_session] WARNING: {path} has no 'realized_quote' column. Skipping.")
        return []

    # Parse ts (epoch seconds) into UTC datetime
    def to_dt(val) -> Optional[datetime]:
        if pd.isna(val):
            return None
        try:
            ts = float(val)
        except Exception:
            return None
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    df["dt"] = df["ts"].apply(to_dt)
    df = df[df["dt"].notna()].copy()

    if df.empty:
        print(f"[bt_session] {path}: no valid time rows after parsing, skipping.")
        return []

    df["hour"] = df["dt"].apply(lambda d: int(d.hour))
    df["session"] = df["hour"].apply(session_from_hour)

    df["pnl"] = df["realized_quote"].astype(float)

    stats: List[SessionStat] = []

    for session, g in df.groupby("session"):
        pnls = g["pnl"].tolist()
        if not pnls:
            continue

        trades = len(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_win = sum(wins) if wins else 0.0
        gross_loss = -sum(losses) if losses else 0.0

        if gross_loss > 0:
            pf = gross_win / gross_loss
        else:
            pf = float("inf") if gross_win > 0 else 0.0

        wr = (len(wins) / trades) * 100.0 if trades > 0 else 0.0
        avg_pnl = mean(pnls) if pnls else 0.0
        gross_pnl = sum(pnls)
        max_dd = compute_dd(pnls)

        stats.append(
            SessionStat(
                symbol=symbol,
                tf=tf,
                signal=signal,
                session=str(session),
                trades=trades,
                wr=float(wr),
                pf=float(pf),
                avg_pnl=float(avg_pnl),
                gross_pnl=float(gross_pnl),
                max_dd=float(max_dd),
            )
        )

    return stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Analyze backtest trades by session")
    ap.add_argument(
        "--trades-glob",
        default=str(REPORTS / "bt_*_H1_ml.csv"),
        help="Glob pattern for backtest trades CSVs (default: reports/bt_*_H1_ml.csv)",
    )
    ap.add_argument(
        "--out",
        default=str(REPORTS / "bt_session_stats_H1_ml.csv"),
        help="Output CSV path (default: reports/bt_session_stats_H1_ml.csv)",
    )
    args = ap.parse_args(argv)

    matches = glob.glob(args.trades_glob)
    if not matches:
        raise SystemExit(f"[bt_session] No files matched: {args.trades_glob}")

    # Hard-filter obvious non-trade aggregates like our own output file
    filtered: List[Path] = []
    for m in matches:
        p = Path(m)
        if "session_stats" in p.name:
            print(f"[bt_session] Skipping non-trades file: {p}")
            continue
        filtered.append(p)

    if not filtered:
        raise SystemExit("[bt_session] All matched files were non-trades; nothing to do.")

    all_stats: List[SessionStat] = []

    print(f"[bt_session] Root: {ROOT}")
    print(f"[bt_session] Glob: {args.trades_glob}")
    print(f"[bt_session] Files (after filter): {len(filtered)}")

    for p in filtered:
        print(f"[bt_session] Analyzing {p} ...")
        stats = analyze_file(p)
        all_stats.extend(stats)

    if not all_stats:
        raise SystemExit("[bt_session] No stats produced (no trades or all files skipped).")

    rows = [
        {
            "symbol": s.symbol,
            "tf": s.tf,
            "signal": s.signal,
            "session": s.session,
            "trades": s.trades,
            "wr": s.wr,
            "pf": s.pf,
            "avg_pnl": s.avg_pnl,
            "gross_pnl": s.gross_pnl,
            "max_dd": s.max_dd,
        }
        for s in all_stats
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)

    print(f"[bt_session] Wrote session stats -> {out_path}")
    print("[bt_session] Done.")


if __name__ == "__main__":
    main()