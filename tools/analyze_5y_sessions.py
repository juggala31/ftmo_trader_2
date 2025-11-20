#!/usr/bin/env python
"""
tools/analyze_5y_sessions.py

Analyze a 5-year ML backtest matrix by time-of-day sessions.

- Reads trades from reports/bt_*_ml_5y.csv (default glob).
- For each CSV:
    * infers (symbol, timeframe) from the filename
    * parses trade timestamp (ts) as UTC
    * assigns each trade to a session:
        ASIA   = 00:00–06:59
        LONDON = 07:00–12:59
        NY     = 13:00–19:59
        OFF    = 20:00–23:59
- Computes PF, win-rate, etc. per (symbol, timeframe, session).
- Writes reports/bt_session_matrix_5y.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]  # tools/.. -> project root
REPORTS_ROOT = ROOT / "reports"


def parse_ts(ts: str) -> Optional[datetime]:
    """
    Parse timestamps like:
        2025-08-26 23:00:00+00:00
        2023-01-01 12:34:56
    into timezone-aware UTC datetimes.
    """
    if ts is None:
        return None
    s = str(ts).strip()
    if not s:
        return None

    dt_obj: Optional[datetime] = None

    # Try ISO-8601 first
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
    else:
        dt_obj = dt_obj.astimezone(timezone.utc)

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


@dataclass
class SessionStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    pnl_sum: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0

    def update(self, pnl: float) -> None:
        self.trades += 1
        self.pnl_sum += pnl
        if pnl > 0:
            self.wins += 1
            self.gross_win += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += pnl
        else:
            self.flats += 1

    def to_row(self, symbol: str, tf: str, session: str) -> List[str]:
        if self.trades <= 0:
            wr = 0.0
            pf = 0.0
            avg_pnl = 0.0
        else:
            wr = 100.0 * self.wins / self.trades
            avg_pnl = self.pnl_sum / self.trades
            pf = (
                self.gross_win / abs(self.gross_loss)
                if self.gross_loss < 0
                else 0.0
            )

        return [
            symbol,
            tf,
            session,
            str(self.trades),
            str(self.wins),
            str(self.losses),
            str(self.flats),
            f"{wr:.2f}",
            f"{self.pnl_sum:.2f}",
            f"{avg_pnl:.2f}",
            f"{pf:.2f}",
        ]


def infer_symbol_tf(path: Path) -> Tuple[str, str]:
    """
    Expect files like:
        bt_BTCX25.sim_H1_ml_5y.csv
        bt_US30Z25.sim_M30_ml_5y.csv
    We strip prefix 'bt_' and suffix '.csv' and then split on '_':
        BTCX25.sim_H1_ml_5y -> ["BTCX25.sim", "H1", "ml", "5y"]
    """
    name = path.name
    if not name.startswith("bt_") or not name.endswith(".csv"):
        raise ValueError(f"Unexpected file name for session analysis: {name}")
    core = name[len("bt_") : -len(".csv")]
    parts = core.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer symbol/tf from {name}")
    symbol = parts[0]
    tf = parts[1]
    return symbol, tf


def analyze(glob: str) -> Path:
    paths = sorted(REPORTS_ROOT.glob(glob))
    if not paths:
        raise SystemExit(f"No trade files found for glob='{glob}' in {REPORTS_ROOT}")

    stats: Dict[Tuple[str, str, str], SessionStats] = {}

    for path in paths:
        symbol, tf = infer_symbol_tf(path)
        print(f"[analyze_5y_sessions] Loading {path.name} ({symbol} {tf})", flush=True)

        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get("ts")
                dt_obj = parse_ts(ts)
                if dt_obj is None:
                    session = "UNKNOWN"
                else:
                    session = classify_session(dt_obj)

                pnl_str = row.get("realized_ccy", "") or "0"
                try:
                    pnl = float(pnl_str)
                except Exception:
                    pnl = 0.0

                key = (symbol, tf, session)
                if key not in stats:
                    stats[key] = SessionStats()
                stats[key].update(pnl)

    out_path = REPORTS_ROOT / "bt_session_matrix_5y.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "symbol",
                "tf",
                "session",
                "trades",
                "wins",
                "losses",
                "flats",
                "wr_pct",
                "pnl_sum",
                "avg_pnl",
                "pf",
            ]
        )
        for (symbol, tf, session), st in sorted(
            stats.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
        ):
            writer.writerow(st.to_row(symbol, tf, session))

    print(f"[analyze_5y_sessions] Wrote {out_path}", flush=True)
    return out_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze 5-year ML backtest matrix by sessions."
    )
    p.add_argument(
        "--trades-glob",
        default="bt_*_ml_5y.csv",
        help="Glob for backtest trades in reports/ (default: bt_*_ml_5y.csv).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    analyze(args.trades_glob)


if __name__ == "__main__":
    main()