#!/usr/bin/env python
"""
build_session_filters.py

Build ai/session_filters.json from backtest (or live) trades, based on
time-of-day / session performance.

Sessions:
    ASIA   = 00:00–06:59
    LONDON = 07:00–12:59
    NY     = 13:00–19:59
    OFF    = 20:00–23:59
    UNKNOWN when ts cannot be parsed

Input:
    --trades-glob "reports/bt_*_H1_ml.csv"   (default)
      Any CSVs with columns: ts, symbol, realized_quote/realized_ccy

Filters (per symbol × session group):
    --min-trades N   (default 30)
    --min-pf PF      (default 1.1)

Output:
    ai/session_filters.json
    {
      "US100Z25.sim": ["LONDON", "NY"],
      "BTCX25.sim":   ["NY"]
    }

Example:

    python tools/build_session_filters.py `
      --trades-glob "reports/bt_*_H1_ml.csv" `
      --min-trades 30 `
      --min-pf 1.1 `
      --out "ai/session_filters.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Time parsing & sessions
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
class Metrics:
    trades: int
    wins: int
    losses: int
    flats: int
    wr: float
    pnl: float
    avg_pnl: float
    pf: float


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

    return Metrics(
        trades=trades,
        wins=wins,
        losses=losses,
        flats=flats,
        wr=wr,
        pnl=pnl_sum,
        avg_pnl=avg_pnl,
        pf=pf,
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def load_trades_from_file(path: Path) -> List[Tuple[str, str, float]]:
    """
    Return list of (symbol, session, pnl).
    """
    out: List[Tuple[str, str, float]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue

            ts_str = row.get("ts") or row.get("time") or ""
            dt = parse_ts(ts_str)
            if dt is None:
                session = "UNKNOWN"
            else:
                session = classify_session(dt)

            raw_pnl_str = row.get("realized_quote") or row.get("realized_ccy") or "0"
            try:
                pnl = float(raw_pnl_str)
                if math.isnan(pnl):
                    pnl = 0.0
            except Exception:
                pnl = 0.0

            out.append((sym, session, pnl))

    return out


def build_session_filters(
    trades_files: List[Path],
    min_trades: int,
    min_pf: float,
) -> Dict[str, List[str]]:
    """
    Return mapping: symbol -> [allowed_sessions].
    """
    # Collect pnls by symbol×session
    buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for path in trades_files:
        rows = load_trades_from_file(path)
        for sym, session, pnl in rows:
            buckets[(sym, session)].append(pnl)

    # Compute metrics and decide which sessions to allow
    sym_to_sessions: Dict[str, List[str]] = defaultdict(list)

    for (sym, session), pnls in buckets.items():
        m = compute_metrics(pnls)
        if m.trades < min_trades:
            continue
        if m.pf < min_pf:
            continue
        sym_to_sessions[sym].append(session)

    # Sort sessions for stability
    for sym, sess_list in sym_to_sessions.items():
        # enforce a stable order: ASIA < LONDON < NY < OFF < UNKNOWN
        order = {"ASIA": 0, "LONDON": 1, "NY": 2, "OFF": 3, "UNKNOWN": 4}
        sess_list.sort(key=lambda s: order.get(s, 99))

    return dict(sym_to_sessions)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build ai/session_filters.json from trades (session stats)."
    )
    parser.add_argument(
        "--trades-glob",
        default="reports/bt_*_H1_ml.csv",
        help="Glob for trades CSVs (default: reports/bt_*_H1_ml.csv).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=30,
        help="Minimum trades per symbol×session to consider (default: 30).",
    )
    parser.add_argument(
        "--min-pf",
        type=float,
        default=1.1,
        help="Minimum profit factor per symbol×session (default: 1.1).",
    )
    parser.add_argument(
        "--out",
        default="ai/session_filters.json",
        help="Output JSON path (default: ai/session_filters.json).",
    )

    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    pattern = str(root / args.trades_glob)
    matches = sorted(Path(p) for p in glob.glob(pattern))

    print(f"[session_filters] Root: {root}")
    print(f"[session_filters] Glob: {args.trades_glob}")
    print(f"[session_filters] Files: {len(matches)}")
    for p in matches:
        print(f"  - {p}")

    if not matches:
        print("[session_filters] No matching trades files. Nothing to do.")
        return

    print(
        f"[session_filters] Criteria: min_trades={args.min_trades}, "
        f"min_pf={args.min_pf:.2f}"
    )

    mapping = build_session_filters(
        trades_files=matches,
        min_trades=args.min_trades,
        min_pf=args.min_pf,
    )

    if not mapping:
        print("[session_filters] No symbol×session combos met the criteria.")
        print("[session_filters] Not writing session_filters.json (would be empty).")
        return

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"[session_filters] Wrote {out_path}")
    print("[session_filters] Contents:")
    for sym, sessions in sorted(mapping.items()):
        print(f"  {sym}: {sessions}")


if __name__ == "__main__":
    main()