#!/usr/bin/env python
"""
tools/run_5y_bt_matrix.py

Comprehensive 5-year ML backtest matrix.

- Runs ML backtests for multiple symbols × timeframes using backtest.runner.
- Uses the existing ai/datasets/csv/{TF}/{symbol}.csv files (5 years of history).
- Thresholds come from ai/ai_profile.json (same as live + standard runner).
- Writes trades to reports/bt_{symbol}_{tf}_ml_5y.csv

You can later post-process these with tools/analyze_5y_sessions.py
to get PF/WR by symbol × timeframe × session (ASIA/LONDON/NY/OFF).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]  # tools/.. -> project root
sys.path.insert(0, str(ROOT))

from backtest.runner import main as bt_main  # type: ignore


DEFAULT_SYMBOLS = [
    "US30Z25.sim",
    "US100Z25.sim",
    "US500Z25.sim",
    "XAUZ25.sim",
    "USOILZ25.sim",
    "BTCX25.sim",
]

DEFAULT_TFS = ["M15", "M30", "H1", "H4"]


def iter_jobs(
    symbols: Iterable[str], tfs: Iterable[str]
) -> Iterable[Tuple[str, str, Path, Path]]:
    """
    Yield (symbol, tf, csv_path, out_path) for all valid combos.
    """
    csv_root = ROOT / "ai" / "datasets" / "csv"
    reports_root = ROOT / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    for tf in tfs:
        tf_dir = csv_root / tf
        if not tf_dir.exists():
            print(f"[skip] TF={tf}: no directory {tf_dir}", flush=True)
            continue
        for sym in symbols:
            csv_path = tf_dir / f"{sym}.csv"
            if not csv_path.exists():
                print(f"[skip] {sym} {tf}: {csv_path} missing", flush=True)
                continue
            out_name = f"bt_{sym}_{tf}_ml_5y.csv"
            out_path = reports_root / out_name
            yield sym, tf, csv_path, out_path


def run_all(symbols: List[str], tfs: List[str], overwrite: bool = False) -> None:
    jobs = list(iter_jobs(symbols, tfs))
    if not jobs:
        print("[run_5y_bt_matrix] No jobs to run (no matching CSVs).", flush=True)
        return

    print(
        f"[run_5y_bt_matrix] Jobs: {len(jobs)} combos "
        f"({len(symbols)} symbols × {len(tfs)} timeframes)",
        flush=True,
    )

    for idx, (sym, tf, csv_path, out_path) in enumerate(jobs, start=1):
        if out_path.exists() and not overwrite:
            print(
                f"[{idx}/{len(jobs)}] {sym} {tf}: skipping (exists: {out_path.name})",
                flush=True,
            )
            continue

        print(
            f"[{idx}/{len(jobs)}] {sym} {tf}: running ML backtest "
            f"from {csv_path.name} -> {out_path.name}",
            flush=True,
        )

        argv = [
            "--csv",
            str(csv_path),
            "--symbol",
            sym,
            "--tf",
            tf,
            "--signal",
            "ml",
            "--out",
            str(out_path),
        ]
        try:
            bt_main(argv)
        except SystemExit as e:
            # backtest.runner.main uses argparse; convert SystemExit into log.
            print(f"[ERROR] runner failed for {sym} {tf}: {e}", flush=True)
        except Exception as e:
            print(f"[ERROR] unexpected error for {sym} {tf}: {e}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 5-year ML backtest matrix.")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to backtest (default: common index/gold/oil/BTC .sim set).",
    )
    p.add_argument(
        "--tfs",
        nargs="+",
        default=DEFAULT_TFS,
        help="Timeframes to backtest (default: M15 M30 H1 H4).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bt_*_ml_5y.csv files (default: skip existing).",
    )
    return p.parse_args(argv)


def main_cli(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_all(list(args.symbols), list(args.tfs), overwrite=args.overwrite)


if __name__ == "__main__":
    main_cli()