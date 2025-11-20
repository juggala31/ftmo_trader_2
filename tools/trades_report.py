import argparse
import csv
import datetime as dt
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_TRADES = DATA_DIR / "trades.csv"


@dataclass
class SymbolStats:
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    pnl: float = 0.0
    max_dd: float = 0.0  # positive number
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def update_dd(stats: SymbolStats, pnl_delta: float, ts: Optional[float]) -> None:
    """
    Update running PnL / max drawdown for a single symbol.
    We keep max_dd as a positive number.
    """
    # running equity
    cur_equity = getattr(stats, "_equity", 0.0) + pnl_delta
    setattr(stats, "_equity", cur_equity)

    peak = getattr(stats, "_peak", 0.0)
    if cur_equity > peak:
        peak = cur_equity
        setattr(stats, "_peak", peak)

    dd = peak - cur_equity
    if dd > stats.max_dd:
        stats.max_dd = dd

    # timestamps
    if ts is not None:
        if stats.first_ts is None or ts < stats.first_ts:
            stats.first_ts = ts
        if stats.last_ts is None or ts > stats.last_ts:
            stats.last_ts = ts


def summarize_trades(
    trades_csv: pathlib.Path,
    since_days: Optional[float] = None,
    allowed_symbols: Optional[List[str]] = None,
) -> Dict[str, SymbolStats]:
    stats: Dict[str, SymbolStats] = {}

    if allowed_symbols:
        allowed = set(allowed_symbols)
    else:
        allowed = None

    cutoff_ts: Optional[float] = None
    if since_days is not None and since_days > 0:
        now = dt.datetime.utcnow().timestamp()
        cutoff_ts = now - since_days * 86400.0

    with trades_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get("symbol", "").strip()
            if not symbol:
                continue
            if allowed is not None and symbol not in allowed:
                continue

            ts_raw = row.get("ts", "").strip()
            ts_val = parse_float(ts_raw) if ts_raw else None
            if cutoff_ts is not None and ts_val is not None and ts_val < cutoff_ts:
                continue

            pnl_raw = row.get("realized_ccy") or row.get("realized_quote") or ""
            pnl_val = parse_float(str(pnl_raw)) or 0.0

            sym_stats = stats.get(symbol)
            if sym_stats is None:
                sym_stats = SymbolStats(symbol=symbol)
                stats[symbol] = sym_stats

            sym_stats.trades += 1
            if pnl_val > 0:
                sym_stats.wins += 1
            elif pnl_val < 0:
                sym_stats.losses += 1
            else:
                sym_stats.flats += 1

            sym_stats.pnl += pnl_val
            update_dd(sym_stats, pnl_val, ts_val)

    return stats


def format_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


def print_report(stats: Dict[str, SymbolStats]) -> None:
    if not stats:
        print("[report] No trades found for given filters.")
        return

    header = (
        f"{'Symbol':<12}"
        f"{'Trades':>8}"
        f"{'Wins':>8}"
        f"{'Losses':>8}"
        f"{'Win%':>8}"
        f"{'PnL':>12}"
        f"{'AvgPnL':>12}"
        f"{'MaxDD':>12}"
        f"{'First':>12}"
        f"{'Last':>12}"
    )
    print(header)
    print("-" * len(header))

    symbols = sorted(stats.keys())
    for sym in symbols:
        s = stats[sym]
        win_pct = (s.wins / s.trades * 100.0) if s.trades > 0 else 0.0
        avg_pnl = (s.pnl / s.trades) if s.trades > 0 else 0.0
        line = (
            f"{sym:<12}"
            f"{s.trades:>8d}"
            f"{s.wins:>8d}"
            f"{s.losses:>8d}"
            f"{win_pct:>7.1f}%"
            f"{s.pnl:>12.2f}"
            f"{avg_pnl:>12.2f}"
            f"{s.max_dd:>12.2f}"
            f"{format_ts(s.first_ts):>12}"
            f"{format_ts(s.last_ts):>12}"
        )
        print(line)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Summarize trades.csv by symbol (count, PnL, winrate, drawdown)."
    )
    p.add_argument(
        "--trades-csv",
        type=pathlib.Path,
        default=DEFAULT_TRADES,
        help=f"Path to trades.csv (default: {DEFAULT_TRADES})",
    )
    p.add_argument(
        "--since-days",
        type=float,
        default=None,
        help="Only include trades from the last N days (based on ts epoch).",
    )
    p.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="If given, only include these symbols. Can be passed multiple times.",
    )

    args = p.parse_args(argv)

    trades_csv: pathlib.Path = args.trades_csv
    if not trades_csv.exists():
        print(f"[report] ERROR: trades file not found: {trades_csv}")
        return

    stats = summarize_trades(
        trades_csv=trades_csv,
        since_days=args.since_days,
        allowed_symbols=args.symbol,
    )
    print_report(stats)


if __name__ == "__main__":
    main()