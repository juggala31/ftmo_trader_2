import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List


@dataclass
class SymbolStats:
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0  # negative sum
    max_dd: float = 0.0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def wr(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades > 0 else 0.0

    def pf(self) -> float:
        if self.gross_loss >= -1e-9:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def load_stats(
    path: Path,
    since_days: float,
    symbol_filter: Optional[str],
) -> Dict[str, SymbolStats]:
    stats: Dict[str, SymbolStats] = {}

    if not path.exists():
        print(f"[report_trades] trades file not found: {path}")
        return stats

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts: Optional[float] = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    total_rows = 0
    used_rows = 0

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1

            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue
            if symbol_filter and sym != symbol_filter:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if cutoff_ts is not None and ts_val is not None and ts_val < cutoff_ts:
                continue

            pnl_str = (row.get("realized_ccy") or "").strip()
            if not pnl_str:
                pnl_str = (row.get("realized_quote") or "").strip()
            pnl = parse_float(pnl_str) or 0.0

            used_rows += 1

            s = stats.get(sym)
            if s is None:
                s = SymbolStats(symbol=sym)
                stats[sym] = s

            s.trades += 1
            if pnl > 0:
                s.wins += 1
                s.gross_profit += pnl
            elif pnl < 0:
                s.losses += 1
                s.gross_loss += pnl
            else:
                s.flats += 1
            s.pnl += pnl

            # equity curve for DD
            # we recompute per symbol by walking in row order; not perfect across shuffled files
            # but assuming roughly chronological this is fine.
            # Track equity and running peak for each symbol separately
            # We'll store a temporary "eq" on the object (not persisted across calls)
            if not hasattr(s, "_eq"):
                s._eq = 0.0  # type: ignore[attr-defined]
                s._peak = 0.0  # type: ignore[attr-defined]

            s._eq += pnl  # type: ignore[attr-defined]
            if s._eq > s._peak:  # type: ignore[attr-defined]
                s._peak = s._eq  # type: ignore[attr-defined]
            dd = s._peak - s._eq  # type: ignore[attr-defined]
            if dd > s.max_dd:
                s.max_dd = dd

            if s.first_ts is None or (ts_val is not None and ts_val < s.first_ts):
                s.first_ts = ts_val
            if s.last_ts is None or (ts_val is not None and ts_val > s.last_ts):
                s.last_ts = ts_val

    print(f"[report_trades] total rows in file: {total_rows}")
    print(f"[report_trades] rows used after filters: {used_rows}")
    if used_rows == 0:
        print("[report_trades] no trades in the requested window / filter.")
    return stats


def fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


def print_report(stats: Dict[str, SymbolStats]) -> None:
    if not stats:
        return

    symbols = sorted(stats.keys())
    print("")
    print("Per-symbol summary")
    print("------------------")
    header = (
        f"{'SYMBOL':<12} {'TRADES':>6} {'WINS':>5} {'LOSS':>5} "
        f"{'WR%':>7} {'PNL':>12} {'PF':>7} {'MAX_DD':>12}"
    )
    print(header)
    print("-" * len(header))

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_flats = 0
    total_pnl = 0.0
    total_gp = 0.0
    total_gl = 0.0

    for sym in symbols:
        s = stats[sym]
        total_trades += s.trades
        total_wins += s.wins
        total_losses += s.losses
        total_flats += s.flats
        total_pnl += s.pnl
        total_gp += s.gross_profit
        total_gl += s.gross_loss

        print(
            f"{sym:<12} {s.trades:>6} {s.wins:>5} {s.losses:>5} "
            f"{s.wr():>7.2f} {s.pnl:>12.2f} {s.pf():>7.2f} {s.max_dd:>12.2f}"
        )

    print("")
    print("Totals")
    print("------")
    total_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    if total_gl >= -1e-9:
        total_pf = float("inf") if total_gp > 0 else 0.0
    else:
        total_pf = total_gp / abs(total_gl)

    print(f"Total trades : {total_trades}")
    print(f"Wins / Losses: {total_wins} / {total_losses} (flats={total_flats})")
    print(f"Win rate     : {total_wr:.2f}%")
    print(f"PNL          : {total_pnl:.2f}")
    print(f"Profit factor: {total_pf:.2f}")

    # overall date range
    first_ts = None
    last_ts = None
    for s in stats.values():
        if s.first_ts is not None:
            if first_ts is None or s.first_ts < first_ts:
                first_ts = s.first_ts
        if s.last_ts is not None:
            if last_ts is None or s.last_ts > last_ts:
                last_ts = s.last_ts

    print(f"Date range   : {fmt_ts(first_ts)} -> {fmt_ts(last_ts)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Summarize trades CSV (per-symbol PnL, WR, PF, DD, totals)."
    )
    ap.add_argument(
        "--trades",
        default="data/trades.csv",
        help="Path to trades CSV (default: data/trades.csv).",
    )
    ap.add_argument(
        "--since-days",
        type=float,
        default=365.0,
        help="Lookback window in days (0 = all history, default: 365).",
    )
    ap.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol filter (exact match, e.g. US100Z25.sim).",
    )
    args = ap.parse_args(argv)

    path = Path(args.trades)
    print(f"[report_trades] trades:     {path}")
    print(f"[report_trades] since_days: {args.since_days}")
    print(f"[report_trades] symbol:     {args.symbol or '(all)'}")

    stats = load_stats(path, since_days=args.since_days, symbol_filter=args.symbol)
    print_report(stats)


if __name__ == "__main__":
    main()