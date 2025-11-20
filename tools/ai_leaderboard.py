import argparse
import csv
import datetime as dt
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_CSV = DATA_DIR / "trades.csv"
REPORTS_DIR = ROOT / "reports" / "ai_leaderboard"


@dataclass
class Trade:
    ts: float
    ticket: int
    symbol: str
    side: str
    volume: float
    entry: float
    close: float
    realized_quote: float
    realized_ccy: float
    reason: str


@dataclass
class Stats:
    n: int = 0
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    pnl_avg: float = 0.0
    winrate_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best: float = 0.0
    worst: float = 0.0


def _parse_row(row: Dict[str, str]) -> Optional[Trade]:
    try:
        ts = float(row["ts"])
        ticket = int(row["ticket"])
        symbol = row["symbol"]
        side = row["side"].upper()
        volume = float(row["volume"])
        entry = float(row["entry"])
        close = float(row["close"])
        realized_quote = float(row["realized_quote"])
        realized_ccy = float(row["realized_ccy"])
        reason = row.get("reason", "")
    except Exception as e:
        print(f"[ai_leaderboard][warn] skip row parse error: {e!r} row={row}")
        return None

    return Trade(
        ts=ts,
        ticket=ticket,
        symbol=symbol,
        side=side,
        volume=volume,
        entry=entry,
        close=close,
        realized_quote=realized_quote,
        realized_ccy=realized_ccy,
        reason=reason,
    )


def load_trades(csv_path: pathlib.Path, since_days: Optional[float] = None) -> List[Trade]:
    if not csv_path.exists():
        print(f"[ai_leaderboard][err] CSV not found: {csv_path}")
        return []

    trades: List[Trade] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = _parse_row(row)
            if t is None:
                continue
            trades.append(t)

    if not trades:
        return trades

    if since_days is not None and since_days > 0:
        # Filter trades by ts >= now - since_days
        now_ts = dt.datetime.now().timestamp()
        cutoff = now_ts - since_days * 86400.0
        trades = [t for t in trades if t.ts >= cutoff]

    return trades


def compute_stats(trades: List[Trade]) -> Stats:
    if not trades:
        return Stats()

    pnls = [t.realized_ccy for t in trades]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    pnl_total = sum(pnls)
    pnl_avg = pnl_total / n if n else 0.0
    win_count = len(wins)
    loss_count = len(losses)
    winrate_pct = (win_count / n * 100.0) if n else 0.0
    avg_win = sum(wins) / win_count if win_count else 0.0
    avg_loss = sum(losses) / loss_count if loss_count else 0.0
    best = max(pnls)
    worst = min(pnls)

    return Stats(
        n=n,
        wins=win_count,
        losses=loss_count,
        pnl_total=pnl_total,
        pnl_avg=pnl_avg,
        winrate_pct=winrate_pct,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best=best,
        worst=worst,
    )


def compute_equity_curve(trades: List[Trade]) -> Tuple[float, float]:
    """
    Return (total_pnl, max_drawdown) using realized_ccy sorted by ts.
    max_drawdown is returned as a positive number.
    """
    if not trades:
        return 0.0, 0.0

    trades_sorted = sorted(trades, key=lambda t: t.ts)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0  # as positive

    for t in trades_sorted:
        equity += t.realized_ccy
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    total_pnl = equity
    return total_pnl, max_dd


def group_by_symbol(trades: List[Trade]) -> Dict[str, List[Trade]]:
    out: Dict[str, List[Trade]] = {}
    for t in trades:
        out.setdefault(t.symbol, []).append(t)
    return out


def group_by_symbol_side(trades: List[Trade]) -> Dict[Tuple[str, str], List[Trade]]:
    out: Dict[Tuple[str, str], List[Trade]] = {}
    for t in trades:
        key = (t.symbol, t.side)
        out.setdefault(key, []).append(t)
    return out


def group_by_side(trades: List[Trade]) -> Dict[str, List[Trade]]:
    out: Dict[str, List[Trade]] = {}
    for t in trades:
        out.setdefault(t.side, []).append(t)
    return out


def print_stats_table(title: str, stats_rows: List[Tuple[str, Optional[str], Stats]]):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    header = f"{'SYMBOL':<15} {'SIDE':<6} {'TRADES':>7} {'WIN%':>7} {'PNL':>12} {'AVG':>10} {'AVG_WIN':>10} {'AVG_LOSS':>10} {'BEST':>10} {'WORST':>10}"
    print(header)
    print("-" * len(header))

    for sym, side, s in stats_rows:
        side_str = side or "-"
        print(
            f"{sym:<15} {side_str:<6} "
            f"{s.n:7d} {s.winrate_pct:7.1f} "
            f"{s.pnl_total:12.2f} {s.pnl_avg:10.2f} "
            f"{s.avg_win:10.2f} {s.avg_loss:10.2f} "
            f"{s.best:10.2f} {s.worst:10.2f}"
        )


def write_csv(path: pathlib.Path, stats_rows: List[Tuple[str, Optional[str], Stats]], title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", title])
        writer.writerow([])
        writer.writerow(["symbol", "side", "trades", "wins", "losses", "winrate_pct", "pnl_total", "pnl_avg", "avg_win", "avg_loss", "best", "worst"])
        for sym, side, s in stats_rows:
            writer.writerow([
                sym,
                side or "",
                s.n,
                s.wins,
                s.losses,
                f"{s.winrate_pct:.2f}",
                f"{s.pnl_total:.2f}",
                f"{s.pnl_avg:.2f}",
                f"{s.avg_win:.2f}",
                f"{s.avg_loss:.2f}",
                f"{s.best:.2f}",
                f"{s.worst:.2f}",
            ])


def main():
    parser = argparse.ArgumentParser(description="AI Leaderboard: per-symbol and per-direction trade stats from TradeLogger CSV.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help=f"Path to trades.csv (default: {DEFAULT_CSV})")
    parser.add_argument("--since-days", type=float, default=None, help="Only include trades from the last N days (float).")
    parser.add_argument("--out-dir", type=str, default=str(REPORTS_DIR), help=f"Output directory for CSV reports (default: {REPORTS_DIR})")

    args = parser.parse_args()
    csv_path = pathlib.Path(args.csv)
    out_dir = pathlib.Path(args.out_dir)

    print(f"[ai_leaderboard] CSV: {csv_path}")
    trades = load_trades(csv_path, since_days=args.since_days)
    print(f"[ai_leaderboard] Loaded trades: {len(trades)}")

    if not trades:
        print("[ai_leaderboard] No trades found. Nothing to do.")
        return

    total_pnl, max_dd = compute_equity_curve(trades)
    print()
    print("OVERALL:")
    print(f"  Total PnL (realized_ccy): {total_pnl:.2f}")
    print(f"  Max drawdown (realized_ccy): {max_dd:.2f}")

    # By symbol
    g_sym = group_by_symbol(trades)
    sym_rows: List[Tuple[str, Optional[str], Stats]] = []
    for sym, ts in sorted(g_sym.items()):
        s = compute_stats(ts)
        sym_rows.append((sym, None, s))
    print_stats_table("BY SYMBOL", sym_rows)
    write_csv(out_dir / "ai_by_symbol.csv", sym_rows, "By Symbol")

    # By symbol + side
    g_sym_side = group_by_symbol_side(trades)
    sym_side_rows: List[Tuple[str, Optional[str], Stats]] = []
    for (sym, side), ts in sorted(g_sym_side.items()):
        s = compute_stats(ts)
        sym_side_rows.append((sym, side, s))
    print_stats_table("BY SYMBOL + SIDE", sym_side_rows)
    write_csv(out_dir / "ai_by_symbol_side.csv", sym_side_rows, "By Symbol + Side")

    # By side overall
    g_side = group_by_side(trades)
    side_rows: List[Tuple[str, Optional[str], Stats]] = []
    for side, ts in sorted(g_side.items()):
        s = compute_stats(ts)
        side_rows.append(("ALL", side, s))
    print_stats_table("BY SIDE (ALL SYMBOLS)", side_rows)
    write_csv(out_dir / "ai_by_side.csv", side_rows, "By Side (All Symbols)")


if __name__ == "__main__":
    main()