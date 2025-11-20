import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


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

    def pf(self) -> float:
        if self.gross_loss >= -1e-9:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)

    def win_rate(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades > 0 else 0.0


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def load_trades_stats(
    path: Path,
    since_days: float,
) -> Dict[str, SymbolStats]:
    """
    Load trades CSV and compute per-symbol stats over the last since_days.
    CSV schema expected: ts, ticket, symbol, side, volume, entry, close,
                         realized_quote, realized_ccy, reason
    """
    stats: Dict[str, SymbolStats] = {}
    if not path.exists():
        print(f"[auto_tuner] trades CSV not found: {path}")
        return stats

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts: Optional[float] = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    rows: Dict[str, List[Tuple[float, float]]] = {}

    total_rows = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if cutoff_ts is not None and ts_val is not None and ts_val < cutoff_ts:
                continue

            pnl_str = (row.get("realized_ccy") or "").strip()
            if not pnl_str:
                pnl_str = (row.get("realized_quote") or "").strip()
            pnl = parse_float(pnl_str) or 0.0

            if sym not in rows:
                rows[sym] = []
            rows[sym].append((ts_val or now_ts, pnl))

    print(f"[auto_tuner] Loaded trades (after since_days filter): {sum(len(v) for v in rows.values())} from {path}")
    if total_rows > 0 and not rows:
        print("[auto_tuner] Note: trades exist in file, but none in the requested window.")

    # Compute stats per symbol with DD
    for sym, lst in rows.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        s = SymbolStats(symbol=sym)
        eq = 0.0
        peak = 0.0
        for ts, pnl in lst_sorted:
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

            # equity/DD
            eq += pnl
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > s.max_dd:
                s.max_dd = dd

            if s.first_ts is None or ts < s.first_ts:
                s.first_ts = ts
            if s.last_ts is None or ts > s.last_ts:
                s.last_ts = ts

        stats[sym] = s

    return stats


def classify_bucket(s: SymbolStats, min_trades: int) -> str:
    if s.trades < min_trades:
        return "insufficient"
    if s.pnl <= 0:
        return "weak"

    pf = s.pf()
    wr = s.win_rate()

    # Simple rules:
    # - strong: good PF and WR
    # - weak: PF < 1.0 or PnL <= 0
    # - neutral: in between
    if pf >= 1.5 and wr >= 50.0:
        return "strong"
    if pf < 1.0:
        return "weak"
    return "neutral"


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def ensure_state_dict(state: Dict) -> Dict:
    if not isinstance(state, dict):
        state = {}
    state.setdefault("weak_streak", {})
    state.setdefault("bucket", {})
    state.setdefault("status", {})
    return state


def sync_aliases(profile: Dict) -> None:
    """
    Copy .sim configs to live symbols so run_live.py sees updated risk/mode.
    """
    alias_map = {
        "US30Z25.sim": "US30USD",
        "US100Z25.sim": "NAS100USD",
        "US500Z25.sim": "SPX500USD",
        "XAUZ25.sim": "XAUUSD",
        "USOILZ25.sim": "WTICOUSD",
        "BTCX25.sim": "BTCUSD",
    }

    for src, dst in alias_map.items():
        node = profile.get(src)
        if not isinstance(node, dict):
            continue
        # Deep-ish copy to avoid shared references
        profile[dst] = json.loads(json.dumps(node))
        print(f"[auto_tuner] sync_aliases: {src} -> {dst}")


def tune_for_symbol(
    sym: str,
    s: SymbolStats,
    bucket: str,
    profile: Dict,
    state: Dict,
    mode: str,
    min_risk: float,
    max_risk: float,
    risk_step_up: float,
    risk_step_down: float,
) -> None:
    weak_map = state["weak_streak"]
    bucket_map = state["bucket"]
    status_map = state["status"]

    prev_weak = int(weak_map.get(sym, 0))
    prev_status = str(status_map.get(sym, "live"))
    weak_streak = prev_weak

    pf = s.pf()
    wr = s.win_rate()

    # Log basic stats
    print(
        f"[auto_tuner] {sym}: bucket={bucket} trades={s.trades} "
        f"pnl={s.pnl:.2f} wr={wr:.1f}% pf={pf:.2f} max_dd={s.max_dd:.2f} weak_streak={prev_weak} status={prev_status}"
    )

    bucket_map[sym] = bucket

    node = profile.get(sym)
    if not isinstance(node, dict):
        print(f"[auto_tuner] {sym}: not in ai_profile.json, skipping.")
        return

    enabled = bool(node.get("enabled", False))
    trade_mode = str(node.get("trade_mode", "live"))

    risk = node.setdefault("risk", {})
    if not isinstance(risk, dict):
        risk = {}
        node["risk"] = risk

    atr_risk_pct = risk.get("atr_risk_pct")
    if not isinstance(atr_risk_pct, (int, float)):
        atr_risk_pct = min_risk
    old_risk = atr_risk_pct

    tfs = node.setdefault("timeframes", {})
    if not isinstance(tfs, dict):
        tfs = {}
        node["timeframes"] = tfs

    tf_h1 = tfs.setdefault("H1", {})
    if not isinstance(tf_h1, dict):
        tf_h1 = {}
        tfs["H1"] = tf_h1

    tf_h1_enabled = bool(tf_h1.get("enabled", True))
    long_th = tf_h1.get("long_threshold", 0.60)
    short_th = tf_h1.get("short_threshold", 0.40)

    # --- Decision logic based on bucket + mode ---

    new_risk = old_risk
    new_enabled = enabled
    new_trade_mode = trade_mode
    new_long_th = long_th
    new_short_th = short_th
    status = prev_status

    if bucket == "insufficient":
        # Not enough data to make a strong judgment – leave mostly unchanged.
        status = trade_mode

    elif bucket == "strong":
        weak_streak = 0
        status = "live"

        # In eval mode: allow risk up to max_risk (1%) but not higher.
        # In live mode: allow small step-ups up to max_risk (2%).
        new_risk = min(max_risk, old_risk + risk_step_up)

        # Make sure symbol is live & enabled when it's strong
        new_trade_mode = "live"
        new_enabled = True

        # Ease thresholds slightly (more trades) but keep in sane band
        new_long_th = max(0.50, min(0.70, long_th - 0.02))
        new_short_th = min(0.50, max(0.30, short_th + 0.02))

    elif bucket == "weak":
        weak_streak = prev_weak + 1

        # Tighten risk
        new_risk = max(min_risk, old_risk - risk_step_down)

        # Tighten thresholds (fewer trades)
        new_long_th = min(0.75, long_th + 0.02)
        new_short_th = max(0.25, short_th - 0.02)

        if mode == "eval":
            # In eval mode: anything weak goes to paper immediately.
            print(f"[auto_tuner] {sym}: mode=eval and bucket=weak -> forcing PAPER/observe.")
            new_trade_mode = "paper"
            new_enabled = False
            status = "observe"
        else:
            # In live mode: only after a streak of weakness
            if weak_streak >= 3:
                print(f"[auto_tuner] {sym}: weak streak {weak_streak} -> disabling live trading and entering PAPER/observe mode.")
                new_trade_mode = "paper"
                new_enabled = False
                status = "observe"
            else:
                status = trade_mode

    elif bucket == "neutral":
        weak_streak = 0
        status = "live"
        # Keep risk/thresholds as-is (or gently nudge towards mid if you want later)

    # Clamp risk to [min_risk, max_risk]
    new_risk = max(min_risk, min(max_risk, new_risk))

    # Write back
    risk["mode"] = "ATR_RISK"
    risk["atr_risk_pct"] = new_risk
    node["enabled"] = new_enabled
    node["trade_mode"] = new_trade_mode
    tf_h1["enabled"] = tf_h1_enabled
    tf_h1["long_threshold"] = float(new_long_th)
    tf_h1["short_threshold"] = float(new_short_th)

    weak_map[sym] = weak_streak
    status_map[sym] = status

    print(
        f"[auto_tuner] {sym}: final enabled={new_enabled} "
        f"risk={new_risk*100:.2f}% trade_mode={new_trade_mode} status={status} "
        f"long_th={new_long_th:.2f} short_th={new_short_th:.2f} weak_streak={weak_streak}"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="AI auto-tuner with eval/live mode awareness.")
    ap.add_argument(
        "--trades-csv",
        required=True,
        help="Path to trades CSV (live or backtest).",
    )
    ap.add_argument(
        "--profile",
        required=True,
        help="Path to ai_profile.json.",
    )
    ap.add_argument(
        "--state",
        required=True,
        help="Path to auto_tuner_state.json.",
    )
    ap.add_argument(
        "--since-days",
        type=float,
        default=7.0,
        help="Lookback window in days (default: 7).",
    )
    ap.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum trades per symbol to make a decision (default: 10).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute decisions but do not write profile or state.",
    )

    args = ap.parse_args(argv)

    trades_path = Path(args.trades_csv)
    profile_path = Path(args.profile)
    state_path = Path(args.state)

    print(f"[auto_tuner] trades CSV: {trades_path}")
    print(f"[auto_tuner] profile:    {profile_path}")
    print(f"[auto_tuner] state:      {state_path}")
    print(f"[auto_tuner] since_days: {args.since_days}  min_trades: {args.min_trades}  dry_run={args.dry_run}")

    stats = load_trades_stats(trades_path, args.since_days)
    if not stats:
        print("[auto_tuner] No trades in the window. Nothing to do.")
        return

    profile = load_json(profile_path)
    state = ensure_state_dict(load_json(state_path))

    # Determine global mode and associated risk bands
    mode = str(profile.get("_mode", "eval")).lower()
    if mode not in ("eval", "live"):
        mode = "eval"

    if mode == "eval":
        min_risk = 0.005   # 0.5%
        max_risk = 0.01    # 1.0%
        risk_step_up = 0.001   # 0.1%
        risk_step_down = 0.001 # 0.1%
    else:  # live
        min_risk = 0.005   # 0.5%
        max_risk = 0.02    # 2.0%
        risk_step_up = 0.002   # 0.2%
        risk_step_down = 0.002 # 0.2%

    print(f"[auto_tuner] profile mode={mode}  risk_band=[{min_risk*100:.2f}%, {max_risk*100:.2f}%]")

    # Decide per symbol
    for sym, s in stats.items():
        bucket = classify_bucket(s, args.min_trades)
        tune_for_symbol(
            sym=sym,
            s=s,
            bucket=bucket,
            profile=profile,
            state=state,
            mode=mode,
            min_risk=min_risk,
            max_risk=max_risk,
            risk_step_up=risk_step_up,
            risk_step_down=risk_step_down,
        )

    # Sync .sim configs to live symbols
    sync_aliases(profile)

    if args.dry_run:
        print("[auto_tuner] DRY-RUN enabled: not writing profile or state.")
        return

    save_json(profile_path, profile)
    save_json(state_path, state)
    print("[auto_tuner] Saved updated profile and state.")


if __name__ == "__main__":
    main()