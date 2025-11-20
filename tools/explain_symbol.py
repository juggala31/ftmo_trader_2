import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
DATA_DIR = ROOT / "data"

PROFILE_PATH = AI_DIR / "ai_profile.json"
STATE_PATH = AI_DIR / "auto_tuner_state.json"
TRADES_CSV = DATA_DIR / "trades.csv"
AI_SIGNALS = AI_DIR / "ai_signals.csv"


# ===============================
# Helpers
# ===============================

def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def fmt_ts_date(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


def fmt_ts_full(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


# ===============================
# Profile / tuner state section
# ===============================

def load_profile() -> Dict:
    if not PROFILE_PATH.exists():
        print(f"[explain] profile not found: {PROFILE_PATH}")
        return {}
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state() -> Dict:
    if not STATE_PATH.exists():
        print(f"[explain] tuner state not found: {STATE_PATH}")
        return {}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_symbol_profile(profile: Dict, symbol: str) -> Dict:
    """
    Try a few common layouts:

    - profile[symbol]
    - profile['symbols'][symbol]
    - profile['symbol_cfg'][symbol]
    """
    if symbol in profile and isinstance(profile[symbol], dict):
        return profile[symbol]

    for key in ("symbols", "symbol_cfg"):
        block = profile.get(key)
        if isinstance(block, dict) and symbol in block and isinstance(block[symbol], dict):
            return block[symbol]

    return {}


def get_symbol_state(state: Dict, symbol: str) -> Dict:
    """
    Try common layouts:

    - state['symbols'][symbol]
    - state[symbol]
    """
    # Newer style
    block = state.get("symbols")
    if isinstance(block, dict) and symbol in block and isinstance(block[symbol], dict):
        return block[symbol]

    # Fallback
    if symbol in state and isinstance(state[symbol], dict):
        return state[symbol]

    return {}


def print_profile_section(symbol: str, tf: str = "H1") -> None:
    profile = load_profile()
    sym_cfg = get_symbol_profile(profile, symbol)

    print("Profile / AI settings")
    print("---------------------")

    if not sym_cfg:
        print("  (no profile entry found for this symbol)")
        print("")
        return

    enabled = sym_cfg.get("enabled", True)
    trade_mode = sym_cfg.get("trade_mode", "live")
    sizing = sym_cfg.get("sizing", sym_cfg.get("size_mode", "ATR_RISK"))
    atr_risk_pct = sym_cfg.get("atr_risk_pct", sym_cfg.get("risk_pct", 0.0))

    print(f"  enabled:     {enabled}")
    print(f"  trade_mode:  {trade_mode}")
    print(f"  sizing:      {sizing}")
    print(f"  atr_risk_pct:{atr_risk_pct*100:.2f}% per trade")

    tf_cfg = sym_cfg.get(tf, {})
    if tf_cfg:
        h1_enabled = tf_cfg.get("enabled", True)
        long_th = tf_cfg.get("long_threshold", tf_cfg.get("long_th", 0.0))
        short_th = tf_cfg.get("short_threshold", tf_cfg.get("short_th", 0.0))
        print(f"  {tf}.enabled:  {h1_enabled}")
        print(f"  {tf}.thresh:   long={long_th}  short={short_th}")
    else:
        print(f"  {tf}.enabled:  (no explicit {tf} block)")
    print("")


def print_state_section(symbol: str) -> None:
    state = load_state()
    sym_state = get_symbol_state(state, symbol)

    print("Tuner state")
    print("-----------")

    if not sym_state:
        print("  (no tuner state entry found for this symbol)")
        print("")
        return

    bucket = sym_state.get("bucket", "?")
    status = sym_state.get("status", sym_state.get("mode", "?"))
    weak_streak = sym_state.get("weak_streak", 0)

    print(f"  bucket:      {bucket}")
    print(f"  status:      {status}")
    print(f"  weak_streak: {weak_streak}")
    print("")


# ===============================
# Trades performance section
# ===============================

@dataclass
class TradeStats:
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    flats: int = 0
    pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0  # negative
    max_dd: float = 0.0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def win_rate(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades > 0 else 0.0

    def profit_factor(self) -> float:
        if self.gross_loss >= -1e-9:
            return float("inf") if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)


def compute_trade_stats(symbol: str, since_days: float) -> TradeStats:
    stats = TradeStats(symbol=symbol)

    if not TRADES_CSV.exists():
        print(f"[explain] trades CSV not found: {TRADES_CSV}")
        return stats

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts: Optional[float] = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    total_rows = 0
    used_rows = 0

    with TRADES_CSV.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        eq = 0.0
        peak = 0.0
        for row in r:
            total_rows += 1
            sym = (row.get("symbol") or "").strip()
            if sym != symbol:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if cutoff_ts is not None and ts_val is not None and ts_val < cutoff_ts:
                continue

            pnl_str = (row.get("realized_ccy") or "").strip()
            if not pnl_str:
                pnl_str = (row.get("realized_quote") or "").strip()
            pnl = parse_float(pnl_str)
            if pnl is None:
                pnl = 0.0

            used_rows += 1
            stats.trades += 1
            if pnl > 0:
                stats.wins += 1
                stats.gross_profit += pnl
            elif pnl < 0:
                stats.losses += 1
                stats.gross_loss += pnl
            else:
                stats.flats += 1
            stats.pnl += pnl

            eq += pnl
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > stats.max_dd:
                stats.max_dd = dd

            if ts_val is not None:
                if stats.first_ts is None or ts_val < stats.first_ts:
                    stats.first_ts = ts_val
                if stats.last_ts is None or ts_val > stats.last_ts:
                    stats.last_ts = ts_val

    print(f"[explain] trades rows total: {total_rows}")
    print(f"[explain] trades rows used : {used_rows}")
    return stats


def print_trades_section(symbol: str, since_days: float) -> None:
    stats = compute_trade_stats(symbol, since_days)

    print(f"Recent performance (trades.csv, symbol={symbol}, since_days={since_days})")
    print("---------------------------------------------------------------------")
    if stats.trades == 0:
        print("  (no trades found in this window)")
        print("")
        return

    print(f"  trades:      {stats.trades}")
    print(
        f"  wins/losses: {stats.wins}/{stats.losses} "
        f"(flats={stats.flats})"
    )
    print(f"  win_rate:    {stats.win_rate():.1f}%")
    print(f"  pnl:         {stats.pnl:.2f}")
    print(f"  avg_pnl:     {stats.pnl / stats.trades:.2f}")
    print(f"  max_dd:      {stats.max_dd:.2f}")
    print(f"  first_ts:    {fmt_ts_date(stats.first_ts)}")
    print(f"  last_ts:     {fmt_ts_date(stats.last_ts)}")
    print("")


# ===============================
# Regime analysis (from ai_signals)
# ===============================

@dataclass
class RegimeStats:
    symbol: str
    tf: str
    n: int = 0
    n_price_ema: int = 0
    above_ema: int = 0
    prices: List[float] = None
    rsis: List[float] = None
    rets: List[float] = None
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def __post_init__(self):
        if self.prices is None:
            self.prices = []
        if self.rsis is None:
            self.rsis = []
        if self.rets is None:
            self.rets = []

    @property
    def ratio_above_ema(self) -> float:
        if self.n_price_ema == 0:
            return 0.0
        return self.above_ema / self.n_price_ema

    @property
    def avg_abs_ret(self) -> float:
        if not self.rets:
            return 0.0
        return sum(abs(r) for r in self.rets) / len(self.rets)

    @property
    def std_ret(self) -> float:
        if not self.rets:
            return 0.0
        mu = sum(self.rets) / len(self.rets)
        var = sum((r - mu) ** 2 for r in self.rets) / max(len(self.rets) - 1, 1)
        return var ** 0.5

    @property
    def median_rsi(self) -> float:
        if not self.rsis:
            return 0.0
        s = sorted(self.rsis)
        m = len(s)
        mid = m // 2
        if m % 2 == 1:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])


def classify_trend(ratio_above: float) -> str:
    # >= 0.65 → trending up
    # <= 0.35 → trending down
    # else    → ranging
    if ratio_above >= 0.65:
        return "trending up"
    if ratio_above <= 0.35:
        return "trending down"
    return "ranging"


def classify_vol(avg_abs_ret: float) -> str:
    # avg_abs_ret is in fraction (0.001 = 0.1%)
    if avg_abs_ret < 0.0007:
        return "low"
    if avg_abs_ret > 0.0018:
        return "high"
    return "normal"


def compute_regime(symbol: str, tf: str, since_days: float) -> RegimeStats:
    rs = RegimeStats(symbol=symbol, tf=tf)

    if not AI_SIGNALS.exists():
        print(f"[explain] ai_signals.csv not found: {AI_SIGNALS}")
        return rs

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    total_rows = 0
    used_rows = 0

    with AI_SIGNALS.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        prev_price = None
        for row in r:
            total_rows += 1

            sym = (row.get("symbol") or "").strip()
            tf_row = (row.get("tf") or "").strip()
            if sym != symbol or tf_row != tf:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if cutoff_ts is not None and ts_val is not None and ts_val < cutoff_ts:
                continue

            price = parse_float((row.get("price") or "").strip())
            ema = parse_float((row.get("ema50") or "").strip())
            rsi = parse_float((row.get("rsi14") or "").strip())

            used_rows += 1
            rs.n += 1

            if price is not None:
                rs.prices.append(price)
                if prev_price is not None and prev_price != 0:
                    rs.rets.append((price - prev_price) / prev_price)
                prev_price = price

            if rsi is not None:
                rs.rsis.append(rsi)

            if price is not None and ema is not None:
                rs.n_price_ema += 1
                if price > ema:
                    rs.above_ema += 1

            if ts_val is not None:
                if rs.first_ts is None or ts_val < rs.first_ts:
                    rs.first_ts = ts_val
                if rs.last_ts is None or ts_val > rs.last_ts:
                    rs.last_ts = ts_val

    print(f"[explain] ai_signals rows total: {total_rows}")
    print(f"[explain] ai_signals rows used : {used_rows}")
    return rs


def print_regime_section(symbol: str, tf: str, since_days: float) -> None:
    rs = compute_regime(symbol, tf, since_days)

    print(f"Market regime (ai_signals.csv, symbol={symbol}, tf={tf}, since_days={since_days})")
    print("--------------------------------------------------------------------------------")
    if rs.n == 0:
        print("  (no signal rows found for this symbol/TF in this window)")
        print("")
        return

    ratio_above = rs.ratio_above_ema
    avg_abs_ret = rs.avg_abs_ret
    std_ret = rs.std_ret
    med_rsi = rs.median_rsi

    trend_label = classify_trend(ratio_above)
    vol_label = classify_vol(avg_abs_ret)

    print(f"  samples:     {rs.n}")
    print(f"  date_range:  {fmt_ts_full(rs.first_ts)} -> {fmt_ts_full(rs.last_ts)}")
    print("")
    print(f"  trend       : {trend_label}")
    print(f"    price>EMA : {ratio_above*100:.1f}% of samples")
    print("")
    print(f"  volatility  : {vol_label}")
    print(f"    avg |ret| : {avg_abs_ret*100:.3f}%")
    print(f"    std ret   : {std_ret*100:.3f}%")
    print("")
    print(f"  RSI         : median={med_rsi:.1f}")
    print(
        "    (High RSI + trending up → strong bullish regime; "
        "Low RSI + trending down → strong bearish regime.)"
    )
    print("")


# ===============================
# Main
# ===============================

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Explain AI/tuner settings and performance for a symbol.",
    )
    ap.add_argument(
        "--symbol",
        required=True,
        help="Symbol, e.g. US100Z25.sim or XAUZ25.sim",
    )
    ap.add_argument(
        "--since-days",
        type=float,
        default=365.0,
        help="Lookback window in days for trade performance (default: 365).",
    )
    ap.add_argument(
        "--tf",
        default="H1",
        help="Timeframe for regime analysis (default: H1).",
    )
    ap.add_argument(
        "--regime-since-days",
        type=float,
        default=30.0,
        help="Lookback window in days for regime analysis (default: 30).",
    )
    args = ap.parse_args(argv)

    symbol = args.symbol
    tf = args.tf

    print(f"=== Explain: {symbol} ===")
    print("")

    # 1) Profile / AI settings
    print_profile_section(symbol, tf=tf)

    # 2) Tuner state
    print_state_section(symbol)

    # 3) Recent performance from trades.csv
    print_trades_section(symbol, since_days=args.since_days)

    # 4) Market regime from ai_signals.csv
    print_regime_section(symbol, tf=tf, since_days=args.regime_since_days)


if __name__ == "__main__":
    main()