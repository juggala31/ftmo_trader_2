import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
DATA_DIR = ROOT / "data"

PROFILE_PATH = AI_DIR / "ai_profile.json"
STATE_PATH = AI_DIR / "auto_tuner_state.json"
AI_SIGNALS = AI_DIR / "ai_signals.csv"
TRADES_CSV = DATA_DIR / "trades.csv"


# ===============================
# Helpers
# ===============================

def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def load_profile() -> Dict:
    if not PROFILE_PATH.exists():
        return {}
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state() -> Dict:
    if not STATE_PATH.exists():
        return {}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_profile_symbols(profile: Dict) -> List[str]:
    symbols: List[str] = []

    sym_block = profile.get("symbols")
    if isinstance(sym_block, dict):
        symbols.extend([k for k in sym_block.keys()])

    for k, v in profile.items():
        if k in ("global", "symbols"):
            continue
        if isinstance(v, dict):
            symbols.append(k)

    return sorted(sorted(set(symbols)))


def get_profile_cfg(profile: Dict, symbol: str) -> Dict:
    """
    IMPORTANT: Prefer top-level symbol config first (tuner writes here),
    then fall back to the nested 'symbols' block.
    """
    cfg = profile.get(symbol)
    if isinstance(cfg, dict):
        return cfg

    sym_block = profile.get("symbols")
    if isinstance(sym_block, dict):
        cfg = sym_block.get(symbol)
        if isinstance(cfg, dict):
            return cfg

    return {}


def normalize_risk_value(raw: float) -> float:
    """
    Normalize atr_risk_pct / risk_pct into fraction form:
    - If raw > 0.2 and <= 20 → treat as percent (1.0 = 1%) → raw/100.
    - Else → treat as fraction already (0.01 = 1%).
    """
    if not isinstance(raw, (int, float)):
        return 0.0
    if raw > 0.2 and raw <= 20.0:
        return raw / 100.0
    return raw


def merge_profile_and_state_for_symbol(
    profile: Dict,
    state: Dict,
    symbol: str,
    tf: str = "H1",
) -> Tuple[bool, str, float, str, int]:
    """
    Return:
      enabled_final (bool),
      trade_mode_final (str),
      risk_frac (float),
      status_or_bucket (str),
      weak_streak (int)
    """
    cfg = get_profile_cfg(profile, symbol)

    # --- Base enabled/mode from profile (symbol + TF block) ---
    enabled = bool(cfg.get("enabled", True))
    trade_mode = cfg.get("trade_mode") or "live"

    tf_cfg = cfg.get(tf, {})
    if isinstance(tf_cfg, dict):
        if "enabled" in tf_cfg:
            enabled = bool(tf_cfg.get("enabled"))
        if "trade_mode" in tf_cfg and tf_cfg.get("trade_mode"):
            trade_mode = tf_cfg.get("trade_mode")

    # Base risk from profile
    raw_risk = cfg.get("atr_risk_pct", cfg.get("risk_pct", 0.0))
    if not isinstance(raw_risk, (int, float)):
        raw_risk = 0.0

    if isinstance(tf_cfg, dict):
        tf_risk = tf_cfg.get("atr_risk_pct", tf_cfg.get("risk_pct", 0.0))
        if isinstance(tf_risk, (int, float)) and tf_risk != 0.0:
            raw_risk = tf_risk

    risk_frac = normalize_risk_value(raw_risk)

    # --- Overlay tuner state (auto_tuner_state.json) ---
    sym_state: Dict = {}

    # Preferred: nested "symbols" block
    if isinstance(state.get("symbols"), dict):
        sym_state = state["symbols"].get(symbol, {}) or {}

    # Fallback: top-level entry
    if not sym_state:
        maybe = state.get(symbol)
        if isinstance(maybe, dict):
            sym_state = maybe

    status = str(sym_state.get("status") or "").strip()
    bucket = str(sym_state.get("bucket") or "").strip()
    weak_streak = int(sym_state.get("weak_streak", 0))

    # If tuner persisted enabled/trade_mode, respect that
    if "enabled" in sym_state:
        enabled = bool(sym_state.get("enabled"))
    if "trade_mode" in sym_state and sym_state.get("trade_mode"):
        trade_mode = str(sym_state.get("trade_mode"))

    # If tuner stored a risk override, prefer it
    risk_state = sym_state.get("risk")
    if risk_state is None:
        risk_state = sym_state.get("atr_risk_pct")
    if risk_state is None:
        risk_state = sym_state.get("risk_frac")

    if isinstance(risk_state, (int, float)) and risk_state > 0:
        risk_frac = normalize_risk_value(risk_state)

    # Map status to final enabled/mode (status wins)
    if status == "observe":
        enabled_final = False
        mode_final = "paper"
    elif status == "off":
        enabled_final = False
        mode_final = "off"
    elif status == "live":
        enabled_final = True
        mode_final = trade_mode or "live"
    else:
        enabled_final = enabled
        mode_final = trade_mode

    status_or_bucket = status or bucket or ""

    return enabled_final, mode_final, risk_frac, status_or_bucket, weak_streak


# ===============================
# Trade stats
# ===============================

@dataclass
class TradeStats:
    symbol: str
    n: int = 0
    wins: int = 0
    loss: int = 0
    pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0  # positive number
    max_dd: float = 0.0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None

    def update(self, ts: float, pnl_row: float):
        self.n += 1
        self.pnl += pnl_row
        if pnl_row > 0:
            self.wins += 1
            self.gross_profit += pnl_row
        elif pnl_row < 0:
            self.loss += 1
            self.gross_loss += -pnl_row

        if self.first_ts is None or ts < self.first_ts:
            self.first_ts = ts
        if self.last_ts is None or ts > self.last_ts:
            self.last_ts = ts

        eq = self.pnl
        if eq < self.max_dd:
            self.max_dd = eq  # more negative = deeper drawdown

    @property
    def win_rate(self) -> float:
        if self.n == 0:
            return 0.0
        return 100.0 * self.wins / self.n

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            if self.gross_profit > 0:
                return float("inf")
            return 0.0
        return self.gross_profit / self.gross_loss


def compute_trade_stats(trades_path: Path, since_days: float) -> Tuple[Dict[str, TradeStats], int, int]:
    if not trades_path.exists():
        return {}, 0, 0

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts: Optional[float] = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    stats: Dict[str, TradeStats] = {}
    total_rows = 0
    used_rows = 0

    with trades_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if ts_val is None:
                continue
            if cutoff_ts is not None and ts_val < cutoff_ts:
                continue

            pnl_val = parse_float((row.get("realized_ccy") or row.get("realized_quote") or "0").strip())
            if pnl_val is None:
                pnl_val = 0.0

            if sym not in stats:
                stats[sym] = TradeStats(symbol=sym)
            stats[sym].update(ts_val, pnl_val)
            used_rows += 1

    return stats, total_rows, used_rows


# ===============================
# Regime (ai_signals) stats
# ===============================

@dataclass
class RegimeStats:
    symbol: str
    tf: str
    n: int = 0
    n_price_ema: int = 0
    above_ema: int = 0
    rets: List[float] = None

    def __post_init__(self):
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


def classify_trend(ratio_above: float) -> str:
    if ratio_above >= 0.65:
        return "UP"
    if ratio_above <= 0.35:
        return "DOWN"
    return "RANGE"


def classify_vol(avg_abs_ret: float) -> str:
    if avg_abs_ret < 0.0007:
        return "LOW"
    if avg_abs_ret > 0.0018:
        return "HIGH"
    return "NORM"


def compute_regime(tf: str, since_days: float) -> Tuple[Dict[str, Tuple[str, str]], int, int]:
    out: Dict[str, Tuple[str, str]] = {}
    if not AI_SIGNALS.exists():
        return out, 0, 0

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts: Optional[float] = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    total_rows = 0
    used_rows = 0
    stats_map: Dict[str, RegimeStats] = {}
    prev_price: Dict[str, float] = {}

    with AI_SIGNALS.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            total_rows += 1
            sym = (row.get("symbol") or "").strip()
            tf_row = (row.get("tf") or "").strip()
            if not sym or tf_row != tf:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if ts_val is not None and cutoff_ts is not None and ts_val < cutoff_ts:
                continue

            price = parse_float((row.get("price") or "").strip())
            ema50 = parse_float((row.get("ema50") or "").strip())

            used_rows += 1

            if sym not in stats_map:
                stats_map[sym] = RegimeStats(symbol=sym, tf=tf)
                prev_price[sym] = None  # type: ignore

            s = stats_map[sym]

            if price is not None:
                prev = prev_price.get(sym)
                if prev is not None and prev != 0:
                    s.rets.append((price - prev) / prev)
                prev_price[sym] = price

            if price is not None and ema50 is not None:
                s.n_price_ema += 1
                if price > ema50:
                    s.above_ema += 1

            s.n += 1

    for sym, s in stats_map.items():
        if s.n == 0:
            continue
        trend = classify_trend(s.ratio_above_ema)
        vol = classify_vol(s.avg_abs_ret)
        out[sym] = (trend, vol)

    return out, total_rows, used_rows


# ===============================
# Main reporting
# ===============================

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Daily summary: per-symbol trades + AI regime + tuner-aware risk."
    )
    ap.add_argument(
        "--trades",
        default=str(TRADES_CSV),
        help="Trades CSV path (default: data/trades.csv).",
    )
    ap.add_argument(
        "--since-days",
        type=float,
        default=30.0,
        help="Trade lookback window in days (default: 30).",
    )
    ap.add_argument(
        "--regime-since-days",
        type=float,
        default=7.0,
        help="Regime lookback window in days (default: 7).",
    )
    ap.add_argument(
        "--tf",
        default="H1",
        help="Timeframe to use from ai_signals (default: H1).",
    )
    args = ap.parse_args(argv)

    trades_path = Path(args.trades)

    # Load profile + tuner state
    profile = load_profile()
    state = load_state()

    # Trade stats
    stats, total_rows, used_rows = compute_trade_stats(trades_path, args.since_days)
    print(f"[daily] trades rows total: {total_rows}")
    print(f"[daily] trades rows used : {used_rows}")

    # Regime stats
    regime_map, ai_total, ai_used = compute_regime(args.tf, args.regime_since_days)
    print(f"[daily] ai_signals rows total: {ai_total}")
    print(f"[daily] ai_signals rows used : {ai_used}")
    print()

    # Decide which symbols to show: union of profile + any symbols with trades
    profile_syms = set(iter_profile_symbols(profile))
    trade_syms = set(stats.keys())
    all_syms = sorted(profile_syms.union(trade_syms))

    print("DAILY SUMMARY")
    print("=============")
    print(f"(trades since {int(args.since_days)}d, regime from ai_signals {int(args.regime_since_days)}d, tf={args.tf})")
    print()
    print("SYMBOL       EN  MODE   RISK%   TRD   WR%    PNL       PF    TREND  VOL")
    print("-----------------------------------------------------------------------")

    for sym in all_syms:
        ts = stats.get(sym)
        trd = ts.n if ts else 0
        wr = ts.win_rate if ts else 0.0
        pnl = ts.pnl if ts else 0.0
        pf = ts.profit_factor if ts else 0.0

        enabled, mode, risk_frac, status_or_bucket, weak_streak = merge_profile_and_state_for_symbol(
            profile, state, sym, tf=args.tf
        )

        risk_pct = risk_frac * 100.0

        trend, vol = regime_map.get(sym, ("-", "-"))

        en_flag = "Y" if enabled else "N"
        mode_str = (mode or "").lower()
        if mode_str not in ("live", "paper", "off"):
            mode_str = "-"

        print(
            f"{sym:<11} {en_flag:1}  {mode_str:<5} {risk_pct:5.2f} "
            f"{trd:5d} {wr:5.1f} {pnl:9.2f} {pf:6.2f}  {trend:<5}  {vol}"
        )

    print()
    print("Columns:")
    print("  EN    = enabled (Y/N) after tuner state")
    print("  MODE  = trade_mode (live/paper/off) after tuner state")
    print("  RISK% = per-trade risk (approx, fraction*100)")
    print("  TRD   = number of trades in trades.csv over since_days window")
    print("  WR%   = win rate")
    print("  PNL   = total realized PnL in account currency")
    print("  PF    = profit factor")
    print("  TREND = UP/DOWN/RANGE (from ai_signals, ratio of price>EMA)")
    print("  VOL   = LOW/NORM/HIGH (from ai_signals, avg abs return)")


if __name__ == "__main__":
    main()