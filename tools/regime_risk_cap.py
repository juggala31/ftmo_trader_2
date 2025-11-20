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
BACKUP_DIR = AI_DIR / "profile_backups"
AI_SIGNALS = AI_DIR / "ai_signals.csv"


# ===============================
# Generic helpers
# ===============================

def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


# ===============================
# Profile helpers
# ===============================

def load_profile() -> Dict:
    if not PROFILE_PATH.exists():
        raise SystemExit(f"[regime_risk_cap] profile not found: {PROFILE_PATH}")
    with PROFILE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"ai_profile.json.bak_{ts}"
    backup_path.write_bytes(PROFILE_PATH.read_bytes())
    return backup_path


def iter_symbols(profile: Dict) -> List[str]:
    symbols: List[str] = []

    sym_block = profile.get("symbols")
    if isinstance(sym_block, dict):
        symbols.extend(sym_block.keys())

    for k, v in profile.items():
        if k in ("global", "symbols"):
            continue
        if isinstance(v, dict):
            symbols.append(k)

    return sorted(sorted(set(symbols)))


def get_sym_cfg(profile: Dict, symbol: str) -> Dict:
    # Prefer profile['symbols'][symbol]
    sym_block = profile.get("symbols")
    if isinstance(sym_block, dict) and isinstance(sym_block.get(symbol), dict):
        return sym_block[symbol]

    # Fallback: top-level symbol
    if isinstance(profile.get(symbol), dict):
        return profile[symbol]

    # If missing, create it under symbols
    if not isinstance(sym_block, dict):
        profile["symbols"] = {}
        sym_block = profile["symbols"]
    sym_block[symbol] = {}
    return sym_block[symbol]


# ===============================
# Regime analysis (ai_signals.csv)
# ===============================

@dataclass
class RegimeStats:
    symbol: str
    tf: str
    n: int = 0
    n_price_ema: int = 0
    above_ema: int = 0
    prices: List[float] = None
    rets: List[float] = None

    def __post_init__(self):
        if self.prices is None:
            self.prices = []
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
    # Match daily_summary thresholds:
    if ratio_above >= 0.65:
        return "UP"
    if ratio_above <= 0.35:
        return "DOWN"
    return "RANGE"


def classify_vol(avg_abs_ret: float) -> str:
    # avg_abs_ret is in fraction (0.001 = 0.1%)
    if avg_abs_ret < 0.0007:
        return "LOW"
    if avg_abs_ret > 0.0018:
        return "HIGH"
    return "NORM"


def compute_regime_for_all(tf: str, since_days: float) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}

    if not AI_SIGNALS.exists():
        print(f"[regime_risk_cap] ai_signals.csv not found: {AI_SIGNALS}")
        return out

    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff_ts: Optional[float] = None
    if since_days > 0:
        cutoff_ts = now_ts - since_days * 86400.0

    total_rows = 0
    used_rows = 0
    stats_map: Dict[str, RegimeStats] = {}

    with AI_SIGNALS.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        prev_price_map: Dict[str, float] = {}

        for row in r:
            total_rows += 1
            sym = (row.get("symbol") or "").strip()
            tf_row = (row.get("tf") or "").strip()
            if tf_row != tf or not sym:
                continue

            ts_val = parse_float((row.get("ts") or "").strip())
            if cutoff_ts is not None and ts_val is not None and ts_val < cutoff_ts:
                continue

            price = parse_float((row.get("price") or "").strip())
            ema = parse_float((row.get("ema50") or "").strip())

            used_rows += 1

            if sym not in stats_map:
                stats_map[sym] = RegimeStats(symbol=sym, tf=tf)
                prev_price_map[sym] = None  # type: ignore

            s = stats_map[sym]

            if price is not None:
                prev = prev_price_map[sym]
                if prev is not None and prev != 0:
                    s.rets.append((price - prev) / prev)
                prev_price_map[sym] = price
                s.prices.append(price)

            if price is not None and ema is not None:
                s.n_price_ema += 1
                if price > ema:
                    s.above_ema += 1

            s.n += 1

    print(f"[regime_risk_cap] ai_signals rows total: {total_rows}")
    print(f"[regime_risk_cap] ai_signals rows used : {used_rows}")

    for sym, s in stats_map.items():
        if s.n == 0:
            continue
        ratio_above = s.ratio_above_ema
        avg_abs_ret = s.avg_abs_ret
        trend = classify_trend(ratio_above)
        vol = classify_vol(avg_abs_ret)
        out[sym] = (trend, vol)

    return out


# ===============================
# Risk cap logic
# ===============================

def normalize_risk_value(raw: float) -> float:
    """
    Normalize atr_risk_pct into fraction form:
    - If raw > 0.2 and <= 20 → treat as percent (1.0 = 1%) → raw/100.
    - Else → treat as fraction already (0.01 = 1%).
    """
    if not isinstance(raw, (int, float)):
        return 0.0
    if raw > 0.2 and raw <= 20.0:
        return raw / 100.0
    return raw


def apply_regime_risk_cap(max_risk: float, tf: str, since_days: float, dry_run: bool = False) -> None:
    profile = load_profile()
    regime_map = compute_regime_for_all(tf=tf, since_days=since_days)
    symbols = iter_symbols(profile)

    if not symbols:
        print("[regime_risk_cap] No symbols found in profile.")
        return

    changes = 0

    if dry_run:
        print(f"[regime_risk_cap] DRY-RUN: will not write changes.")
    else:
        backup_path = save_backup()
        print(f"[regime_risk_cap] Backup written -> {backup_path}")

    for sym in symbols:
        sym_cfg = get_sym_cfg(profile, sym)

        enabled = sym_cfg.get("enabled", True)
        trade_mode = sym_cfg.get("trade_mode", "live")

        # Only care about enabled symbols; you can tighten this to trade_mode == "live" if you want.
        if not enabled:
            continue

        trend, vol = regime_map.get(sym, (None, None))
        if trend is None or vol is None:
            # No regime info → skip
            continue

        print(f"[regime_risk_cap] {sym}: trend={trend} vol={vol}")

        # Our rule: RANGE + HIGH → cap risk
        if not (trend == "RANGE" and vol == "HIGH"):
            continue

        # Get current risk, with fallback to per-TF block
        raw_risk = sym_cfg.get("atr_risk_pct", sym_cfg.get("risk_pct", 0.0))
        if not isinstance(raw_risk, (int, float)):
            raw_risk = 0.0

        if raw_risk == 0.0:
            tf_cfg = sym_cfg.get(tf, {})
            tf_risk = tf_cfg.get("atr_risk_pct", tf_cfg.get("risk_pct", 0.0))
            if isinstance(tf_risk, (int, float)) and tf_risk != 0.0:
                raw_risk = tf_risk

        cur_frac = normalize_risk_value(raw_risk)

        if cur_frac <= max_risk or cur_frac == 0.0:
            print(f"[regime_risk_cap] {sym}: risk {cur_frac:.4f} already <= cap {max_risk:.4f}, no change.")
            continue

        new_frac = max_risk
        print(
            f"[regime_risk_cap] {sym}: RANGE/HIGH regime → cap risk {cur_frac:.4f} -> {new_frac:.4f}"
        )

        # Apply to symbol-level and TF-level
        sym_cfg["atr_risk_pct"] = new_frac
        tf_cfg = sym_cfg.setdefault(tf, {})
        tf_cfg["atr_risk_pct"] = new_frac
        changes += 1

    if dry_run:
        print(f"[regime_risk_cap] DRY-RUN complete. Changes that WOULD be applied: {changes}")
    else:
        PROFILE_PATH.write_text(
            json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[regime_risk_cap] Updated profile -> {PROFILE_PATH}")
        print(f"[regime_risk_cap] Changes applied: {changes}")


# ===============================
# Main
# ===============================

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Cap symbol risk based on market regime (ai_signals)."
    )
    ap.add_argument(
        "--tf",
        default="H1",
        help="Timeframe to use from ai_signals (default: H1).",
    )
    ap.add_argument(
        "--since-days",
        type=float,
        default=7.0,
        help="Lookback window in days for regime classification (default: 7).",
    )
    ap.add_argument(
        "--max-risk",
        type=float,
        default=0.005,
        help="Max risk per trade as FRACTION (0.005 = 0.5%%).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write profile.json, just log what would change.",
    )
    args = ap.parse_args(argv)

    apply_regime_risk_cap(
        max_risk=args.max_risk,
        tf=args.tf,
        since_days=args.since_days,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()