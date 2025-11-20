import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


AI_SIGNALS = Path("ai/ai_signals.csv")


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


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def classify_trend(ratio_above: float) -> str:
    # Very simple rule:
    #   >= 0.65 → trending up
    #   <= 0.35 → trending down
    #   else    → ranging
    if ratio_above >= 0.65:
        return "trending up"
    if ratio_above <= 0.35:
        return "trending down"
    return "ranging"


def classify_vol(avg_abs_ret: float) -> str:
    # avg_abs_ret is in fractional terms (e.g. 0.001 = 0.1%)
    # Rough heuristic thresholds for H1:
    #   < 0.0007  (~0.07%) → low
    #   0.0007–0.0018      → normal
    #   > 0.0018  (~0.18%) → high
    if avg_abs_ret < 0.0007:
        return "low"
    if avg_abs_ret > 0.0018:
        return "high"
    return "normal"


def load_regime(symbol: str, tf: str, since_days: float) -> RegimeStats:
    rs = RegimeStats(symbol=symbol, tf=tf)

    if not AI_SIGNALS.exists():
        print(f"[regime] ai_signals.csv not found at {AI_SIGNALS}")
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

    print(f"[regime] ai_signals rows total: {total_rows}")
    print(f"[regime] rows used for {symbol} {tf}: {used_rows}")
    return rs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Analyze market regime from ai_signals.csv (trend + volatility)."
    )
    ap.add_argument("--symbol", required=True, help="Symbol, e.g. US100Z25.sim")
    ap.add_argument("--tf", default="H1", help="Timeframe, e.g. H1 (default) or M30")
    ap.add_argument(
        "--since-days",
        type=float,
        default=30.0,
        help="Lookback window in days for signals (default: 30).",
    )
    args = ap.parse_args(argv)

    print(f"[regime] symbol:     {args.symbol}")
    print(f"[regime] tf:         {args.tf}")
    print(f"[regime] since_days: {args.since_days}")
    print(f"[regime] file:       {AI_SIGNALS}")

    stats = load_regime(args.symbol, args.tf, args.since_days)

    if stats.n == 0:
        print("\nNo signal rows found for this symbol/TF in the requested window.")
        return

    ratio_above = stats.ratio_above_ema
    avg_abs_ret = stats.avg_abs_ret
    std_ret = stats.std_ret
    med_rsi = stats.median_rsi

    trend_label = classify_trend(ratio_above)
    vol_label = classify_vol(avg_abs_ret)

    print("\n=== Regime analysis ===")
    print(f"Samples       : {stats.n}")
    print(f"Date range    : {fmt_ts(stats.first_ts)} -> {fmt_ts(stats.last_ts)}")
    print("")
    print(f"Trend regime  : {trend_label}")
    print(f"  - price>EMA : {ratio_above*100:.1f}% of samples")
    print("")
    print(f"Vol regime    : {vol_label}")
    print(f"  - avg |ret| : {avg_abs_ret*100:.3f}%")
    print(f"  - std ret   : {std_ret*100:.3f}%")
    print("")
    print(f"RSI snapshot  : median={med_rsi:.1f}")
    print(
        "  (High RSI + trending up → strong bullish regime; "
        "Low RSI + trending down → strong bearish regime.)"
    )


if __name__ == "__main__":
    main()