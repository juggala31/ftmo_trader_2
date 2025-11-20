from __future__ import annotations
import sys, time, json, pathlib, argparse, datetime as dt
from typing import List, Dict, Any, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

try:
    import yaml
except Exception:
    yaml = None

from settings import load_settings

TF_MAP = {
    "M5":  ("M5",  5,  getattr(mt5, "TIMEFRAME_M5",  None) if mt5 else None),
    "M15": ("M15", 15, getattr(mt5, "TIMEFRAME_M15", None) if mt5 else None),
    "M30": ("M30", 30, getattr(mt5, "TIMEFRAME_M30", None) if mt5 else None),
    "H1":  ("H1",  60, getattr(mt5, "TIMEFRAME_H1",  None) if mt5 else None),
    "H4":  ("H4",  240,getattr(mt5, "TIMEFRAME_H4",  None) if mt5 else None),
}

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def mt5_connect(s):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not installed. pip install MetaTrader5")
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    ok = mt5.login(login=int(s.mt5_login) if str(s.mt5_login).isdigit() else s.mt5_login,
                   password=s.mt5_password, server=s.mt5_server)
    if not ok:
        raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")

def ensure_visible(symbol: str):
    si = mt5.symbol_info(symbol)
    if si is None:
        cand = [x.name for x in (mt5.symbols_get() or []) if symbol.lower() in x.name.lower()]
        raise RuntimeError(f"Symbol '{symbol}' not found. Candidates: {cand[:20]}")
    if not si.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select({symbol}) failed")

def fetch_ohlc_range(symbol: str, tf_const: int, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    frames = []
    cur = start
    step = dt.timedelta(days=60)
    while cur < end:
        nxt = min(cur + step, end)
        rates = mt5.copy_rates_range(symbol, tf_const, cur, nxt)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)[["time","open","high","low","close","tick_volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            frames.append(df)
        cur = nxt
        time.sleep(0.03)
    if not frames:
        return pd.DataFrame(columns=["time","open","high","low","close","tick_volume"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out

def add_features(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df["ret_1"] = df["close"].pct_change()
    df["logret_1"] = np.log(df["close"]).diff()
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["rvol_20"] = df["logret_1"].rolling(20).std() * np.sqrt((60*24)/tf_minutes)
    df["rsi14"] = rsi(df["close"], 14)
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["hour"] = df["time"].dt.tz_convert("UTC").dt.hour
    df["dow"]  = df["time"].dt.tz_convert("UTC").dt.dayofweek
    for h in [1,3,6]:
        df[f"fwd_ret_{h}"] = df["close"].pct_change(periods=h).shift(-h)
        df[f"up_{h}"] = (df[f"fwd_ret_{h}"] > 0).astype("int8")
    return df

def run_export(symbols: List[str], timeframes: List[str], years: int = 5,
               start: str|None = None, end: str|None = None,
               out_dir: str|pathlib.Path = _ROOT / "ai" / "datasets",
               csv: bool = True, parquet: bool = True,
               log_cb=None) -> Dict[str, Any]:
    """
    Export OHLCV + features for the given symbols/timeframes/date range.
    Returns summary dict with counts.
    """
    if log_cb is None:
        log_cb = lambda m: None

    s = load_settings()
    mt5_connect(s)

    tfs = [tf.upper() for tf in timeframes]
    for tf in tfs:
        if tf not in TF_MAP or TF_MAP[tf][2] is None:
            raise RuntimeError(f"Unsupported timeframe: {tf}")

    tz_utc = dt.timezone.utc
    if end:
        end_dt = dt.datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=tz_utc)
    else:
        end_dt = dt.datetime.utcnow().replace(tzinfo=tz_utc)
    if start:
        start_dt = dt.datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=tz_utc)
    else:
        start_dt = end_dt - dt.timedelta(days=365*int(years))

    out_root = pathlib.Path(out_dir)
    out_csv_dir = out_root / "csv"
    out_parq_dir = out_root / "parquet"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_parq_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    counts = {"symbols": len(symbols), "frames": 0, "rows": 0}

    for sym in symbols:
        try:
            ensure_visible(sym)
        except Exception as e:
            log_cb(f"[SKIP] {sym}: {e}")
            continue
        for tf in tfs:
            code = TF_MAP[tf][2]
            tf_min = TF_MAP[tf][1]
            log_cb(f"[DL] {sym} {tf} {start_dt.date()} → {end_dt.date()}")
            df = fetch_ohlc_range(sym, code, start_dt, end_dt)
            if df.empty:
                log_cb(f"[WARN] no data for {sym} {tf}")
                continue
            feat = add_features(df, tf_min)
            feat.insert(0, "symbol", sym)
            feat.insert(1, "tf", tf)
            counts["frames"] += 1
            counts["rows"] += len(feat)

            if csv:
                p = out_csv_dir / tf / f"{sym}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                feat.to_csv(p, index=False)
            if parquet and HAVE_PARQUET:
                combined_rows.append(feat)

    if parquet and HAVE_PARQUET and combined_rows:
        big = pd.concat(combined_rows, ignore_index=True)
        outp = out_parq_dir / f"oanda_ai_{start_dt.date()}_{end_dt.date()}.parquet"
        big.to_parquet(outp, index=False)
        log_cb(f"[OK] Parquet: {outp} rows={len(big)}")

    return counts

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbols")
    ap.add_argument("--timeframes", default="M5,M15,M30,H1,H4")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--out-dir", default=str(_ROOT / "ai" / "datasets"))
    ap.add_argument("--csv", action="store_true", default=True)
    ap.add_argument("--parquet", action="store_true", default=True)
    return ap.parse_args()

def main():
    args = _parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("Provide --symbols list.")
    tfs = [x.strip().upper() for x in args.timeframes.split(",") if x.strip()]
    res = run_export(symbols=symbols, timeframes=tfs, years=args.years,
                     start=(args.start or None), end=(args.end or None),
                     out_dir=args.out_dir, csv=args.csv, parquet=args.parquet,
                     log_cb=lambda m: print(m, flush=True))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()