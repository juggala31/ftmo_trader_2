#!/usr/bin/env python
"""
build_ml_dataset.py

Builds an ML-ready dataset CSV for a given symbol×TF from a raw MT5 export.

Input CSV (export) is expected to look like:

    time,open,high,low,close,tick_volume,spread,real_volume

Output CSV (dataset) will live under:

    ai/datasets/csv/<TF>/<symbol>.csv

and will contain at least the columns required by ai/train_xgb.py:

    symbol,tf,time,open,high,low,close,volume,
    ret_1,logret_1,atr14,rvol_20,rsi14,ma20,ma50,hour,dow,
    fwd_ret_1,up_1,fwd_ret_3,up_3,fwd_ret_6,up_6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "ai" / "datasets" / "csv"


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)

    # Returns
    df["ret_1"] = close.pct_change()
    df["logret_1"] = np.log(close / close.shift(1))

    # ATR
    df["atr14"] = compute_atr(df, period=atr_period)

    # Realized vol
    df["rvol_20"] = df["logret_1"].rolling(20, min_periods=20).std()

    # RSI
    df["rsi14"] = compute_rsi(close, period=14)

    # Moving averages
    df["ma20"] = close.rolling(20, min_periods=20).mean()
    df["ma50"] = close.rolling(50, min_periods=50).mean()

    # Time features
    df["hour"] = df["time"].dt.hour.astype(float)
    df["dow"] = df["time"].dt.dayofweek.astype(float)

    return df


def build_dataset(symbol: str, tf: str, export_csv: Path, out_path: Path, atr_period: int = 14) -> Path:
    if not export_csv.is_file():
        raise SystemExit(f"[build_ml_dataset] Export CSV not found: {export_csv}")

    df = pd.read_csv(export_csv)

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    required = ["time", "open", "high", "low", "close"]
    for r in required:
        if r not in cols:
            raise SystemExit(
                f"[build_ml_dataset] Export CSV missing '{r}' column. "
                f"Found columns: {list(df.columns)}"
            )

    df = df.rename(
        columns={
            cols["time"]: "time",
            cols["open"]: "open",
            cols["high"]: "high",
            cols["low"]: "low",
            cols["close"]: "close",
        }
    )

    # Volume: prefer tick_volume if present, else real_volume, else 0
    vol_col: Optional[str] = None
    for cand in ("tick_volume", "Tick_volume", "real_volume", "Real_volume"):
        if cand in df.columns:
            vol_col = cand
            break
    if vol_col is not None:
        df["volume"] = df[vol_col].astype(float)
    else:
        df["volume"] = 0.0

    # Parse time with UTC
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    if df.empty:
        raise SystemExit("[build_ml_dataset] No valid rows after parsing time.")

    # Add features
    df_feat = add_features(df, atr_period=atr_period)

    close = df_feat["close"].astype(float)

    # Forward returns and up flags
    horizons = [1, 3, 6]
    for h in horizons:
        col_ret = f"fwd_ret_{h}"
        col_up = f"up_{h}"
        df_feat[col_ret] = close.shift(-h) / close - 1.0
        df_feat[col_up] = (df_feat[col_ret] > 0.0).astype(int)

    # Attach symbol/tf columns at front
    df_feat.insert(0, "tf", tf)
    df_feat.insert(0, "symbol", symbol)

    # Drop rows where we don't have full forward labels (tail)
    drop_cols = [f"fwd_ret_{h}" for h in horizons]
    df_out = df_feat.dropna(subset=drop_cols).copy()

    # Final column ordering (must match what train_xgb expects)
    cols_out: List[str] = [
        "symbol",
        "tf",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1",
        "logret_1",
        "atr14",
        "rvol_20",
        "rsi14",
        "ma20",
        "ma50",
        "hour",
        "dow",
        "fwd_ret_1",
        "up_1",
        "fwd_ret_3",
        "up_3",
        "fwd_ret_6",
        "up_6",
    ]
    missing = [c for c in cols_out if c not in df_out.columns]
    if missing:
        raise SystemExit(f"[build_ml_dataset] Missing expected columns in output frame: {missing}")

    df_out = df_out.loc[:, cols_out]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[build_ml_dataset] Wrote dataset -> {out_path}")
    print(f"[build_ml_dataset] Rows: {len(df_out)}")
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build ML dataset for ai/train_xgb.py from MT5 export CSV.")
    ap.add_argument("--symbol", required=True, help="Symbol name, e.g. XAUZ25.sim")
    ap.add_argument("--tf", required=True, help="Timeframe label, e.g. H1")
    ap.add_argument(
        "--export-csv",
        required=True,
        help="Path to raw export CSV (time,open,high,low,close,tick_volume,spread,real_volume)",
    )
    ap.add_argument(
        "--out",
        help="Output dataset CSV path. Default: ai/datasets/csv/<TF>/<symbol>.csv",
    )
    args = ap.parse_args(argv)

    symbol = args.symbol
    tf = args.tf
    export_csv = Path(args.export_csv)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = DATASET_DIR / tf / out_path
    else:
        out_path = DATASET_DIR / tf / f"{symbol}.csv"

    build_dataset(symbol, tf, export_csv, out_path)


if __name__ == "__main__":
    main()