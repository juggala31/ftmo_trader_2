"""
XGBoost training script for Trader 2.0.

- Trains one binary classifier per (symbol, timeframe) using ai/datasets/csv.
- Uses engineered features like in backtests.run_bt.add_features.
- Saves models under ai/models/xgb_{symbol}_{tf}.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover - environment without xgboost
    print("[ERR] xgboost is not installed. Please run: pip install xgboost")
    print("Error:", e)
    sys.exit(1)

DATA_DIR = ROOT / "ai" / "datasets" / "csv"
MODEL_DIR = ROOT / "ai" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = MODEL_DIR / "xgb_meta.json"

FEATURE_COLS: List[str] = [
    "ret_1",
    "logret_1",
    "atr14",
    "rvol_20",
    "rsi14",
    "ma20",
    "ma50",
    "hour",
    "dow",
]
TARGET_COL = "up_3"

DEFAULT_SYMBOLS: List[str] = [
    "XAUZ25.sim",
    "US30Z25.sim",
    "US100Z25.sim",
    "US500Z25.sim",
    "USOILZ25.sim",
    "BTCX25.sim",
]
DEFAULT_TFS: List[str] = ["M30", "H1", "H4"]


def _load_symbol_tf_df(symbol: str, tf: str) -> pd.DataFrame:
    path = DATA_DIR / tf / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found for {symbol} {tf}: {path}")
    df = pd.read_csv(path)
    return df


def _prepare_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    work = df.loc[:, cols].dropna().copy()
    if work.empty:
        raise ValueError("No rows left after dropping NaNs")
    X = work[FEATURE_COLS].astype("float32").values
    y = work[TARGET_COL].astype("int32").values
    return X, y


def _train_one(symbol: str, tf: str) -> Dict[str, float]:
    df = _load_symbol_tf_df(symbol, tf)
    X, y = _prepare_xy(df)

    n = X.shape[0]
    if n < 500:
        print(f"[WARN] {symbol} {tf}: only {n} usable rows; model may be unstable", flush=True)

    split = max(int(n * 0.8), 1)
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if pos == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = max(1.0, neg / pos)

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
    )

    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    if X_valid.shape[0] > 0:
        prob_valid = clf.predict_proba(X_valid)[:, 1]
        pred_valid = (prob_valid >= 0.5).astype("int32")
        acc = float((pred_valid == y_valid).mean())
    else:
        acc = float("nan")

    path = MODEL_DIR / f"xgb_{symbol}_{tf}.json"
    clf.save_model(str(path))

    stats = {
        "symbol": symbol,
        "tf": tf,
        "rows": int(n),
        "train_rows": int(X_train.shape[0]),
        "valid_rows": int(X_valid.shape[0]),
        "pos_frac": float((y == 1).mean()),
        "valid_acc": acc,
    }
    print(f"[OK] trained {symbol} {tf}: rows={n} pos_frac={stats['pos_frac']:.3f} acc={acc:.3f}", flush=True)
    return stats


def _save_meta(meta: List[Dict[str, float]]) -> None:
    by_key = {(m["symbol"], m["tf"]): m for m in meta}
    payload = {
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "models": [
            {
                "symbol": s,
                "tf": tf,
                **m,
            }
            for (s, tf), m in sorted(by_key.items())
        ],
    }
    META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote meta to {META_PATH}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost models per symbol×TF.")
    parser.add_argument("--symbol", action="append", dest="symbols", help="Limit to these symbols (can repeat)")
    parser.add_argument("--tf", action="append", dest="tfs", help="Limit to these timeframes (M30,H1,H4).")
    args = parser.parse_args(argv)

    symbols = args.symbols or DEFAULT_SYMBOLS
    tfs = args.tfs or DEFAULT_TFS

    all_stats: List[Dict[str, float]] = []
    for sym in symbols:
        for tf in tfs:
            try:
                stats = _train_one(sym, tf)
                all_stats.append(stats)
            except FileNotFoundError as e:
                print(f"[SKIP] {sym} {tf}: {e}", flush=True)
            except Exception as e:
                print(f"[ERR] training failed for {sym} {tf}: {e}", flush=True)

    if all_stats:
        _save_meta(all_stats)
    else:
        print("[WARN] no models trained (no data?)")


if __name__ == "__main__":
    main()