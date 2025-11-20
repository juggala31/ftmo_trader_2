"""
XGBoost inference helper for Trader 2.0.

- Looks for per-symbol/TF models saved by ai/train_xgb.py.
- Exposes FEATURE_COLS (must match training) and predict_proba().
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Tuple

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = pathlib.Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
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

try:
    import xgboost as xgb  # type: ignore
    HAVE_XGB = True
    XGB_STATUS = "ok"
except Exception as e:  # pragma: no cover - env without xgboost
    xgb = None  # type: ignore
    HAVE_XGB = False
    XGB_STATUS = f"xgboost not available: {e}"

_model_cache: Dict[Tuple[str, str], "xgb.XGBClassifier"] = {}  # type: ignore


def model_available_for(symbol: str, tf: str) -> bool:
    """Return True if a model file exists for this (symbol, tf)."""
    path = MODEL_DIR / f"xgb_{symbol}_{tf}.json"
    return path.exists()


def _load_model(symbol: str, tf: str):
    if not HAVE_XGB:
        raise RuntimeError(XGB_STATUS)
    key = (symbol, tf)
    if key in _model_cache:
        return _model_cache[key]
    path = MODEL_DIR / f"xgb_{symbol}_{tf}.json"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found for {symbol} {tf}: {path}")
    model = xgb.XGBClassifier()  # type: ignore
    model.load_model(str(path))
    _model_cache[key] = model
    return model


def predict_proba(symbol: str, tf: str, features: Dict[str, float]) -> float | None:
    """Return P(up) for this bar, or None if model/XGBoost unavailable."""
    if not HAVE_XGB:
        return None
    if not model_available_for(symbol, tf):
        return None

    model = _load_model(symbol, tf)
    x = np.array([[float(features[c]) for c in FEATURE_COLS]], dtype="float32")
    proba = model.predict_proba(x)[0, 1]
    return float(proba)