"""
XGBoost loader API for Trader 2.0.

This is the single integration point for all XGBoost model usage.

Exports:
    - HAVE_XGB_INF: bool
    - XGB_FEATURE_COLS: List[str]
    - xgb_predict_proba(symbol, tf, features) -> float | None

Internally delegates to ai.xgb_infer, which:
    - Knows where the xgb_*.json models live
    - Defines FEATURE_COLS
    - Implements predict_proba()
"""

from __future__ import annotations

from typing import Dict, List, Optional

HAVE_XGB_INF: bool = False
XGB_FEATURE_COLS: List[str] = []


def xgb_predict_proba(symbol: str, tf: str, features: Dict[str, float]) -> Optional[float]:
    """Return P(up) or None if model/XGBoost unavailable."""
    return None


try:
    # Import the lower-level helper
    from ai import xgb_infer as _infer  # type: ignore

    # FEATURE_COLS from xgb_infer is our canonical feature list
    cols = getattr(_infer, "FEATURE_COLS", [])
    if isinstance(cols, (list, tuple)):
        XGB_FEATURE_COLS = list(cols)
    else:
        XGB_FEATURE_COLS = []

    # XGBoost availability flag in xgb_infer
    HAVE_XGB_INF = bool(getattr(_infer, "HAVE_XGB", False)) and bool(XGB_FEATURE_COLS)

    def xgb_predict_proba(symbol: str, tf: str, features: Dict[str, float]) -> Optional[float]:  # type: ignore[override]
        """Delegate to ai.xgb_infer.predict_proba, catching any runtime errors."""
        try:
            return _infer.predict_proba(symbol, tf, features)  # type: ignore[attr-defined]
        except Exception:
            # Any problem: treat as "no prediction"
            return None

except Exception:
    # Hard failure importing ai.xgb_infer → no ML backend
    HAVE_XGB_INF = False
    XGB_FEATURE_COLS = []
