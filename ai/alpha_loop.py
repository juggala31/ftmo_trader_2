"""
AI alpha loop for Trader 2.0

- Pulls bars from MT5 for a list of symbols / timeframes.
- Builds features and queries XGBoost (if available).
- Falls back to an EMA/RSI rule if no model.
- Logs signals to ai/ai_signals.csv.
- If DRY_RUN_AI=0 and ai.alpha_exec_mt5.maybe_flip_position is available,
  it will open/hold/flip a single MT5 position per symbol.

Features:
  * Only acts once per NEW bar (per symbol×TF).
  * XGBoost thresholds configurable via env:
      AI_PUP_LONG  (default 0.60)
      AI_PUP_SHORT (default 0.40)
  * Session filter:
      AI_SESSION_START_HOUR (default 7)
      AI_SESSION_END_HOUR   (default 20)
  * Multi-timeframe trend alignment:
      AI_MTF_ENABLE  (default 1 = on)
      AI_MTF_TF      (default H4)
  * Entry vs confirmation timeframes:
      AI_ENTRY_TF        (default M30)
      AI_CONFIRM_TF      (default H1)
      AI_CONFIRM_ENABLE  (default 1 = on)
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"[AI] MetaTrader5 module unavailable: {e}")


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AI_DIR = ROOT / "ai"
SIGNALS_CSV = AI_DIR / "ai_signals.csv"
SESSION_FILTERS_PATH = AI_DIR / "session_filters.json"


def _ensure_mt5_init() -> None:
    if mt5.initialize():
        print("[MT5] initialize() OK", flush=True)
        return
    print("[MT5] initialize() failed, trying with login/env...", flush=True)

    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    if not login or not password or not server:
        raise RuntimeError("MT5 initialize() failed and MT5_LOGIN/PASSWORD/SERVER not set")

    ok = mt5.initialize(login=int(login), password=password, server=server)
    if not ok:
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print(f"[MT5] initialize() -> {mt5.last_error()}", flush=True)


def get_bars_and_rates(symbol: str, tf: str, n: int = 300):
    mt5_tf = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }.get(tf)
    if mt5_tf is None:
        print(f"[AI] unsupported timeframe {tf} for {symbol}", flush=True)
        return None, None

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, n)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"[AI] no rates for {symbol} {tf} -> {err}", flush=True)
        return None, None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour"] = df["time"].dt.hour
    return df, rates


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int = 50) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def basic_signal(close: float, ema50: float, rsi: float) -> int:
    if close > ema50 and rsi > 55.0:
        return 1
    if close < ema50 and rsi < 45.0:
        return -1
    return 0


_maybe_flip_position = None
_EXEC_IMPORT_ERR = None
_warned_missing_exec = False

try:
    from ai.alpha_exec_mt5 import maybe_flip_position as _maybe_flip_position
except Exception as e1:  # pragma: no cover
    try:
        from .alpha_exec_mt5 import maybe_flip_position as _maybe_flip_position  # type: ignore
    except Exception as e2:  # pragma: no cover
        _EXEC_IMPORT_ERR = (e1, e2)
        _maybe_flip_position = None

try:
    from ai.xgb_loader_api import (
        HAVE_XGB_INF,
        XGB_FEATURE_COLS,
        xgb_predict_proba,
    )
except Exception:
    HAVE_XGB_INF = False
    XGB_FEATURE_COLS: List[str] = []
    xgb_predict_proba = None  # type: ignore


def getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


def getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


def getenv_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "y", "yes", "on"):
        return True
    if v in ("0", "false", "n", "no", "off"):
        return False
    return default


DEFAULT_SYMBOLS: List[str] = [
    "XAUZ25.sim",
    "US30Z25.sim",
    "US100Z25.sim",
    "US500Z25.sim",
    "USOILZ25.sim",
    "BTCX25.sim",
]

DEFAULT_TFS: List[str] = ["H1"]

TF_MAP: Dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

SYMBOLS: List[str] = [
    s.strip()
    for s in os.getenv("AI_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",")
    if s.strip()
]

TFS: List[str] = [
    t.strip()
    for t in os.getenv("AI_TFS", ",".join(DEFAULT_TFS)).split(",")
    if t.strip()
]

LONG_THRESH: float = getenv_float("AI_PUP_LONG", 0.60)
SHORT_THRESH: float = getenv_float("AI_PUP_SHORT", 0.40)

SESSION_START_HOUR: int = getenv_int("AI_SESSION_START_HOUR", 7)
SESSION_END_HOUR: int = getenv_int("AI_SESSION_END_HOUR", 20)

AI_MTF_ENABLE: bool = getenv_bool("AI_MTF_ENABLE", True)
AI_MTF_TF: str = os.getenv("AI_MTF_TF", "H4").upper()

DRY_RUN: bool = getenv_bool("DRY_RUN_AI", True)
SLEEP_SEC: float = getenv_float("AI_LOOP_SLEEP_SEC", 30.0)

ENTRY_TF: str = os.getenv("AI_ENTRY_TF", "M30").upper()
CONFIRM_TF: str = os.getenv("AI_CONFIRM_TF", "H1").upper()
AI_CONFIRM_ENABLE: bool = getenv_bool("AI_CONFIRM_ENABLE", True)


def _load_session_filters() -> Dict[str, List[str]]:
    try:
        if not SESSION_FILTERS_PATH.exists():
            print("[AI] session_filters.json not found; no per-symbol session filters.", flush=True)
            return {}
        import json
        with SESSION_FILTERS_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, List[str]] = {}
        for key, val in raw.items():
            if not isinstance(val, (list, tuple)):
                continue
            sym = str(key)
            out[sym] = [str(s).upper() for s in val]
        print(f"[AI] Loaded session_filters.json for {len(out)} symbols.", flush=True)
        return out
    except Exception as e:
        print(f"[AI] Failed to load session_filters.json: {e}", flush=True)
        return {}


def map_hour_to_session(hour: int) -> str:
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 20:
        return "NY"
    return "OFF"


SESSION_FILTERS: Dict[str, List[str]] = _load_session_filters()


def ensure_signals_csv_header() -> None:
    if SIGNALS_CSV.exists():
        return
    SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SIGNALS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "loop_ts",
                "symbol",
                "tf",
                "close",
                "rsi14",
                "ema50",
                "p_up",
                "signal",
                "src",
                "hour",
                "session_ok",
                "htf_tf",
                "htf_trend",
            ]
        )


def compute_htf_trend(symbol: str, tf: str) -> Tuple[int, Optional[float], Optional[float]]:
    df_htf, _ = get_bars_and_rates(symbol, tf, n=200)
    if df_htf is None or len(df_htf) < 50:
        return 0, None, None
    close_htf = float(df_htf["close"].iloc[-1])
    ema50_htf = float(compute_ema(df_htf["close"], span=50).iloc[-1])
    if math.isnan(ema50_htf):
        return 0, close_htf, ema50_htf
    if close_htf > ema50_htf:
        return 1, close_htf, ema50_htf
    if close_htf < ema50_htf:
        return -1, close_htf, ema50_htf
    return 0, close_htf, ema50_htf


def main() -> None:
    print(f"[AI] symbols={SYMBOLS}  tfs={TFS}", flush=True)
    print(f"[AI] DRY_RUN={DRY_RUN}  HAVE_XGB_INF={HAVE_XGB_INF}", flush=True)
    print(f"[AI] thresholds LONG={LONG_THRESH:.2f} SHORT={SHORT_THRESH:.2f}", flush=True)
    print(f"[AI] entry_tf={ENTRY_TF} confirm_tf={CONFIRM_TF} confirm_enable={AI_CONFIRM_ENABLE}", flush=True)
    print(f"[AI] session_filters symbols={len(SESSION_FILTERS)}", flush=True)

    mtf_on = AI_MTF_ENABLE and AI_MTF_TF in TF_MAP
    if mtf_on:
        print(f"[AI] multi-timeframe filter: ENABLED (higher TF={AI_MTF_TF})", flush=True)
    else:
        print("[AI] multi-timeframe filter: DISABLED", flush=True)

    _ensure_mt5_init()
    ensure_signals_csv_header()

    last_bar_time: Dict[Tuple[str, str], int] = {}

    while True:
        loop_ts = time.time()

        # Higher-TF trend snapshot
        htf_trend_map: Dict[str, int] = {}
        if mtf_on:
            for symbol in SYMBOLS:
                trend, close_htf, ema50_htf = compute_htf_trend(symbol, AI_MTF_TF)
                htf_trend_map[symbol] = trend
                print(
                    f"[AI] HTF {symbol} {AI_MTF_TF}: trend={trend} (+1=up,-1=down,0=flat) "
                    f"close_htf={close_htf} ema50_htf={ema50_htf}",
                    flush=True,
                )

        # Confirmation-TF trend snapshot
        confirm_trend_map: Dict[str, int] = {}
        confirm_on = AI_CONFIRM_ENABLE and CONFIRM_TF in TF_MAP
        if confirm_on:
            for symbol in SYMBOLS:
                c_trend, c_close, c_ema = compute_htf_trend(symbol, CONFIRM_TF)
                confirm_trend_map[symbol] = c_trend
                print(
                    f"[AI] CONFIRM {symbol} {CONFIRM_TF}: trend={c_trend} (+1=up,-1=down,0=flat) "
                    f"close={c_close} ema50={c_ema}",
                    flush=True,
                )

        for symbol in SYMBOLS:
            for tf in TFS:
                df, rates = get_bars_and_rates(symbol, tf, n=300)
                if df is None or rates is None or len(df) < 50:
                    continue

                last_t = int(df["time"].iloc[-1].timestamp())
                key = (symbol, tf)
                if last_bar_time.get(key) == last_t:
                    continue
                last_bar_time[key] = last_t

                df["ema50"] = compute_ema(df["close"], span=50)
                df["rsi14"] = compute_rsi(df["close"], period=14)

                row = df.iloc[-1]
                close = float(row["close"])
                ema50_val = float(row["ema50"])
                rsi_val = float(row["rsi14"])

                features: Dict[str, float] = {}
                if HAVE_XGB_INF and XGB_FEATURE_COLS:
                    for col in XGB_FEATURE_COLS:
                        val = row.get(col, np.nan)
                        features[col] = float(val) if not pd.isna(val) else float("nan")

                p_up: Optional[float] = None
                sig_source = "rule"
                sig = 0

                if HAVE_XGB_INF and xgb_predict_proba is not None and features:
                    if all(not math.isnan(features[c]) for c in XGB_FEATURE_COLS):
                        try:
                            p_up = xgb_predict_proba(symbol, tf, features)
                            if p_up is not None:
                                sig_source = "xgb"
                                if p_up >= LONG_THRESH:
                                    sig = 1
                                elif p_up <= SHORT_THRESH:
                                    sig = -1
                                else:
                                    sig = 0
                        except Exception as e:
                            print(f"[AI] xgb_predict_proba failed for {symbol} {tf}: {e}", flush=True)

                if sig_source == "rule":
                    sig = basic_signal(close, ema50_val, rsi_val)

                session_ok = True
                session_name = ""
                try:
                    hour_val = int(row.get("hour", -1))
                except Exception:
                    hour_val = -1

                if hour_val >= 0:
                    if SESSION_START_HOUR <= SESSION_END_HOUR:
                        if not (SESSION_START_HOUR <= hour_val < SESSION_END_HOUR):
                            session_ok = False
                    else:
                        if not (hour_val >= SESSION_START_HOUR or hour_val < SESSION_END_HOUR):
                            session_ok = False

                    session_name = map_hour_to_session(hour_val)
                    allowed_sessions = SESSION_FILTERS.get(symbol)
                    if allowed_sessions is not None and session_name:
                        if session_name.upper() not in allowed_sessions:
                            session_ok = False
                else:
                    session_ok = True

                if not session_ok and sig != 0:
                    print(
                        f"[AI] {symbol} {tf} session filter blocking sig={sig} at hour={hour_val} session={session_name}",
                        flush=True,
                    )
                    sig = 0

                htf_trend = htf_trend_map.get(symbol, 0)
                if mtf_on and tf == ENTRY_TF and sig != 0:
                    if htf_trend == 1 and sig < 0:
                        print(
                            f"[AI] {symbol} {tf} MTF filter: HTF={AI_MTF_TF} trend=UP, blocking SHORT sig={sig}",
                            flush=True,
                        )
                        sig = 0
                    elif htf_trend == -1 and sig > 0:
                        print(
                            f"[AI] {symbol} {tf} MTF filter: HTF={AI_MTF_TF} trend=DOWN, blocking LONG sig={sig}",
                            flush=True,
                        )
                        sig = 0

                confirm_trend = confirm_trend_map.get(symbol, 0)
                if confirm_on and tf == ENTRY_TF and sig != 0:
                    if confirm_trend == 1 and sig < 0:
                        print(
                            f"[AI] {symbol} {tf} confirm filter: CONFIRM_TF={CONFIRM_TF} trend=UP, blocking SHORT sig={sig}",
                            flush=True,
                        )
                        sig = 0
                    elif confirm_trend == -1 and sig > 0:
                        print(
                            f"[AI] {symbol} {tf} confirm filter: CONFIRM_TF={CONFIRM_TF} trend=DOWN, blocking LONG sig={sig}",
                            flush=True,
                        )
                        sig = 0

                print(
                    f"[AI] {symbol} {tf} price={close:.2f} rsi14={rsi_val:.1f} ema50={ema50_val:.2f} "
                    f"p_up={p_up if p_up is not None else 'NA'} sig={sig} src={sig_source} "
                    f"hour={hour_val} session_ok={session_ok} session={session_name} "
                    f"htf_tf={AI_MTF_TF if mtf_on else ''} htf_trend={htf_trend}",
                    flush=True,
                )

                with SIGNALS_CSV.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            loop_ts,
                            symbol,
                            tf,
                            close,
                            rsi_val,
                            ema50_val,
                            p_up if p_up is not None else "",
                            sig,
                            sig_source,
                            hour_val,
                            1 if session_ok else 0,
                            AI_MTF_TF if mtf_on else "",
                            htf_trend,
                        ]
                    )

                if not DRY_RUN and _maybe_flip_position is not None and tf == ENTRY_TF:
                    try:
                        _maybe_flip_position(
                            symbol=symbol,
                            signal=float(sig),
                            rates=rates,
                        )
                    except Exception as e:
                        global _warned_missing_exec
                        if not _warned_missing_exec:
                            print(f"[AI] maybe_flip_position error: {e}", flush=True)
                            _warned_missing_exec = True

        time.sleep(SLEEP_SEC)


def start_ai_loop(controller=None):
    if controller is not None:
        try:
            setattr(controller, "_ai_loop_started", True)
        except Exception:
            pass
    t = threading.Thread(target=main, name="alpha_loop", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    main()