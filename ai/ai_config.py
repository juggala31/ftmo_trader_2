from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _default_profile_path() -> pathlib.Path:
    """Return the default ai_profile.json path (same folder as this file)."""
    return pathlib.Path(__file__).with_name("ai_profile.json")


def load_profile(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the AI profile JSON once and cache it.

    Order of resolution:
      1) explicit *path* argument
      2) env AI_PROFILE_JSON
      3) ai/ai_profile.json next to this module
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        env_path = os.getenv("AI_PROFILE_JSON")
        if env_path:
            path_obj = pathlib.Path(env_path)
        else:
            path_obj = _default_profile_path()
    else:
        path_obj = pathlib.Path(path)

    try:
        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _CONFIG_CACHE = data
        print(f"[AI][cfg] loaded profile: {path_obj}", flush=True)
        return data
    except Exception as e:  # pragma: no cover - purely diagnostic
        print(f"[AI][cfg] WARNING: could not load profile {path_obj}: {e}", flush=True)
        _CONFIG_CACHE = None
        return None


def get_profile() -> Optional[Dict[str, Any]]:
    """Convenience accessor that always returns the cached profile (or None)."""
    return load_profile(path=None)


def _symbols_dict(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = profile.get("symbols", {})
    # Allow either dict(name -> cfg) or list[{'name':..., ...}]
    if isinstance(raw, dict):
        return {str(k): (v or {}) for k, v in raw.items()}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            out[str(name)] = item
    return out


def get_enabled_symbols(profile: Optional[Dict[str, Any]] = None) -> List[str]:
    p = profile or get_profile()
    if not p:
        return []
    out: List[str] = []
    for name, cfg in _symbols_dict(p).items():
        if cfg.get("enabled", True):
            out.append(name)
    return out


def get_enabled_tfs_for_symbol(symbol: str, profile: Optional[Dict[str, Any]] = None) -> List[str]:
    p = profile or get_profile()
    if not p:
        return []
    sym_cfg = _symbols_dict(p).get(symbol, {})
    tfs_cfg = sym_cfg.get("timeframes") or sym_cfg.get("tfs") or {}
    # If no explicit TFs, fall back to profile-level default or H1 only
    if not tfs_cfg:
        default_tfs = p.get("default_timeframes") or p.get("default_tfs") or ["H1"]
        return [str(tf) for tf in default_tfs]
    enabled: List[str] = []
    if isinstance(tfs_cfg, dict):
        for tf_name, tf_cfg in tfs_cfg.items():
            if isinstance(tf_cfg, dict):
                if tf_cfg.get("enabled", True):
                    enabled.append(str(tf_name))
            elif bool(tf_cfg):
                enabled.append(str(tf_name))
    else:
        # If someone used a simple list, treat all as enabled
        if isinstance(tfs_cfg, list):
            enabled.extend(str(x) for x in tfs_cfg)
    return enabled


def get_thresholds(
    symbol: str,
    tf: str,
    default_long: float,
    default_short: float,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    p = profile or get_profile()
    if not p:
        return default_long, default_short

    g = p.get("global", {})
    long_v = float(g.get("long_threshold", default_long))
    short_v = float(g.get("short_threshold", default_short))

    sym_cfg = _symbols_dict(p).get(symbol, {})
    if sym_cfg:
        long_v = float(sym_cfg.get("long_threshold", long_v))
        short_v = float(sym_cfg.get("short_threshold", short_v))

        tfs_cfg = sym_cfg.get("timeframes") or sym_cfg.get("tfs") or {}
        tf_cfg = tfs_cfg.get(tf) if isinstance(tfs_cfg, dict) else None
        if isinstance(tf_cfg, dict):
            long_v = float(tf_cfg.get("long_threshold", tf_cfg.get("long", long_v)))
            short_v = float(tf_cfg.get("short_threshold", tf_cfg.get("short", short_v)))

    return long_v, short_v


def apply_env_from_profile(profile: Optional[Dict[str, Any]] = None) -> None:
    """Translate JSON config into env vars that alpha_loop / alpha_exec_mt5 already understand.

    This keeps existing code paths working while letting the profile own the config.
    """
    p = profile or get_profile()
    if not p:
        return

    g = p.get("global", {})

    # Global toggles
    if "dry_run" in g and os.getenv("DRY_RUN_AI") is None:
        os.environ["DRY_RUN_AI"] = "1" if g.get("dry_run") else "0"

    if "loop_sleep_sec" in g and os.getenv("AI_LOOP_SLEEP_SEC") is None:
        os.environ["AI_LOOP_SLEEP_SEC"] = str(g.get("loop_sleep_sec"))

    if "long_threshold" in g and os.getenv("AI_PUP_LONG") is None:
        os.environ["AI_PUP_LONG"] = str(g.get("long_threshold"))

    if "short_threshold" in g and os.getenv("AI_PUP_SHORT") is None:
        os.environ["AI_PUP_SHORT"] = str(g.get("short_threshold"))

    # Enabled symbols / tfs for compatibility with old env-based config
    syms = get_enabled_symbols(p)
    if syms and os.getenv("AI_SYMBOLS") is None:
        os.environ["AI_SYMBOLS"] = ",".join(syms)

    # Union of all enabled TFs across symbols
    tf_set = set()
    for s in syms:
        for tf in get_enabled_tfs_for_symbol(s, p):
            tf_set.add(tf)
    if tf_set and os.getenv("AI_TFS") is None:
        os.environ["AI_TFS"] = ",".join(sorted(tf_set))

    # Per-symbol risk / execution tweaks
    sym_dict = _symbols_dict(p)
    for symbol, cfg in sym_dict.items():
        if not cfg.get("enabled", True):
            continue
        sym_key = symbol.replace(".", "_")
        risk = cfg.get("risk", {}) or {}
        mode = str(risk.get("mode", "FIXED_LOT")).upper()
        if os.getenv(f"AI_RISK_MODE_{sym_key}") is None:
            os.environ[f"AI_RISK_MODE_{sym_key}"] = mode

        lots = risk.get("lots", risk.get("fixed_lot"))
        if lots is not None and os.getenv(f"AI_LOTS_{sym_key}") is None:
            os.environ[f"AI_LOTS_{sym_key}"] = str(lots)

        sl_mult = risk.get("sl_mult")
        if sl_mult is not None and os.getenv(f"AI_SL_MULT_{sym_key}") is None:
            os.environ[f"AI_SL_MULT_{sym_key}"] = str(sl_mult)

        tp_mult = risk.get("tp_mult")
        if tp_mult is not None and os.getenv(f"AI_TP_MULT_{sym_key}") is None:
            os.environ[f"AI_TP_MULT_{sym_key}"] = str(tp_mult)

        deviation = risk.get("deviation")
        if deviation is not None and os.getenv(f"AI_DEVIATION_{sym_key}") is None:
            os.environ[f"AI_DEVIATION_{sym_key}"] = str(int(deviation))

        buf_pts = risk.get("stops_buffer_points", risk.get("stops_buffer"))
        if buf_pts is not None and os.getenv(f"AI_STOPS_BUFFER_POINTS_{sym_key}") is None:
            os.environ[f"AI_STOPS_BUFFER_POINTS_{sym_key}"] = str(int(buf_pts))

        # Optional ATR-risk fields (not yet used by alpha_exec_mt5, but reserved)
        risk_pct = risk.get("atr_risk_pct", risk.get("risk_pct"))
        if risk_pct is not None and os.getenv(f"AI_RISK_PCT_{sym_key}") is None:
            os.environ[f"AI_RISK_PCT_{sym_key}"] = str(risk_pct)


__all__ = [
    "load_profile",
    "get_profile",
    "get_enabled_symbols",
    "get_enabled_tfs_for_symbol",
    "get_thresholds",
    "apply_env_from_profile",
]