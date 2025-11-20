from __future__ import annotations
import os, pathlib
from dataclasses import dataclass, field
from typing import Dict, List

_ROOT = pathlib.Path(__file__).resolve().parent
_ENV_PATH = _ROOT / ".env"

def _load_env(path: pathlib.Path) -> Dict[str, str]:
    env: Dict[str,str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env

def _getenv(envmap: Dict[str,str], key: str, default: str = "") -> str:
    return envmap.get(key, os.getenv(key, default)).strip()

def _resolve_symbols(defaults: List[str], envmap: Dict[str,str]) -> List[str]:
    raw = _getenv(envmap, "SYMBOLS", "")
    if not raw:
        return defaults
    seen = set()
    out: List[str] = []
    for tok in raw.split(","):
        s = tok.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or defaults

@dataclass
class Settings:
    app_mode: str = "live"     # "live" or "paper" etc.
    log_level: str = "INFO"
    mt5_login: str = ""
    mt5_password: str = ""
    mt5_server: str = ""
    symbols: List[str] = field(default_factory=lambda: ["XAUUSD","US30USD","NAS100USD","SPX500USD","BTCUSD","WTICOUSD"])
    # points per "point" (broker pip/point value normalizer)
    sizing: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "points": {
            "XAUUSD":   0.10,   # gold often quoted 0.01 or 0.10 point -> tune if needed
            "US30USD":  1.0,
            "NAS100USD":1.0,
            "SPX500USD":1.0,
            "WTICOUSD": 0.01,
            "BTCUSD":   1.0,
        }
    })
    # value-per-point for 1.0 volume (rough; adjust to your account/broker)
    vpp: Dict[str, float] = field(default_factory=lambda: {
        "XAUUSD":   1.0,
        "US30USD":  1.0,
        "NAS100USD":1.0,
        "SPX500USD":1.0,
        "WTICOUSD": 1.0,
        "BTCUSD":   1.0,
    })

def load_settings() -> Settings:
    envmap = _load_env(_ENV_PATH)

    s = Settings()
    s.app_mode    = _getenv(envmap, "APP_MODE", s.app_mode) or s.app_mode
    s.log_level   = _getenv(envmap, "LOG_LEVEL", s.log_level) or s.log_level
    s.mt5_login   = _getenv(envmap, "MT5_LOGIN", s.mt5_login)
    s.mt5_password= _getenv(envmap, "MT5_PASSWORD", s.mt5_password)
    s.mt5_server  = _getenv(envmap, "MT5_SERVER", s.mt5_server)

    # allow SYMBOLS override from .env
    s.symbols = _resolve_symbols(s.symbols, envmap)

    return s