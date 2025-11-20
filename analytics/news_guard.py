from __future__ import annotations
import json, time, pathlib, sys, datetime as dt
from typing import Dict, Any, List

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except Exception:
    yaml = None

CACHE = _ROOT / "config" / "news_cache.json"

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    if yaml:
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    out={}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k,v=line.split(":",1); out[k.strip()]=v.strip()
    return out

def _load_cache() -> List[Dict[str, Any]]:
    if not CACHE.exists():
        return []
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return []

def _symbol_currencies(cfg: Dict[str,Any], symbol: str) -> List[str]:
    sc = cfg.get("symbol_currencies", {}) or {}
    for k,v in sc.items():
        if k.upper() == symbol.upper(): return [x.upper() for x in v]
    # fallbacks
    base = symbol.split(".",1)[0].upper()
    bc = cfg.get("base_currencies", {}) or {}
    for k,v in bc.items():
        if k.upper() == base: return [x.upper() for x in v]
    # special aliases
    alias = {"GOLD":"XAUUSD","XAU":"XAUUSD","OIL":"WTICOUSD","US30":"US30USD","US100":"NAS100USD","US500":"SPX500USD","BTC":"BTCUSD"}
    a = alias.get(base)
    if a and a in sc: return [x.upper() for x in sc[a]]
    return []

def _impact_window(cfg: Dict[str,Any], impact: str) -> tuple[int,int]:
    w = (cfg.get("windows", {}) or {}).get(impact.lower(), {})
    b = int(w.get("before", 0)); a = int(w.get("after", 0))
    return b, a

def wrap_adapter_with_news_guard(adapter, settings, cfg_path: str):
    """
    Blocks new orders if now is within [event-before, event+after] for any
    event whose currency intersects the symbol's driving currencies.
    """
    cfg = _load_yaml(pathlib.Path(cfg_path))
    if not cfg.get("enabled", True):
        return
    reason_prefix = str(cfg.get("reason_prefix", "NEWS"))

    base_place = adapter.place_order

    # local cache refreshed once per 30s
    _ev = {"ts": 0.0, "data": []}
    def _events():
        now = time.time()
        if now - _ev["ts"] > 30.0:
            _ev["data"] = _load_cache()
            _ev["ts"] = now
        return _ev["data"]

    def guarded_place(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        # currencies for this symbol
        curs = _symbol_currencies(cfg, symbol)
        if not curs:
            return base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)

        now = time.time()  # UTC epoch
        block_msgs = []
        for ev in _events():
            cur = str(ev.get("currency","")).upper()
            if cur not in curs: continue
            impact = str(ev.get("impact","low")).lower()
            t_ev = float(ev.get("ts_utc", 0.0))
            b,a = _impact_window(cfg, impact)
            if t_ev - (b*60) <= now <= t_ev + (a*60):
                ttl = ev.get("title","")
                block_msgs.append(f"{cur}/{impact} {b}m↔{a}m {ttl}")

        if block_msgs:
            return {"ok": False, "message": f"{reason_prefix}: in event window: " + " | ".join(block_msgs[:2])}

        return base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)

    adapter.place_order = guarded_place  # type: ignore