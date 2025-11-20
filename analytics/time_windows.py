from __future__ import annotations
import pathlib, sys, time, datetime as dt
from typing import Dict, Any, Tuple

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except Exception:
    yaml = None

# cooldown state in-memory
_last_outcome: Dict[str, Tuple[float,bool]] = {}  # symbol -> (ts, is_loss)

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    if yaml:
        try: return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception: return {}
    return {}

def _now_tz(tzname: str) -> dt.datetime:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tzname)
        return dt.datetime.now(tz)
    except Exception:
        return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def _in_window(now: dt.datetime, windows) -> bool:
    if not windows: return True
    h = now.hour; m = now.minute
    cur = h*60+m
    for w in windows:
        a,b = w.split("-",1)
        ah,am = [int(x) for x in a.split(":")]
        bh,bm = [int(x) for x in b.split(":")]
        start = ah*60+am; end = bh*60+bm
        if start <= end:
            if start <= cur <= end: return True
        else:
            # spans midnight
            if cur >= start or cur <= end: return True
    return False

def _cooldown_left(symbol: str, cfg: Dict[str,Any]) -> int:
    item = _last_outcome.get(symbol)
    if not item: return 0
    ts, is_loss = item
    cd = cfg.get("cooldowns") or {}
    dflt = cd.get("default") or {}
    mins = int(dflt.get("after_loss" if is_loss else "after_win", 0))
    left = int(max(0, ts + mins*60 - time.time()))
    if left == 0 and symbol in _last_outcome:
        _last_outcome.pop(symbol, None)
    return left

def wrap_adapter_with_time_windows(adapter, settings, cfg_path: str):
    cfg = _load_yaml(pathlib.Path(cfg_path))
    if not cfg.get("enabled", True): return
    tz = str(cfg.get("timezone","UTC"))
    rp = str(cfg.get("reason_prefix","TIME"))
    allowed = cfg.get("allowed") or {}

    base_place = adapter.place_order
    base_close = adapter.close_order if hasattr(adapter, "close_order") else None

    def place_guard(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        now = _now_tz(tz)
        windows = allowed.get(symbol) or allowed.get(symbol.split(".",1)[0]) or []
        if not _in_window(now, windows):
            return {"ok": False, "message": f"{rp}: outside allowed window for {symbol}"}
        left = _cooldown_left(symbol, cfg)
        if left > 0:
            return {"ok": False, "message": f"{rp}: cooldown {left}s remaining on {symbol}"}
        return base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)

    def closed(ticket: int):
        res = {"ok": False}
        if base_close:
            res = base_close(ticket)
        # infer pnl sign from res if provided; fallback none
        try:
            pnl = float(res.get("pnl_ccy") or 0.0)
        except Exception:
            pnl = 0.0
        sym = str(res.get("symbol") or "")
        if sym:
            _last_outcome[sym] = (time.time(), pnl < 0)
        return res

    adapter.place_order = place_guard  # type: ignore
    if base_close:
        adapter.close_order = closed  # type: ignore