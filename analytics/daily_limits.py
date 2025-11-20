from __future__ import annotations
import pathlib, sys, datetime as dt, time
from typing import Dict, Any, Optional

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    if yaml:
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    # fallback
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k,v = line.split(":",1); out[k.strip()] = v.strip()
    return out

def _now_in_tz(tzname: str) -> dt.datetime:
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tzname)
        return dt.datetime.now(tz)
    except Exception:
        return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def _today_bounds(tzname: str, reset_time: str) -> tuple[dt.datetime, dt.datetime]:
    now = _now_in_tz(tzname)
    if reset_time:
        hh, mm = [int(x) for x in reset_time.split(":")]
        anchor = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        # window from last anchor to next anchor
        if now >= anchor:
            start = anchor
            end = anchor + dt.timedelta(days=1)
        else:
            start = anchor - dt.timedelta(days=1)
            end = anchor
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=1)
    # return in UTC for MT5 history queries
    return (start.astimezone(dt.timezone.utc), end.astimezone(dt.timezone.utc))

def _pnl_today_mt5(use_unrealized: bool) -> float:
    """Get realized PnL from today's deals + optional unrealized from open positions."""
    if mt5 is None:
        return 0.0
    # realized: sum of profit from deals (close)
    # time range: today UTC-ish (we'll use a broad window)
    utcnow = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    start = utcnow - dt.timedelta(days=2)
    deals = mt5.history_deals_get(start, utcnow) or []
    realized = 0.0
    for d in deals:
        try:
            realized += float(getattr(d, "profit", 0.0))
        except Exception:
            pass
    # unrealized
    unreal = 0.0
    if use_unrealized:
        poss = mt5.positions_get() or []
        for p in poss:
            try:
                unreal += float(getattr(p, "profit", 0.0))
            except Exception:
                pass
    return realized + unreal

def wrap_adapter_with_daily_limits(adapter, settings, cfg_path: str):
    """
    Blocks new orders when today's PnL breaches caps.
    Uses MT5 history_deals_get + positions_get; independent of strategy logic.
    """
    cfg = _load_yaml(pathlib.Path(cfg_path))
    if not cfg.get("enabled", True):
        return
    tzname = str(cfg.get("timezone", "UTC"))
    dl_cap = float(cfg.get("daily_loss_cap", -float("inf")))
    dp_cap = float(cfg.get("daily_profit_cap", float("inf")))
    use_realized = bool(cfg.get("use_realized", True))
    use_unreal = bool(cfg.get("use_unrealized", True))
    reset_time = str(cfg.get("reset_time") or "")
    reason_prefix = str(cfg.get("reason_prefix", "LIMIT"))

    base_place = adapter.place_order

    # small cache to avoid hammering MT5 each call
    _cache = {"ts": 0.0, "pnl": 0.0}
    def _pnl_today():
        now = time.time()
        if now - _cache["ts"] > 5.0:  # refresh every 5s
            # we always compute realized; unrealized if requested
            _cache["pnl"] = _pnl_today_mt5(use_unreal)
            _cache["ts"] = now
        return _cache["pnl"]

    def guarded_place(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        pnl = _pnl_today()
        if pnl <= dl_cap:
            return {"ok": False, "message": f"{reason_prefix}: daily PnL {pnl:+.2f} <= cap {dl_cap:+.2f}"}
        if pnl >= dp_cap:
            return {"ok": False, "message": f"{reason_prefix}: daily PnL {pnl:+.2f} >= profit cap {dp_cap:+.2f}"}
        return base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)

    adapter.place_order = guarded_place  # type: ignore