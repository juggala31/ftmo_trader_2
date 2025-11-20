from __future__ import annotations
import pathlib, sys
from typing import Dict, Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists(): return {}
    if yaml:
        try: return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception: return {}
    return {}

def _point_size(settings, symbol: str) -> float:
    try:
        return float((settings.sizing.get("points") or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

def _max_pts(cfg: Dict[str,Any], symbol: str) -> float:
    mp = (cfg.get("max_slippage_points") or {})
    if symbol in mp: return float(mp[symbol])
    base = symbol.split(".",1)[0]
    if base in mp: return float(mp[base])
    return float(cfg.get("default_points", 30))

def wrap_adapter_with_slippage(adapter, settings, cfg_path: str):
    cfg = _load_yaml(pathlib.Path(cfg_path))
    if not cfg.get("enabled", True): return
    rp = str(cfg.get("reason_prefix","SLIP"))
    r_cfg = cfg.get("retries") or {}
    do_retry = bool(r_cfg.get("enabled", True))
    attempts = int(r_cfg.get("attempts", 0))
    shrink   = float(r_cfg.get("shrink_factor", 0.5))

    base_place = adapter.place_order

    def place_checked(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        pt = _point_size(settings, symbol)
        want = None
        try:
            t = adapter.get_tick(symbol) or {}
            want = float(t["ask"] if side.lower()=="buy" else t["bid"])
        except Exception:
            pass

        def try_once(vol):
            return base_place(side=side, symbol=symbol, volume=vol, sl=sl, tp=tp, comment=comment)

        # first attempt
        res = try_once(volume)
        if not (res and res.get("ok") and want is not None):
            return res

        got = float(res.get("price") or want)
        slip_pts = abs(got - want) / (pt if pt>0 else 1.0)
        lim = _max_pts(cfg, symbol)
        if slip_pts <= lim:
            return res

        # retries
        if not do_retry or attempts <= 0:
            return {"ok": False, "message": f"{rp}: {slip_pts:.1f}pts>{lim:.1f} on {symbol}"}

        cur_vol = float(volume)
        for i in range(attempts):
            cur_vol *= float(shrink)
            if cur_vol <= 0:
                break
            r2 = try_once(cur_vol)
            if not (r2 and r2.get("ok") and want is not None):
                return r2
            got2 = float(r2.get("price") or want)
            slip2 = abs(got2 - want) / (pt if pt>0 else 1.0)
            if slip2 <= lim:
                return r2
        return {"ok": False, "message": f"{rp}: exceeded after retries on {symbol}"}

    adapter.place_order = place_checked  # type: ignore