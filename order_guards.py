from __future__ import annotations
import time, pathlib, sys
from typing import Dict, Any

_ROOT = pathlib.Path(__file__).resolve().parents[0].parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    if yaml:
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    # ultra-light fallback parser (key: value, flat)
    out: Dict[str, Any] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out

def _get_point_size(settings, symbol: str) -> float:
    try:
        return float((settings.sizing.get("points") or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

def wrap_adapter_with_guards(adapter, settings, cfg_path: str) -> None:
    """
    Monkey-patch adapter.place_order to enforce:
      - spread <= limit (in points)
      - max open positions (per_symbol, global)
    """
    cfg = _load_yaml(pathlib.Path(cfg_path))
    spread_limits = (cfg.get("spread_limits_points") or {})  # dict symbol->max points
    caps = (cfg.get("max_open") or {})
    cap_sym = int(caps.get("per_symbol") or 0)
    cap_glob = int(caps.get("global") or 0)
    reason_prefix = str(cfg.get("reason_prefix") or "GUARD")

    original_place = adapter.place_order

    def guarded_place(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        # 1) spread check
        t = adapter.get_tick(symbol)
        if not t:
            return {"ok": False, "message": f"{reason_prefix}: no tick for {symbol}"}
        bid = float(t["bid"]); ask = float(t["ask"])
        spread_abs = ask - bid
        pt = _get_point_size(settings, symbol)
        spread_pts = (spread_abs / pt) if pt > 0 else spread_abs

        lim_pts = None
        # try exact symbol first; else try a base (e.g., strip suffix like ".m")
        lim_pts = spread_limits.get(symbol)
        if lim_pts is None and "." in symbol:
            base = symbol.split(".",1)[0]
            lim_pts = spread_limits.get(base)
        if lim_pts is None:
            # also try common aliases (XAUUSD/GOLD, WTICOUSD/OIL, indices bases)
            alias = {"GOLD":"XAUUSD","XAU":"XAUUSD","OIL":"WTICOUSD","US30":"US30USD","US100":"NAS100USD","US500":"SPX500USD","BTC":"BTCUSD"}
            base = alias.get(symbol.upper(), None)
            if base:
                lim_pts = spread_limits.get(base)
        # enforce
        if lim_pts is not None:
            try:
                lim_pts = float(lim_pts)
                if spread_pts > lim_pts:
                    return {"ok": False, "message": f"{reason_prefix}: spread {spread_pts:.1f}pts>{lim_pts:.1f} on {symbol}"}
            except Exception:
                pass

        # 2) caps check
        try:
            open_tkts = adapter.open_tickets()
        except Exception:
            open_tkts = []
        if cap_glob and len(open_tkts) >= cap_glob:
            return {"ok": False, "message": f"{reason_prefix}: global cap reached {len(open_tkts)}/{cap_glob}"}
        if cap_sym:
            # need snapshot to count per symbol
            sym_count = 0
            # Try to infer per-symbol from paper map if present
            if hasattr(adapter, "_paper_positions") and isinstance(adapter._paper_positions, dict):
                for pos in adapter._paper_positions.values():
                    if str(pos.get("symbol")) == symbol:
                        sym_count += 1
            else:
                # live: query positions if MetaTrader5 available
                try:
                    import MetaTrader5 as mt5  # type: ignore
                    poss = mt5.positions_get() or []
                    for p in poss:
                        if str(getattr(p, "symbol", "")) == symbol:
                            sym_count += 1
                except Exception:
                    pass
            if sym_count >= cap_sym:
                return {"ok": False, "message": f"{reason_prefix}: cap for {symbol} reached {sym_count}/{cap_sym}"}

        # passthrough to original
        return original_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)

    # patch
    adapter.place_order = guarded_place  # type: ignore