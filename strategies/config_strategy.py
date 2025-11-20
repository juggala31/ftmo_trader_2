from __future__ import annotations
import time
from typing import List, Dict, Any

class ConfigStrategy:
    """
    Alternates buy/sell per symbol on an interval.
    Added (Phase H):
      - optional "risk_pct" per symbol
      - optional "use_atr": true with "atr_mult_sl/tp" to derive SL/TP points from ATR
    """
    def __init__(self, sym_cfg: List[Dict[str, Any]], defaults: Dict[str, Any]):
        self.defaults = defaults or {}
        self.syms: Dict[str, Dict[str, Any]] = {}
        for s in sym_cfg or []:
            name = s.get("name")
            if not name: 
                continue
            self.syms[name] = {**self.defaults, **s}
        if not self.syms:
            self.syms["XAUUSD"] = {**self.defaults}
        self._last_ts: Dict[str, float] = {k: 0.0 for k in self.syms}
        self._flip: Dict[str, bool] = {k: False for k in self.syms}

    def on_tick(self, tick: dict) -> None:
        pass

    def signals(self) -> List[Dict[str, Any]]:
        now = time.time()
        out: List[Dict[str, Any]] = []
        for name, cfg in self.syms.items():
            interval = float(cfg.get("interval_sec", self.defaults.get("interval_sec", 10.0)))
            if now - self._last_ts.get(name, 0.0) < interval:
                continue
            self._last_ts[name] = now
            self._flip[name] = not self._flip.get(name, False)
            side = "buy" if self._flip[name] else "sell"

            out_sig = {
                "side": side,
                "symbol": name,
                "volume": float(cfg.get("volume", self.defaults.get("volume", 0.10))),
                "comment": f"cfg-{side}",
                "autoclose_ms": int(cfg.get("autoclose_ms", self.defaults.get("autoclose_ms", 0))),
            }
            # Fixed SL/TP points (Phase D behavior)
            if "sl_points" in cfg: out_sig["sl_points"] = float(cfg["sl_points"])
            if "tp_points" in cfg: out_sig["tp_points"] = float(cfg["tp_points"])

            # Phase H: risk sizing & ATR flags passed to runner
            if "risk_pct" in cfg:
                out_sig["risk_pct"] = float(cfg["risk_pct"])
            if bool(cfg.get("use_atr", False)):
                out_sig["use_atr"] = True
                out_sig["atr_mult_sl"] = float(cfg.get("atr_mult_sl", cfg.get("atr_mult", 1.0)))
                out_sig["atr_mult_tp"] = float(cfg.get("atr_mult_tp", 2.0))

            out.append(out_sig)
        return out