# -*- coding: utf-8 -*-
"""
brokers/mt5_adapter.py

Thin wrapper around the MetaTrader5 module so the rest of the codebase/tools
have a stable, testable surface. No external deps.

Features:
- connect(login, password, server=None) with graceful "attach to existing terminal"
- symbol helpers: visible_symbols(), ensure_selected(), symbol_info(), last_tick()
- order_market() with volume normalization (min/max/step) and fill-mode fallback [RET -> IOC -> FOK]
- constants() exposing key MT5 constants for callers (retcodes, order types)
"""

from __future__ import annotations
import os
from typing import Dict, Optional

try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None

class MT5Adapter:
    def __init__(self):
        self.mt5 = mt5
        self.connected = False
        self.login = None
        self.server = None

    # ---------- connect API ----------
    def connect(self, login: Optional[int]=None, password: Optional[str]=None, server: Optional[str]=None):
        """
        Primary signature. Compatible with scripts that call:
          connect(l, p, s)  OR  connect(l, p)
        We first try to attach to an already-open terminal, and if that fails,
        try explicit initialize(login/password/server) if given.
        Returns: (ok: bool, message: str)
        """
        if self.mt5 is None:
            return (False, "MetaTrader5 module not available")

        # 1) try attach to already-running terminal
        try:
            if self.mt5.initialize():
                ai = self.mt5.account_info()
                if ai:
                    self.connected = True
                    self.login = getattr(ai, "login", None)
                    self.server = getattr(ai, "server", None)
                    return (True, f"attached {self.login} {self.server}")
        except Exception:
            pass

        # 2) try explicit login
        if login and password:
            try:
                if server:
                    ok = self.mt5.initialize(login=int(login), password=password, server=str(server))
                else:
                    ok = self.mt5.initialize(login=int(login), password=password)
                if ok:
                    ai = self.mt5.account_info()
                    self.connected = True
                    self.login = getattr(ai, "login", None)
                    self.server = getattr(ai, "server", None)
                    return (True, f"logged in {self.login} {self.server}")
                # not ok -> get reason
                try:
                    code, msg = self.mt5.last_error()
                    return (False, f"mt5.initialize() failed: ({code}, '{msg}')")
                except Exception:
                    return (False, "mt5.initialize() failed")
            except Exception as e:
                try:
                    code, msg = self.mt5.last_error()
                    return (False, f"mt5.initialize() exception: ({code}, '{msg}')")
                except Exception:
                    return (False, f"mt5.initialize() exception: {e}")

        return (False, "no terminal attached and no credentials provided")

    # Backward-compat overload used by some diagnostics
    def connect_l_p(self, login: int, password: str):
        return self.connect(login, password, None)

    # ---------- symbol helpers ----------
    def visible_symbols(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if not self.mt5: return out
        try:
            info = self.mt5.symbols_get()
            for s in info or []:
                if getattr(s, "visible", False):
                    name = getattr(s, "name", "")
                    out[name] = {"name": name, "visible": True}
        except Exception:
            pass
        return out

    def ensure_selected(self, broker_symbol: str) -> bool:
        if not self.mt5: return False
        try:
            return bool(self.mt5.symbol_select(broker_symbol, True))
        except Exception:
            return False

    def symbol_info(self, broker_symbol: str):
        if not self.mt5: return None
        try:
            return self.mt5.symbol_info(broker_symbol)
        except Exception:
            return None

    def last_tick(self, broker_symbol: str) -> Optional[dict]:
        if not self.mt5: return None
        try:
            t = self.mt5.symbol_info_tick(broker_symbol)
            if t is None: return None
            bid = getattr(t, "bid", None)
            ask = getattr(t, "ask", None)
            if not bid or not ask: return None
            mid = (float(bid) + float(ask)) / 2.0
            return {"bid": float(bid), "ask": float(ask), "mid": mid}
        except Exception:
            return None

    # ---------- execution helpers ----------
    def _normalize_volume(self, info, desired: float) -> float:
        try:
            vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
            vmax = float(getattr(info, "volume_max", 100.0) or 100.0)
            vstep= float(getattr(info, "volume_step", 0.01) or 0.01)
        except Exception:
            vmin, vmax, vstep = 0.01, 100.0, 0.01
        vol = max(vmin, min(vmax, float(desired)))
        steps = max(1, int(vol / vstep + 1e-8))
        vol = steps * vstep
        if vol < vmin:
            vol = vmin
        # precision from step
        prec = 2
        try:
            s = f"{vstep:.10f}".rstrip("0")
            if "." in s: prec = len(s.split(".")[1])
        except Exception:
            pass
        return float(f"{vol:.{prec}f}")

    def _fill_modes(self, info):
        m = self.mt5
        FOK = getattr(m, "ORDER_FILLING_FOK", 0)
        IOC = getattr(m, "ORDER_FILLING_IOC", 1)
        RET = getattr(m, "ORDER_FILLING_RETURN", 2)
        modes = [RET, IOC, FOK]
        try:
            adv = getattr(info, "fill_mode", None) or getattr(info, "trade_fill_mode", None)
            if adv in (FOK, IOC, RET):
                modes = [adv] + [x for x in modes if x != adv]
        except Exception:
            pass
        return modes

    def order_market(self, broker_symbol: str, side: str, volume: float) -> dict:
        """
        Places a market order with fill-mode fallback.
        Returns a dict with retcode, order/deal, price, comment, normalized_volume, fill_mode
        """
        if not self.mt5:
            return {"comment": "mt5 module not available"}
        info = self.symbol_info(broker_symbol)
        if not info:
            return {"comment": "symbol_info unavailable"}

        norm_vol = self._normalize_volume(info, volume)
        modes = self._fill_modes(info)

        action = self.mt5.TRADE_ACTION_DEAL
        typ    = self.mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else self.mt5.ORDER_TYPE_SELL

        last = None
        for fill in modes:
            req = {
                "action": action,
                "symbol": broker_symbol,
                "volume": float(norm_vol),
                "type": typ,
                "deviation": 50,
                "type_filling": fill,
                "type_time": getattr(self.mt5, "ORDER_TIME_GTC", 0),
                "magic": 0,
                "comment": "adapter",
            }
            res = self.mt5.order_send(req)
            last = {
                "retcode": getattr(res, "retcode", None),
                "order": getattr(res, "order", None),
                "deal": getattr(res, "deal", None),
                "price": getattr(res, "price", None),
                "comment": getattr(res, "comment", ""),
                "normalized_volume": norm_vol,
                "fill_mode": fill,
            }
            ok_codes = [getattr(self.mt5, "TRADE_RETCODE_DONE", 10009),
                        getattr(self.mt5, "TRADE_RETCODE_PLACED", 10008)]
            if last["retcode"] in ok_codes or last["order"] or last["deal"]:
                return last
            if "Unsupported filling mode" in (last["comment"] or ""):
                continue
        return last or {"comment": "order_send failed"}

    # ---------- constants for callers ----------
    def constants(self) -> Dict[str, int]:
        m = self.mt5
        if not m:
            return {}
        out = {}
        for k in ("ORDER_FILLING_FOK","ORDER_FILLING_IOC","ORDER_FILLING_RETURN",
                  "TRADE_RETCODE_DONE","TRADE_RETCODE_PLACED",
                  "ORDER_TYPE_BUY","ORDER_TYPE_SELL",
                  "TRADE_ACTION_DEAL","ORDER_TIME_GTC"):
            out[k] = getattr(m, k, None)
        return out
