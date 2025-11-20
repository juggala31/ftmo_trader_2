from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import datetime as _dt

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

@dataclass
class RiskState:
    paused: bool = False
    reason: str = ""

class RiskGuard:
    def __init__(self, cfg: dict, corr_map: Dict[str, str] | None = None,
                 initial_today: float = 0.0, initial_history: List[float] | None = None):
        self.cfg = cfg or {}
        self.state = RiskState(paused=False, reason="")
        limits = (self.cfg.get("limits") or {})
        ops = (self.cfg.get("ops") or {})
        risk = (self.cfg.get("risk") or {})
        self.max_daily_loss = float(limits.get("max_daily_loss", -5000.0))
        self.max_total_dd   = float(limits.get("max_total_drawdown", -10000.0))
        self.pause_on_limit = bool(ops.get("pause_on_limit_breach", True))
        self.timezone_name  = str(ops.get("timezone", "America/New_York"))
        self.max_corr       = int(risk.get("max_correlated_positions", 3))
        self.corr_map       = corr_map or {}

        # account-currency tracking
        self.realized_today = float(initial_today or 0.0)
        self.unrealized_now = 0.0
        self.total_pnl      = 0.0  # across entire run; seed at 0 (not persisted)

        # daily reset
        self._tz = ZoneInfo(self.timezone_name) if ZoneInfo else None
        self._day_key = self._current_day_key()
        self._daily_history: List[float] = list(initial_history or [])

    # --- time helpers
    def _now(self) -> _dt.datetime:
        if self._tz:
            return _dt.datetime.now(self._tz)
        return _dt.datetime.now()

    def _current_day_key(self) -> str:
        n = self._now()
        return f"{n.year:04d}-{n.month:02d}-{n.day:02d}"

    def day_key(self) -> str:
        return self._day_key

    def daily_history(self) -> List[float]:
        return list(self._daily_history)

    def _check_daily_rollover(self):
        k = self._current_day_key()
        if k != self._day_key:
            self._daily_history.append(self.realized_today)
            self.realized_today = 0.0
            self._day_key = k

    # --- correlated exposure
    def _group_of(self, symbol: str) -> str:
        return self.corr_map.get(symbol, symbol)

    def _violates_corr_cap(self, new_symbol: str, open_positions: list[dict]) -> tuple[bool, str]:
        if self.max_corr <= 0:
            return (False, "")
        new_group = self._group_of(new_symbol)
        count = 0
        for p in open_positions or []:
            sym = p.get("symbol")
            if not sym:
                continue
            if self._group_of(sym) == new_group:
                count += 1
        if count >= self.max_corr:
            return (True, f"Correlated exposure cap reached for group {new_group} (cap={self.max_corr})")
        return (False, "")

    # --- API
    def pause(self, why: str): self.state = RiskState(True, why)
    def resume(self): self.state = RiskState(False, "")

    def pre_trade_check(self, order_ctx: Dict[str, Any]) -> Tuple[bool, str]:
        self._check_daily_rollover()
        if self.state.paused:
            return False, f"Trading paused: {self.state.reason}"
        daily_pnl = self.realized_today + self.unrealized_now
        if daily_pnl <= self.max_daily_loss:
            if self.pause_on_limit:
                self.pause("Daily loss limit reached")
            return False, "Daily loss limit reached"
        if self.total_pnl <= self.max_total_dd:
            if self.pause_on_limit:
                self.pause("Max total drawdown reached")
            return False, "Max total drawdown reached"
        symbol = str(order_ctx.get("symbol", ""))
        vio, msg = self._violates_corr_cap(symbol, order_ctx.get("open_positions") or [])
        if vio:
            return False, msg
        return True, "OK"

    def post_fill_update(self, pnl_ctx: Dict[str, Any]) -> None:
        self._check_daily_rollover()
        realized_delta = float(pnl_ctx.get("realized_delta_ccy", pnl_ctx.get("realized_delta", 0.0)))
        unreal_total   = float(pnl_ctx.get("unrealized_total_ccy", pnl_ctx.get("unrealized_total", self.unrealized_now)))
        self.realized_today += realized_delta
        self.unrealized_now  = unreal_total
        self.total_pnl       += realized_delta

    def snapshot(self) -> dict:
        return {
            "paused": self.state.paused,
            "reason": self.state.reason,
            "daily_pnl": round(self.realized_today + self.unrealized_now, 2),
            "total_pnl": round(self.total_pnl, 2),
            "max_daily_loss": self.max_daily_loss,
            "max_total_drawdown": self.max_total_dd,
            "day": self._day_key,
            "daily_avg": round((sum(self._daily_history[-5:]) / len(self._daily_history[-5:])) if self._daily_history[-5:] else 0.0, 2),
        }