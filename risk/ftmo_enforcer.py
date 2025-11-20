# -*- coding: utf-8 -*-
"""
risk/ftmo_enforcer.py

Minimal FTMO-style guard:
- Tracks daily loss based on start-of-day equity
- Tracks running max drawdown based on peak equity
- Persists state to JSON so restarts keep context
- Provides status_text() for UI/console
- Sets .violated True and a reason if a rule is broken

No 3rd-party deps.
"""

from __future__ import annotations
import os, json, datetime as _dt
from typing import Optional, Dict, Any

def _today_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d")

def _ensure_dir(p: str):
    try:
        d = os.path.dirname(os.path.abspath(p))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass

class FTMOEnforcer:
    def __init__(self, daily_loss_limit: float, max_dd_limit: float, state_path: str):
        self.daily_loss_limit = float(daily_loss_limit or 0.0)
        self.max_dd_limit     = float(max_dd_limit or 0.0)
        self.state_path       = state_path

        # runtime
        self.violated: bool   = False
        self.violation: str   = ""

        # rolling telemetry (latest)
        self.pnl_today: float = 0.0
        self.equity_now: float = 0.0
        self.balance_now: float = 0.0

        # persisted state
        self.day: str = _today_str()
        self.start_equity: Optional[float] = None   # equity at start of day
        self.peak_equity: Optional[float]  = None   # highest equity ever seen (for max DD)

        self._load_state()

    # ---------- persistence ----------
    def _load_state(self):
        try:
            if os.path.isfile(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.day          = data.get("day", self.day)
                self.start_equity = data.get("start_equity")
                self.peak_equity  = data.get("peak_equity")
        except Exception:
            # safe defaults on error
            self.day = _today_str()
            self.start_equity = None
            self.peak_equity = None

    def _save_state(self):
        try:
            _ensure_dir(self.state_path)
            data = {
                "day": self.day,
                "start_equity": self.start_equity,
                "peak_equity": self.peak_equity,
            }
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # ---------- lifecycle ----------
    def reset_if_new_day(self):
        today = _today_str()
        if self.day != today:
            self.day = today
            # On a fresh day we will lazily set start_equity at the first update() call
            self.start_equity = None
            # peak_equity persists across days for overall max DD
            self._save_state()

    # ---------- core update ----------
    def update(self, pnl_today: float, equity_now: float, balance_now: float):
        """
        Feed current account telemetry each heartbeat.
        """
        self.pnl_today   = float(pnl_today or 0.0)
        self.equity_now  = float(equity_now or 0.0)
        self.balance_now = float(balance_now or 0.0)

        # Initialize start-of-day equity if needed
        if self.start_equity is None and self.equity_now > 0:
            self.start_equity = self.equity_now

        # Track peak equity (for total DD)
        if self.peak_equity is None or (self.equity_now > self.peak_equity):
            self.peak_equity = self.equity_now

        # Compute day DD and total DD
        day_dd   = 0.0
        total_dd = 0.0
        if self.start_equity is not None and self.equity_now > 0:
            day_dd = max(0.0, self.start_equity - self.equity_now)
        if self.peak_equity is not None and self.equity_now > 0:
            total_dd = max(0.0, self.peak_equity - self.equity_now)

        # Check limits
        self.violated = False
        self.violation = ""
        if self.daily_loss_limit > 0 and day_dd >= self.daily_loss_limit:
            self.violated = True
            self.violation = f"Daily loss limit hit: {day_dd:.2f} ≥ {self.daily_loss_limit:.2f}"
        if not self.violated and self.max_dd_limit > 0 and total_dd >= self.max_dd_limit:
            self.violated = True
            self.violation = f"Max drawdown limit hit: {total_dd:.2f} ≥ {self.max_dd_limit:.2f}"

        self._save_state()

    # ---------- presentation ----------
    def status_text(self) -> str:
        # Friendly single-line status for console/UI
        day_dd = 0.0
        total_dd = 0.0
        if self.start_equity is not None and self.equity_now > 0:
            day_dd = max(0.0, self.start_equity - self.equity_now)
        if self.peak_equity is not None and self.equity_now > 0:
            total_dd = max(0.0, self.peak_equity - self.equity_now)

        parts = [
            f"[FTMO] pnl_today={self.pnl_today:.2f}",
            f"day_dd={day_dd:.2f}" + (f"/{self.daily_loss_limit:.0f}" if self.daily_loss_limit > 0 else ""),
            f"max_dd={total_dd:.2f}" + (f"/{self.max_dd_limit:.0f}" if self.max_dd_limit > 0 else ""),
        ]
        if self.violated:
            parts.append(f"VIOLATION: {self.violation}")
        return " ".join(parts)
