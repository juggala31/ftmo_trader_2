#!/usr/bin/env python
"""
Phase Manager — thresholds, progress, rolling metrics, persistence.
Standalone, importable, and safe to run without your live system.
"""

from __future__ import annotations
import json, math, os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

DATA_DIR = Path(r"C:\ftmo_trader_2\data")
STATE_PATH = DATA_DIR / "phase_state.json"

# Defaults (override later via .env or controller if you want)
DEFAULT_PHASE_TARGETS = [75, 150]     # P1 end @75, P2 end @150, P3 = 150+
DEFAULT_MIN_CONF      = [0.01, 0.05, 0.10]
DEFAULT_COOLDOWN_S    = [60, 90, 120]
DEFAULT_MAX_POS       = [5, 4, 3]
ROLLING_WINDOW        = 50            # recent trades window for WR/PF

@dataclass
class TradeRecord:
    pnl: float
    # You can extend with fields you care about (symbol, side, ts, etc.)

@dataclass
class PhaseConfig:
    targets: List[int] = field(default_factory=lambda: list(DEFAULT_PHASE_TARGETS))
    min_conf: List[float] = field(default_factory=lambda: list(DEFAULT_MIN_CONF))
    cooldown_s: List[int] = field(default_factory=lambda: list(DEFAULT_COOLDOWN_S))
    max_positions: List[int] = field(default_factory=lambda: list(DEFAULT_MAX_POS))

@dataclass
class PhaseState:
    total_trades: int = 0
    trade_history: List[TradeRecord] = field(default_factory=list)  # recent only (bounded)
    # Derived metrics (cached on save for easy UI reads)
    phase_index: int = 0
    phase_name: str = "Phase 1 — Data Collection"
    progress_pct: float = 0.0
    trades_remaining: int = 75
    wr_recent: float = 0.0
    pf_recent: float = 0.0
    # Active thresholds (min-conf, cooldown, max-pos) — resolved for current phase
    active_min_conf: float = DEFAULT_MIN_CONF[0]
    active_cooldown_s: int = DEFAULT_COOLDOWN_S[0]
    active_max_positions: int = DEFAULT_MAX_POS[0]

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trade_history"] = [asdict(t) for t in self.trade_history]
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "PhaseState":
        ph = PhaseState()
        ph.total_trades = int(d.get("total_trades", 0))
        ph.trade_history = [TradeRecord(**t) for t in d.get("trade_history", [])]
        ph.phase_index = int(d.get("phase_index", 0))
        ph.phase_name = d.get("phase_name", "Phase 1 — Data Collection")
        ph.progress_pct = float(d.get("progress_pct", 0.0))
        ph.trades_remaining = int(d.get("trades_remaining", 75))
        ph.wr_recent = float(d.get("wr_recent", 0.0))
        ph.pf_recent = float(d.get("pf_recent", 0.0))
        ph.active_min_conf = float(d.get("active_min_conf", DEFAULT_MIN_CONF[0]))
        ph.active_cooldown_s = int(d.get("active_cooldown_s", DEFAULT_COOLDOWN_S[0]))
        ph.active_max_positions = int(d.get("active_max_positions", DEFAULT_MAX_POS[0]))
        return ph

PHASE_NAMES = [
    "Phase 1 — Data Collection",
    "Phase 2 — Quality Improvement",
    "Phase 3 — Performance Focus",
]

class PhaseManager:
    def __init__(self, cfg: Optional[PhaseConfig] = None):
        self.cfg = cfg or PhaseConfig()
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    # ---------- Persistence ----------
    def _load(self) -> PhaseState:
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                st = PhaseState.from_json(data)
                self._recompute(st)  # ensure derived fields up-to-date
                return st
            except Exception:
                pass
        st = PhaseState()
        self._recompute(st)
        self._save(st)
        return st

    def _save(self, st: PhaseState) -> None:
        STATE_PATH.write_text(json.dumps(st.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Core API ----------
    def add_trade(self, pnl: float) -> PhaseState:
        st = self.state
        st.total_trades += 1
        st.trade_history.append(TradeRecord(pnl=pnl))
        # Bound the rolling window
        if len(st.trade_history) > max(ROLLING_WINDOW, 1):
            st.trade_history = st.trade_history[-ROLLING_WINDOW:]
        self._recompute(st)
        self._save(st)
        return st

    def reset(self) -> PhaseState:
        st = PhaseState()
        self._recompute(st)
        self._save(st)
        return st

    def set_targets(self, phase_targets: List[int]) -> PhaseState:
        if len(phase_targets) < 1:
            return self.state
        self.cfg.targets = list(phase_targets)
        st = self.state
        self._recompute(st)
        self._save(st)
        return st

    def snapshot(self) -> PhaseState:
        # Returns a fresh computed state without mutating history
        st = self.state
        self._recompute(st)
        self._save(st)
        return st

    # ---------- Internals ----------
    def _recompute(self, st: PhaseState) -> None:
        # Determine phase index based on total_trades and targets
        t = st.total_trades
        t1 = self.cfg.targets[0] if len(self.cfg.targets) >= 1 else 75
        t2 = self.cfg.targets[1] if len(self.cfg.targets) >= 2 else 150

        if t < t1:
            idx = 0
            target = t1
            done = t
        elif t < t2:
            idx = 1
            target = t2
            done = t - t1
            denom = max(t2 - t1, 1)
            st.progress_pct = round(100.0 * done / denom, 2)
        else:
            idx = 2
            target = t2
            done = t2
            st.progress_pct = 100.0

        # For Phase 1, compute progress within [0, t1]
        if idx == 0:
            denom = max(t1, 1)
            st.progress_pct = round(100.0 * t / denom, 2)
        # For Phase 3, progress is pinned to 100

        st.phase_index = idx
        st.phase_name = PHASE_NAMES[idx]
        st.trades_remaining = max(target - t, 0)

        # Rolling metrics (WR, PF) using recent window
        wins, losses = 0, 0
        sum_win, sum_loss = 0.0, 0.0
        for tr in st.trade_history:
            if tr.pnl >= 0:
                wins += 1
                sum_win += tr.pnl
            else:
                losses += 1
                sum_loss += abs(tr.pnl)
        total = wins + losses
        st.wr_recent = round(100.0 * wins / total, 2) if total else 0.0
        st.pf_recent = round((sum_win / sum_loss), 2) if sum_loss > 1e-12 else (float('inf') if sum_win > 0 else 0.0)

        # Activate thresholds for the current phase
        st.active_min_conf   = self._pick(self.cfg.min_conf, idx, DEFAULT_MIN_CONF[idx])
        st.active_cooldown_s = int(self._pick(self.cfg.cooldown_s, idx, DEFAULT_COOLDOWN_S[idx]))
        st.active_max_positions = int(self._pick(self.cfg.max_positions, idx, DEFAULT_MAX_POS[idx]))

    @staticmethod
    def _pick(arr: List, idx: int, default_val):
        try:
            return arr[idx]
        except Exception:
            return default_val

# Convenience function for other modules
def get_phase_snapshot() -> Dict[str, Any]:
    pm = PhaseManager()
    st = pm.snapshot()
    return {
        "phase_index": st.phase_index,
        "phase_name": st.phase_name,
        "total_trades": st.total_trades,
        "progress_pct": st.progress_pct,
        "trades_remaining": st.trades_remaining,
        "wr_recent": st.wr_recent,
        "pf_recent": st.pf_recent,
        "min_conf": st.active_min_conf,
        "cooldown_s": st.active_cooldown_s,
        "max_positions": st.active_max_positions,
    }

if __name__ == "__main__":
    pm = PhaseManager()
    snap = get_phase_snapshot()
    print(json.dumps(snap, indent=2))