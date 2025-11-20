#!/usr/bin/env python
"""
Execution simulator:
- Position sizing (FIXED_LOT or ATR_RISK)
- SL/TP placement (ATR-based or FIXED pips)
- Order lifecycle with cooldown and max concurrent positions
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Position:
    symbol: str
    side: str         # BUY or SELL
    entry: float
    size: float
    sl: Optional[float]
    tp: Optional[float]
    opened_at: int    # bar index
    closed_at: Optional[int] = None
    exit: Optional[float] = None
    pnl: float = 0.0

    def alive(self) -> bool:
        return self.closed_at is None

@dataclass
class SimConfig:
    symbol: str
    min_conf: float
    cooldown_bars: int
    max_positions: int

    # sizing
    sizing_mode: str = "ATR_RISK"   # "ATR_RISK" or "FIXED_LOT"
    fixed_lot: float = 0.10
    atr_risk_pct: float = 0.50

    # ATR (for SL/TP and sizing)
    atr_period: int = 14
    sl_atr_mult: float = 1.00
    tp_atr_mult: float = 2.00

    # fixed SL/TP (in "points" = price units)
    use_fixed_sltp: bool = False
    fixed_sl_points: float = 100.0
    fixed_tp_mult: float = 2.0

@dataclass
class SimState:
    last_signal_bar: int = -10_000
    open_positions: List[Position] = field(default_factory=list)
    equity: float = 100_000.0

def enter_if_allowed(cfg: SimConfig, st: SimState, side: str, confidence: float, bar_idx: int, price: float, atr: float) -> Optional[Position]:
    if confidence < cfg.min_conf:
        return None
    if bar_idx - st.last_signal_bar < cfg.cooldown_bars:
        return None
    # max positions
    alive = [p for p in st.open_positions if p.alive()]
    if len(alive) >= cfg.max_positions:
        return None

    # size
    if cfg.sizing_mode.upper() == "ATR_RISK":
        # risk_pct * equity / (SL distance)
        sl_dist = (cfg.sl_atr_mult * atr) if not cfg.use_fixed_sltp else cfg.fixed_sl_points
        if sl_dist <= 0:
            return None
        dollars = (cfg.atr_risk_pct/100.0) * st.equity
        # 1 "lot" == 1 unit notionally (you can adapt to contract multipliers later)
        size = max(dollars / sl_dist, 0.0)
    else:
        size = cfg.fixed_lot

    # sl/tp
    if cfg.use_fixed_sltp:
        sl = price - cfg.fixed_sl_points if side == "BUY" else price + cfg.fixed_sl_points
        tp = price + cfg.fixed_tp_mult * cfg.fixed_sl_points if side == "BUY" else price - cfg.fixed_tp_mult * cfg.fixed_sl_points
    else:
        sl = price - cfg.sl_atr_mult * atr if side == "BUY" else price + cfg.sl_atr_mult * atr
        tp = price + cfg.tp_atr_mult * atr if side == "BUY" else price - cfg.tp_atr_mult * atr

    pos = Position(symbol=cfg.symbol, side=side, entry=price, size=size, sl=sl, tp=tp, opened_at=bar_idx)
    st.open_positions.append(pos)
    st.last_signal_bar = bar_idx
    return pos

def update_positions(st: SimState, high: float, low: float, close: float, bar_idx: int):
    for p in st.open_positions:
        if not p.alive():
            continue
        # check SL / TP intrabar
        hit_sl = (low <= p.sl <= high) if p.side == "BUY" else (low <= p.sl <= high)
        hit_tp = (low <= p.tp <= high) if p.side == "BUY" else (low <= p.tp <= high)
        # Priority: TP first then SL (or flip; configurable if desired)
        exit_price = None
        if hit_tp and hit_sl:
            # if both touched, assume worst-case: SL first for simplicity
            exit_price = p.sl
        elif hit_tp:
            exit_price = p.tp
        elif hit_sl:
            exit_price = p.sl

        if exit_price is not None:
            p.exit = exit_price
            p.closed_at = bar_idx
            p.pnl = (p.exit - p.entry) * p.size if p.side == "BUY" else (p.entry - p.exit) * p.size

def flatten_all(st: SimState, close: float, bar_idx: int):
    for p in st.open_positions:
        if p.alive():
            p.exit = close
            p.closed_at = bar_idx
            p.pnl = (p.exit - p.entry) * p.size if p.side == "BUY" else (p.entry - p.exit) * p.size

def realized_pnl(st: SimState) -> float:
    return sum(p.pnl for p in st.open_positions if not p.alive())