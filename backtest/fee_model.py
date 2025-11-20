#!/usr/bin/env python
DEFAULT_POINT_VALUES = {
    "XAUUSD":   1.0,
    "US30":     1.0, "US30USD": 1.0,
    "US100":    1.0, "NAS100USD": 1.0,
    "SPX500":   1.0, "SPX500USD": 1.0,
    "WTICOUSD": 1.0,
    "BTCUSD":   1.0,
}

def point_value_for(symbol: str, override: float|None, table: dict|None):
    if override and override > 0: return float(override)
    if table and symbol in table:
        try: return float(table[symbol])
        except Exception: pass
    return DEFAULT_POINT_VALUES.get(symbol, 1.0)

def apply_costs(symbol, positions, commission_per_side=0.0, spread_points=0.0, slippage_points=0.0,
                point_value_override=None, point_value_table=None):
    pv = point_value_for(symbol, point_value_override, point_value_table)
    for p in positions:
        if p.closed_at is None: continue
        comm = float(commission_per_side) * 2.0          # open + close
        spread_cost = abs(float(spread_points)) * pv      # total spread in points
        slip_cost   = abs(float(slippage_points)) * 2.0 * pv  # per-side slippage
        p.pnl -= (comm + spread_cost + slip_cost)