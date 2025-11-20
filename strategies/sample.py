from __future__ import annotations
import time
from typing import List, Dict, Any

class SampleStrategy:
    """
    Emits a small test order every `interval_sec` per symbol (buy/sell alternating).
    Use APP_MODE=dry or sim first.
    """
    def __init__(self, symbols: list[str], volume: float=0.10, interval_sec: float=10.0, autoclose_ms: int=3000):
        self.symbols = symbols
        self.volume = float(volume)
        self.interval = float(interval_sec)
        self.autoclose_ms = int(autoclose_ms)
        self._last_ts = 0.0
        self._flip = False

    def on_tick(self, tick: dict) -> None:
        pass

    def signals(self) -> List[Dict[str, Any]]:
        now = time.time()
        if now - self._last_ts < self.interval:
            return []
        self._last_ts = now
        self._flip = not self._flip
        side = "buy" if self._flip else "sell"
        # emit one order for the first symbol only to keep things gentle
        sym = self.symbols[0] if self.symbols else "XAUUSD"
        return [{
            "side": side,
            "symbol": sym,
            "volume": self.volume,
            "comment": f"sample-{side}",
            "autoclose_ms": self.autoclose_ms
        }]