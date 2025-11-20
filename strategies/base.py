from __future__ import annotations
from typing import List, Dict, Any

class Strategy:
    """
    Base strategy interface. Phase B is a no-op that produces no signals.
    Implementors should return a list of order dicts:
      {"side": "buy"|"sell", "symbol": "XAUUSD", "volume": 0.1, "sl": None, "tp": None, "comment": "reason"}
    """
    def on_tick(self, tick: dict) -> None:
        pass

    def signals(self) -> List[Dict[str, Any]]:
        return []