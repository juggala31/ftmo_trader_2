from __future__ import annotations
from collections import deque
from typing import Deque, Optional

class RollingATR:
    """
    Classic ATR on synthetic candles built from ticks:
      - we keep a rolling "mid" price and derive high/low with a small buffer
      - simple but works for sizing presets in DRY/SIM
    """
    def __init__(self, period: int = 14):
        self.period = max(2, int(period))
        self._prev_close: Optional[float] = None
        self._tr: Deque[float] = deque(maxlen=self.period)

    def update_with_mid(self, mid: float, band: float = 0.0) -> float:
        hi = mid + band
        lo = mid - band
        return self.update_candle(hi, lo, mid)

    def update_candle(self, high: float, low: float, close: float) -> float:
        if self._prev_close is None:
            tr = float(high - low)
        else:
            tr = max(
                float(high - low),
                abs(float(high - self._prev_close)),
                abs(float(low - self._prev_close))
            )
        self._tr.append(tr)
        self._prev_close = float(close)
        if len(self._tr) < self.period:
            return sum(self._tr) / len(self._tr)
        return sum(self._tr) / self.period