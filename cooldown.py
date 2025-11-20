from __future__ import annotations
import time
from typing import Tuple

class Cooldown:
    def __init__(self, cfg: dict):
        c = (cfg or {}).get("cooldown", {})
        self.enabled = bool(c.get("enabled", True))
        self.loss_trigger = float(c.get("loss_trigger_ccy", 200.0))
        self.minutes = float(c.get("minutes", 10))
        self._until_ts: float = 0.0

    def on_realized(self, delta_ccy: float):
        if not self.enabled:
            return
        if delta_ccy <= -abs(self.loss_trigger):
            self._until_ts = time.time() + (self.minutes * 60.0)

    def status(self) -> Tuple[bool, float]:
        if not self.enabled:
            return (False, 0.0)
        now = time.time()
        if now < self._until_ts:
            return (True, max(0.0, self._until_ts - now))
        return (False, 0.0)