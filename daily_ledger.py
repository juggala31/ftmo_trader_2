from __future__ import annotations
import json, pathlib
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
LEDGER_PATH = DATA / "daily_ledger.json"

class DailyLedger:
    def __init__(self, path: pathlib.Path = LEDGER_PATH):
        self.path = path
        self.current_day: str = ""
        self.realized_today: float = 0.0
        self.history: List[float] = []  # completed days (realized totals)
        self._load()

    def _load(self):
        if not self.path.exists():
            self._save()
            return
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            self.current_day = str(obj.get("current_day", ""))
            self.realized_today = float(obj.get("realized_today", 0.0))
            self.history = [float(x) for x in (obj.get("history", []) or [])]
        except Exception:
            # start fresh if corrupted
            self.current_day = ""
            self.realized_today = 0.0
            self.history = []

    def _save(self):
        obj = {
            "current_day": self.current_day,
            "realized_today": self.realized_today,
            "history": self.history,
        }
        self.path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    def bootstrap_if_empty(self, day_key: str):
        if not self.current_day:
            self.current_day = day_key
            self._save()

    def sync_from_risk(self, day_key: str, realized_today: float, history: list[float] | None = None):
        changed = False
        if day_key != self.current_day:
            # rollover: push previous today's realized into history
            if self.current_day:
                self.history.append(self.realized_today)
            self.current_day = day_key
            self.realized_today = 0.0
            changed = True
        # keep realized_today persistent
        if self.realized_today != realized_today:
            self.realized_today = realized_today
            changed = True
        if history is not None:
            # store only last 30 for compactness
            trimmed = list(history)[-30:]
            if trimmed != self.history:
                self.history = trimmed
                changed = True
        if changed:
            self._save()