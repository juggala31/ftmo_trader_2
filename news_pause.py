from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

@dataclass
class NewsEvent:
    name: str
    when: dt.datetime
    before_min: int
    after_min: int

class NewsPause:
    def __init__(self, cfg: dict, timezone_name: str):
        self.enabled = bool(cfg.get("enabled", False))
        self.tzname = timezone_name
        self.tz = ZoneInfo(timezone_name) if ZoneInfo else None
        self.events: List[NewsEvent] = []
        for ev in (cfg.get("events") or []):
            name = str(ev.get("name", "event"))
            when_s = str(ev.get("when", ""))
            try:
                # Parse as local time (no timezone in string)
                y,m,d_h = when_s.split(" ", 1)
                y = int(y.split("-")[0])  # not used but keeps the parse strict-ish
                dt_local = dt.datetime.strptime(when_s, "%Y-%m-%d %H:%M")
                if self.tz:
                    dt_local = dt_local.replace(tzinfo=self.tz)
                self.events.append(NewsEvent(
                    name=name,
                    when=dt_local,
                    before_min=int(ev.get("before_min", 0)),
                    after_min=int(ev.get("after_min", 0))
                ))
            except Exception:
                # ignore invalid rows
                pass

    def now(self) -> dt.datetime:
        if self.tz:
            return dt.datetime.now(self.tz)
        return dt.datetime.now()

    def is_pause_window(self) -> tuple[bool, str]:
        if not self.enabled:
            return (False, "")
        now = self.now()
        for ev in self.events:
            start = ev.when - dt.timedelta(minutes=ev.before_min)
            end   = ev.when + dt.timedelta(minutes=ev.after_min)
            if start <= now <= end:
                return (True, f"news-pause: {ev.name}")
        return (False, "")