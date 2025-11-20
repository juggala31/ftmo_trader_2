from __future__ import annotations
import datetime as dt
from typing import Dict, List, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

class RoutesEngine:
    def __init__(self, routes_cfg: dict, timezone_name: str):
        self.routes_cfg = (routes_cfg or {}).get("routes", {})
        self.tzname = timezone_name
        self.tz = ZoneInfo(timezone_name) if ZoneInfo else None
        # key: (day_key, symbol, window_idx) -> used count
        self._used: Dict[Tuple[str,str,int], int] = {}
        self._last_day = self._day_key()

    def _now(self) -> dt.datetime:
        return dt.datetime.now(self.tz) if self.tz else dt.datetime.now()

    def _day_key(self) -> str:
        n = self._now()
        return f"{n.year:04d}-{n.month:02d}-{n.day:02d}"

    def _maybe_roll(self):
        dk = self._day_key()
        if dk != self._last_day:
            self._used.clear()
            self._last_day = dk

    def _parse_windows(self, arr: List[str]) -> List[Tuple[dt.time, dt.time]]:
        out = []
        for w in arr or []:
            try:
                a,b = [x.strip() for x in w.split("-")]
                t1 = dt.datetime.strptime(a, "%H:%M").time()
                t2 = dt.datetime.strptime(b, "%H:%M").time()
                out.append((t1,t2))
            except Exception:
                pass
        return out

    def current_window_index(self, now: dt.datetime, windows: List[Tuple[dt.time,dt.time]]) -> int:
        for i,(t1,t2) in enumerate(windows):
            if (now.time() >= t1) and (now.time() <= t2):
                return i
        return -1

    def check(self, symbol: str) -> Tuple[bool, str, Dict]:
        self._maybe_roll()
        cfg = self.routes_cfg.get(symbol)
        if not cfg:
            return True, "no-routes (allow)", {"window": None, "used": 0, "cap": None, "interval_sec": None}
        windows = self._parse_windows(cfg.get("windows", []))
        now = self._now()
        idx = self.current_window_index(now, windows)
        if idx < 0:
            return False, "outside window", {"window": None, "used": None, "cap": None, "interval_sec": cfg.get("interval_sec")}
        dk = self._day_key()
        cap = int(cfg.get("max_trades_per_window", 999))
        used = self._used.get((dk, symbol, idx), 0)
        if used >= cap:
            return False, f"window quota reached ({used}/{cap})", {"window": windows[idx], "used": used, "cap": cap, "interval_sec": cfg.get("interval_sec")}
        return True, "ok", {"window": windows[idx], "used": used, "cap": cap, "interval_sec": cfg.get("interval_sec")}

    def consume(self, symbol: str) -> None:
        dk = self._day_key()
        cfg = self.routes_cfg.get(symbol)
        if not cfg:
            return
        windows = self._parse_windows(cfg.get("windows", []))
        idx = self.current_window_index(self._now(), windows)
        if idx < 0:
            return
        key = (dk, symbol, idx)
        self._used[key] = self._used.get(key, 0) + 1