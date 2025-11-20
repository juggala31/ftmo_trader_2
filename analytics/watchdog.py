from __future__ import annotations
import threading, time
from typing import List, Callable, Optional

def start_watchdog(adapter, symbols: List[str], stale_sec: int = 20, on_stale: Optional[Callable]=None):
    """
    Background thread: if no fresh tick for any watched symbol for stale_sec,
    attempts adapter.reconnect() if available; else no-op. Calls on_stale once.
    """
    stop = threading.Event()
    def worker():
        last_alert = 0.0
        while not stop.is_set():
            try:
                now = time.time()
                stale = []
                for s in symbols:
                    t = adapter.get_tick(s) or {}
                    ts = float(t.get("ts", 0.0) or 0.0)
                    if ts and now - ts > stale_sec:
                        stale.append(s)
                if stale and (now - last_alert > stale_sec):
                    last_alert = now
                    if hasattr(adapter, "reconnect"):
                        try: adapter.reconnect()
                        except Exception: pass
                    if on_stale:
                        try: on_stale(stale)
                        except Exception: pass
            except Exception:
                pass
            time.sleep(2.0)
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return stop