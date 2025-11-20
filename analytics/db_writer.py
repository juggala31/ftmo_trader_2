from __future__ import annotations
import pathlib, sys, time, sqlite3
from typing import Dict, Any

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DB_PATH = _ROOT / "db" / "trades.sqlite"

def _ensure_schema():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket INTEGER,
        symbol TEXT,
        side TEXT,
        volume REAL,
        entry REAL,
        exit REAL,
        pnl_ccy REAL,
        t_open REAL,
        t_close REAL
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_tclose ON trades(t_close)")
    con.commit(); con.close()

def _point_size(settings, symbol: str) -> float:
    try:
        return float((settings.sizing.get("points") or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

def _vpp(settings, symbol: str) -> float:
    try:
        return float((settings.vpp or {}).get(symbol, 1.0))
    except Exception:
        return 1.0

def wrap_adapter_for_db(adapter, settings):
    """
    Wrap adapter.place_order and close_order to write rows into SQLite.
    """
    _ensure_schema()
    base_place = adapter.place_order
    base_close = adapter.close_order if hasattr(adapter, "close_order") else None

    # in-memory open map (ticket -> snapshot)
    OPEN: Dict[int, Dict[str, Any]] = {}

    def placed(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        res = base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)
        if res and res.get("ok"):
            t = adapter.get_tick(symbol) or {}
            entry = float(t.get("ask") if side.lower()=="buy" else t.get("bid") or res.get("price") or 0.0)
            ticket = int(res.get("ticket") or 0) or int(time.time()*1000)
            OPEN[ticket] = {
                "ticket": ticket, "symbol": symbol, "side": side.lower(),
                "volume": float(volume), "entry": float(entry),
                "t_open": time.time(),
                "point": _point_size(settings, symbol),
                "vpp": _vpp(settings, symbol),
            }
        return res

    def closed(ticket: int):
        sym = OPEN.get(ticket, {}).get("symbol")
        t = adapter.get_tick(sym) if sym else None
        close_px = None
        if t and sym:
            close_px = float(t["bid"] if OPEN[ticket]["side"]=="buy" else t["ask"])
        res = base_close(ticket) if base_close else {"ok": False}
        if close_px is None:
            close_px = float(res.get("close") or res.get("price") or 0.0)
        pos = OPEN.pop(ticket, None)
        if pos:
            pnl_quote = (close_px - pos["entry"]) if pos["side"]=="buy" else (pos["entry"] - close_px)
            pnl_points = pnl_quote / max(1e-12, pos["point"])
            pnl_ccy = pnl_points * pos["vpp"] * pos["volume"]
            row = {
                "ticket": pos["ticket"], "symbol": pos["symbol"], "side": pos["side"], "volume": pos["volume"],
                "entry": pos["entry"], "exit": float(close_px), "pnl_ccy": float(pnl_ccy),
                "t_open": pos["t_open"], "t_close": time.time(),
            }
            con = sqlite3.connect(DB_PATH); cur = con.cursor()
            cur.execute("""INSERT INTO trades(ticket,symbol,side,volume,entry,exit,pnl_ccy,t_open,t_close)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                           (row["ticket"],row["symbol"],row["side"],row["volume"],row["entry"],row["exit"],row["pnl_ccy"],row["t_open"],row["t_close"]))
            con.commit(); con.close()
        return res

    adapter.place_order = placed  # type: ignore
    if base_close:
        adapter.close_order = closed  # type: ignore