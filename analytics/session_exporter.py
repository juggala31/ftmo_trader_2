from __future__ import annotations
import threading, time, pathlib, json, datetime as dt
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "ai" / "summaries"

class _SessionStore:
    def __init__(self):
        self.lock = threading.Lock()
        # ticket -> position info
        self.open: Dict[int, Dict[str, Any]] = {}
        self.closed: List[Dict[str, Any]] = []
        self.started_ts = time.time()

    def on_open(self, ticket: int, symbol: str, side: str, volume: float, entry: float, point: float, vpp: float):
        with self.lock:
            self.open[ticket] = {
                "ticket": ticket, "symbol": symbol, "side": side, "volume": float(volume),
                "entry": float(entry), "point": float(point), "vpp": float(vpp),
                "t_open": time.time()
            }

    def on_close(self, ticket: int, close_px: float):
        with self.lock:
            pos = self.open.pop(ticket, None)
            if not pos:
                return
            side = pos["side"]; entry = pos["entry"]; vol = pos["volume"]
            point = max(1e-12, pos["point"]); vpp = pos["vpp"]
            pnl_quote = (close_px - entry) if side == "buy" else (entry - close_px)
            pnl_points = pnl_quote / point
            pnl_ccy = pnl_points * vpp * vol
            pos.update({"exit": float(close_px), "t_close": time.time(), "pnl_ccy": float(pnl_ccy)})
            self.closed.append(pos)

    def snapshot_open(self):
        with self.lock:
            return {k: dict(v) for k, v in self.open.items()}

    def snapshot_closed(self):
        with self.lock:
            return list(self.closed)

STORE = _SessionStore()

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

def wrap_adapter_for_session(adapter, settings):
    """
    Wrap adapter.place_order / close_order to capture open/close events.
    Must be called once after adapter is constructed.
    """
    base_place = adapter.place_order
    base_close = adapter.close_order if hasattr(adapter, "close_order") else None

    def placed(side: str, symbol: str, volume: float, sl=None, tp=None, comment: str=""):
        # call through
        res = base_place(side=side, symbol=symbol, volume=volume, sl=sl, tp=tp, comment=comment)
        # when ok, record entry from current tick
        if res and res.get("ok"):
            t = adapter.get_tick(symbol)
            if t:
                entry = float(t["ask"]) if side.lower()=="buy" else float(t["bid"])
            else:
                entry = float(res.get("price", 0.0) or 0.0)
            ticket = int(res.get("ticket") or 0) or int(time.time()*1000)  # fallback
            STORE.on_open(ticket=ticket, symbol=symbol, side=side.lower(), volume=float(volume),
                          entry=float(entry), point=_point_size(settings, symbol), vpp=_vpp(settings, symbol))
        return res

    def closed(ticket: int):
        # get mark and pass through
        sym = None
        # try to find from store
        snap = STORE.snapshot_open()
        if ticket in snap:
            sym = snap[ticket]["symbol"]
        # try to read tick
        close_px = None
        if sym:
            t = adapter.get_tick(sym)
            if t:
                close_px = float(t["bid"] if snap[ticket]["side"]=="buy" else t["ask"])
        res = base_close(ticket) if base_close else {"ok": False, "message": "close not supported"}
        if close_px is None:
            # try to infer from adapter (paper mode) or res
            close_px = float(res.get("close") or res.get("price") or 0.0)
        if ticket in STORE.open:
            STORE.on_close(ticket, close_px or 0.0)
        return res

    adapter.place_order = placed  # type: ignore
    if base_close:
        adapter.close_order = closed  # type: ignore

def _compute_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trades)
    wins = sum(1 for t in trades if t.get("pnl_ccy", 0.0) >= 0)
    losses = n - wins
    total = sum(float(t.get("pnl_ccy", 0.0)) for t in trades)
    avg = (total / n) if n else 0.0
    # equity curve for DD
    eq = 0.0; peak = 0.0; dd = 0.0
    for t in trades:
        eq += float(t.get("pnl_ccy", 0.0))
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return {"n": n, "wins": wins, "losses": losses, "winrate": (wins/n*100.0) if n else 0.0,
            "total_ccy": total, "avg_ccy": avg, "max_dd_ccy": dd}

def export_session(out_dir: Optional[str]=None) -> Dict[str, Any]:
    """
    Writes two files named with current local date:
      - CSV trades      -> ai/summaries/trades_YYYY-MM-DD.csv
      - Parquet (if available) -> ai/summaries/trades_YYYY-MM-DD.parquet
    Also prints/returns a JSON summary with per-symbol stats.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_dir:
        outp = pathlib.Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
    else:
        outp = OUT_DIR

    today = dt.datetime.now().strftime("%Y-%m-%d")

    closed = STORE.snapshot_closed()
    if not closed:
        # still write empty header for convenience
        df = pd.DataFrame(columns=["ticket","symbol","side","volume","entry","exit","pnl_ccy","t_open","t_close"])
        csv_path = outp / f"trades_{today}.csv"
        df.to_csv(csv_path, index=False)
        return {"summary": {}, "rows": 0, "csv": str(csv_path)}

    # normalize dicts
    rows = []
    for t in closed:
        rows.append({
            "ticket": t.get("ticket"),
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "volume": float(t.get("volume", 0.0)),
            "entry": float(t.get("entry", 0.0)),
            "exit":  float(t.get("exit", 0.0)),
            "pnl_ccy": float(t.get("pnl_ccy", 0.0)),
            "t_open": float(t.get("t_open", 0.0)),
            "t_close": float(t.get("t_close", 0.0)),
        })
    df = pd.DataFrame(rows)

    # per-symbol stats
    summary: Dict[str, Any] = {}
    for sym, g in df.groupby("symbol"):
        summary[sym] = _compute_stats(g.to_dict("records"))
    summary["_ALL_"] = _compute_stats(df.to_dict("records"))

    # write CSV
    csv_path = outp / f"trades_{today}.csv"
    df.to_csv(csv_path, index=False)

    # write Parquet if available
    pq_path = None
    if HAVE_PARQUET:
        pq_path = outp / f"trades_{today}.parquet"
        df.to_parquet(pq_path, index=False)

    return {"summary": summary, "rows": int(len(df)), "csv": str(csv_path), "parquet": (str(pq_path) if pq_path else "")}