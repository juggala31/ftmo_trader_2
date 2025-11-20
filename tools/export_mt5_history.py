import os, sys, csv, pathlib, datetime as dt
from typing import Dict, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import MetaTrader5 as mt5

TF_NAMES = ["M1","M5","M15","M30","H1","H4"]
TF_MAP: Dict[str,int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
}

def parse_date(s: str) -> dt.datetime:
    y,m,d = map(int, s.split("-"))
    return dt.datetime(y,m,d)

def ensure_dir(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _coerce_login(val: Optional[str]) -> Optional[int]:
    if not val:
        return None
    try:
        return int(val.strip())
    except Exception:
        return None

def init_mt5() -> bool:
    """Try explicit creds (with int login), then clean fallback to default initialize()."""
    login  = _coerce_login(os.getenv("MT5_LOGIN"))
    pwd    = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    # 1) Try explicit creds if we have all three
    if login is not None and pwd and server:
        if mt5.initialize(login=login, password=pwd, server=server):
            return True
        err1 = mt5.last_error()
        sys.stderr.write(f"[export] MT5 initialize(login) failed: {err1}\n")
        try:
            mt5.shutdown()
        except Exception:
            pass

    # 2) Fallback: try a clean default initialize (uses terminal's saved account/session)
    if mt5.initialize():
        return True
    err2 = mt5.last_error()
    sys.stderr.write(f"[export] MT5 initialize() failed: {err2}\n")
    return False

def export_csv(symbol: str, tf_name: str, d_from: dt.datetime, d_to: dt.datetime, out_csv: pathlib.Path):
    tf_const = TF_MAP[tf_name]
    rates = mt5.copy_rates_range(symbol, tf_const, d_from, d_to)
    if rates is None or len(rates) == 0:
        # fallback: copy a large chunk then filter
        minutes_per_bar = {"M1":1, "M5":5, "M15":15, "M30":30, "H1":60, "H4":240}[tf_name]
        approx_count = int((5*365*24*60)/minutes_per_bar) + 2000
        rates = mt5.copy_rates_from(symbol, tf_const, d_from, approx_count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data for {symbol} {tf_name} in {d_from.date()}..{d_to.date()}")
        t0, t1 = d_from.timestamp(), d_to.timestamp()
        mask = (rates["time"] >= t0) & (rates["time"] <= t1)
        rates = rates[mask]
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No in-window data for {symbol} {tf_name} in {d_from.date()}..{d_to.date()}")

    ensure_dir(out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time","open","high","low","close","tick_volume","spread","real_volume"])
        for row in rates:
            t = dt.datetime.utcfromtimestamp(int(row["time"])).strftime("%Y-%m-%d %H:%M:%S")
            w.writerow([
                t,
                f"{float(row['open']):.6f}",
                f"{float(row['high']):.6f}",
                f"{float(row['low']):.6f}",
                f"{float(row['close']):.6f}",
                int(row["tick_volume"]) if "tick_volume" in rates.dtype.names else "",
                int(row["spread"]) if "spread" in rates.dtype.names else "",
                int(row["real_volume"]) if "real_volume" in rates.dtype.names else "",
            ])

def main():
    symbols     = os.getenv("EXPORT_SYMBOLS", "XAUUSD,US30USD,NAS100USD,SPX500USD,WTICOUSD,BTCUSD").split(",")
    timeframes  = os.getenv("EXPORT_TIMEFRAMES", "M1,M5,M15,M30,H1,H4").split(",")
    d_from      = parse_date(os.getenv("EXPORT_FROM", "2020-11-10"))
    d_to        = parse_date(os.getenv("EXPORT_TO",   "2025-11-10"))
    out_root    = pathlib.Path(os.getenv("EXPORT_OUT", str(ROOT / "data" / "export")))

    if not init_mt5():
        sys.exit(2)

    print(f"[export] symbols={symbols}  tfs={timeframes}  from={d_from.date()}  to={d_to.date()}  out={out_root}")

    errors = 0
    for s in symbols:
        s = s.strip()
        for tf in [t.strip().upper() for t in timeframes]:
            if tf not in TF_MAP:
                sys.stderr.write(f"[export] skip unknown TF {tf}\n")
                continue
            out_csv = out_root / s / f"{s}_{tf}_{d_from.date()}_{d_to.date()}.csv"
            try:
                print(f"[export] {s} {tf} -> {out_csv}")
                export_csv(s, tf, d_from, d_to, out_csv)
            except Exception as e:
                errors += 1
                sys.stderr.write(f"[export][ERR] {s} {tf}: {e}\n")

    mt5.shutdown()
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()