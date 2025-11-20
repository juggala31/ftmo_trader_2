import os, sys, csv, pathlib
import MetaTrader5 as mt5

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "data" / "symbol_scan.csv"
KEYS = ["XAU","US30","NAS","US100","SPX","US500","WTI","USOIL","BTC",".sim"]

def init():
    login=os.getenv("MT5_LOGIN"); pwd=os.getenv("MT5_PASSWORD"); srv=os.getenv("MT5_SERVER")
    if login and pwd and srv:
        try: login=int(login)
        except: pass
        if not mt5.initialize(login=login, password=pwd, server=srv):
            print("[dump] init(login) failed:", mt5.last_error(), file=sys.stderr)
            if not mt5.initialize():
                print("[dump] init() failed:", mt5.last_error(), file=sys.stderr); sys.exit(2)
    else:
        if not mt5.initialize():
            print("[dump] init() failed:", mt5.last_error(), file=sys.stderr); sys.exit(2)

def main():
    init()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_syms = mt5.symbols_get()
    print(f"[dump] total symbols returned: {len(all_syms)}")
    rows = []
    for si in all_syms:
        name = si.name
        ok = mt5.symbol_select(name, True)
        bars = 0; lastc = ""
        if ok:
            rates = mt5.copy_rates_from_pos(name, mt5.TIMEFRAME_M1, 0, 50)
            if rates is not None:
                bars = len(rates)
                if bars:
                    lastc = f"{float(rates[-1]['close']):.6f}"
        rows.append([name, "Y" if ok else "N", bars, lastc, si.path, si.trade_mode, si.visible])

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol","selected","m1_bars_now","last_close","path","trade_mode","visible"])
        w.writerows(rows)

    print(f"[dump] wrote -> {OUT}")

    # Highlight likely candidates by substrings
    hits = [r for r in rows if any(k in r[0].upper() for k in KEYS)]
    if hits:
        print("\n[dump] candidate hits (by name substrings):")
        for name, sel, bars, lastc, *_ in hits:
            print(f"  {name:20s}  selected={sel}  m1_bars_now={bars}  last_close={lastc}")
    else:
        print("\n[dump] no candidates matched substrings; open CSV to inspect all names.")

    mt5.shutdown()

if __name__ == "__main__":
    main()