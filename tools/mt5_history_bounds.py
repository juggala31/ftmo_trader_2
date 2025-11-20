import os, sys, datetime as dt, pathlib
import MetaTrader5 as mt5

ROOT = pathlib.Path(__file__).resolve().parents[1]
SYMS = os.getenv("HB_SYMBOLS","XAUZ25.sim,US30Z25.sim,US100Z25.sim,US500Z25.sim,USOILZ25.sim,BTCX25.sim").split(",")
TFS  = ["M1","M5","M15","M30","H1","H4"]
TF   = {"M1":mt5.TIMEFRAME_M1,"M5":mt5.TIMEFRAME_M5,"M15":mt5.TIMEFRAME_M15,"M30":mt5.TIMEFRAME_M30,"H1":mt5.TIMEFRAME_H1,"H4":mt5.TIMEFRAME_H4}

def init():
    login=os.getenv("MT5_LOGIN"); pwd=os.getenv("MT5_PASSWORD"); srv=os.getenv("MT5_SERVER")
    if login and pwd and srv:
        try: login=int(login)
        except: pass
        if not mt5.initialize(login=login,password=pwd,server=srv):
            print("[bounds] init(login) failed:", mt5.last_error(), file=sys.stderr)
            if not mt5.initialize():
                print("[bounds] init() failed:", mt5.last_error(), file=sys.stderr); sys.exit(2)
    else:
        if not mt5.initialize():
            print("[bounds] init() failed:", mt5.last_error(), file=sys.stderr); sys.exit(2)

def oldest_dt(symbol, tf_const):
    # Pull a big count far in the past; get first bar's time if any
    # Start from 10y ago for safety.
    start = dt.datetime.now() - dt.timedelta(days=365*10)
    rates = mt5.copy_rates_from(symbol, tf_const, start, 500000)
    if rates is None or len(rates)==0:
        return None
    return dt.datetime.utcfromtimestamp(int(rates['time'][0]))

def main():
    init()
    print("[bounds] finding oldest bar per symbol/TF…")
    today = dt.datetime.now().date()
    for s in [x.strip() for x in SYMS]:
        for name in TFS:
            odt = oldest_dt(s, TF[name])
            if odt:
                print(f"[bounds] {s:12s} {name:3s} oldest={odt.date()}  (to={today})  suggested FROM={odt.date()}")
            else:
                print(f"[bounds] {s:12s} {name:3s} oldest=NONE")

    mt5.shutdown()

if __name__ == "__main__":
    main()