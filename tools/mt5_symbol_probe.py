import os, sys
import MetaTrader5 as mt5

TARGETS = {
  "XAU":  ["XAUUSD","XAUUSD.","XAU","GOLD","XAUUSDm","XAUUSD.r"],
  "US30": ["US30","US30USD","US30.cash","DJ30","DJI30","US30m","US30z"],
  "NAS":  ["US100","NAS100","NAS100USD","US100.cash","USTEC","USTECH"],
  "SPX":  ["SPX500","SPX500USD","US500","US500.cash","SPX"],
  "WTI":  ["WTICOUSD","USOIL","OIL.WTI","WTI","WTI.cash","WTICO"],
  "BTC":  ["BTCUSD","BTC","BTCUSD.","XBTUSD"],
}

def init():
    login=os.getenv("MT5_LOGIN"); pwd=os.getenv("MT5_PASSWORD"); srv=os.getenv("MT5_SERVER")
    if login and pwd and srv:
        try: login=int(login)
        except: pass
        if not mt5.initialize(login=login, password=pwd, server=srv):
            print("[probe] init(login) failed:", mt5.last_error(), file=sys.stderr)
            if not mt5.initialize():
                print("[probe] init() failed:", mt5.last_error(), file=sys.stderr); sys.exit(2)
    else:
        if not mt5.initialize():
            print("[probe] init() failed:", mt5.last_error(), file=sys.stderr); sys.exit(2)

def best_symbol(candidates):
    for s in candidates:
        si = mt5.symbol_info(s)
        if si and (si.visible or mt5.symbol_select(s, True)):
            rates = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M1, 0, 50)
            if rates is not None and len(rates) > 0:
                return s, len(rates), float(rates[-1]["close"])
    return None, 0, None

def main():
    init()
    print(f"[probe] Terminal: version={mt5.version()}")
    print(f"[probe] Total symbols in Market Watch listable: {len(mt5.symbols_get())}")
    resolved = {}
    for key, guesses in TARGETS.items():
        sym, bars, lastc = best_symbol(guesses)
        if sym:
            resolved[key] = sym
            print(f"[probe] {key}: {sym}  (M1 bars now={bars}, last_close={lastc})")
        else:
            print(f"[probe] {key}: NOT FOUND (check Market Watch names)")
    # Also print a ready-to-copy line for EXPORT_SYMBOLS:
    ordered = [resolved.get("XAU"), resolved.get("US30"), resolved.get("NAS"), resolved.get("SPX"),
               resolved.get("WTI"), resolved.get("BTC")]
    ordered = [s for s in ordered if s]
    print("\nSUGGESTED_SYMBOL_MAP =", resolved)
    if ordered:
        print("EXPORT_SYMBOLS_LINE =", ",".join(ordered))
    mt5.shutdown()

if __name__ == "__main__":
    main()