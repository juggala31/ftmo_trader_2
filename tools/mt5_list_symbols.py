#!/usr/bin/env python3
import os
try:
    import MetaTrader5 as mt5
except Exception as e:
    print(f"[mt5] import error: {e}")
    raise

FILTERS = [
    # Metals
    "XAU", "XAG", "XPT", "XPD",
    # Indices
    "US30", "US100", "NAS100", "SPX", "US500", "GER40", "DE40", "UK100", "FTSE", "JP225", "EU50",
    # Oil
    "WTI", "OIL", "BRENT", "XBR",
    # Crypto
    "BTC", "ETH"
]

def main():
    kw = {}
    term = os.environ.get("MT5_TERMINAL_PATH") or os.environ.get("TERMINAL_PATH")
    if term: kw["path"] = term
    login  = os.environ.get("MT5_LOGIN")
    pwd    = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")
    ok = mt5.initialize(login=int(login), password=pwd, server=server, **kw) if login and pwd and server else mt5.initialize(**kw)
    print(f"[mt5] initialize -> {ok}")
    if not ok: return

    def shortline(s): return f"{s.name:20s} | point={s.point}  digits={s.digits}  trade_mode={s.trade_mode}"
    all_syms = mt5.symbols_get() or []
    print(f"[info] total symbols: {len(all_syms)}")

    any_hits = False
    for f in FILTERS:
        subset = [s for s in all_syms if f.upper() in s.name.upper()]
        subset.sort(key=lambda x: x.name)
        if subset:
            any_hits = True
            print(f"\n=== matches for '{f}' ({len(subset)}) ===")
            for s in subset[:30]:
                print(" ", shortline(s))

    if not any_hits:
        print("\n=== first 50 symbols (no filter hits) ===")
        for s in all_syms[:50]:
            print(" ", shortline(s))

    mt5.shutdown()

if __name__ == "__main__":
    main()