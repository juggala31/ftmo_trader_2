import sys, re, MetaTrader5 as mt5

# Default search patterns aimed at Cash/Spot names
DEFAULT_PATTERNS = [
    "XAUUSD", "US30Cash", "US100Cash", "US500", "USOILCash", "BTCUSD",
    "US30", "US100", "SPX500", "USA500", "USTECH", "USOIL"
]

AVOID = re.compile(r"\.sim$|[HMUZ]\d{2}\.sim$", re.IGNORECASE)

def main():
    pats = sys.argv[1:] or DEFAULT_PATTERNS
    if not mt5.initialize():
        print("init failed:", mt5.last_error()); return
    try:
        all_syms = mt5.symbols_get("*")
        names = [s.name for s in (all_syms or [])]
        found = []
        for p in pats:
            pl = p.lower()
            for n in names:
                if pl in n.lower() and not AVOID.search(n):
                    if mt5.symbol_select(n, True):
                        print(f"[ENABLE] {n}")
                        found.append(n)
        if not found:
            print("[INFO] no cash/spot matches enabled; adjust patterns or enable via MT5 manually.")
        else:
            print(f"[OK] enabled {len(found)} symbol(s).")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
