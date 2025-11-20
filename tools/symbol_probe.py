import json, re, MetaTrader5 as mt5

WANT = {
    "XAUUSD":   ["XAUUSD","XAU","GOLD"],
    "BTCUSD":   ["BTCUSD","BTC/USD","BTC"],
    "US30":     ["US30Cash","US30","DJ30","DJI"],
    "US100":    ["US100Cash","US100","NAS100","NDX"],
    "SPX500":   ["US500","SPX500","SP500"],
    "WTICOUSD": ["WTICOUSD","USOIL","WTI","USOILCash"],
}

def best_match(hints, all_names):
    # exact CS
    for h in hints:
        if h in all_names: return h, "exact"
    # exact CI
    low = {s.lower(): s for s in all_names}
    for h in hints:
        if h.lower() in low: return low[h.lower()], "iexact"
    # startswith/contains
    for h in hints:
        m = [s for s in all_names if s.startswith(h)]
        if m: return m[0], "startswith"
    for h in hints:
        m = [s for s in all_names if h in s]
        if m: return m[0], "contains"
    # relaxed (alnum)
    def nz(s): return re.sub(r"[^A-Za-z0-9]","",s).lower()
    for h in hints:
        for s in all_names:
            if nz(h) and nz(h) in nz(s): return s, "relaxed"
    return None, "none"

if not mt5.initialize():
    print(json.dumps({"error":"mt5_init_failed","detail":mt5.last_error()})); raise SystemExit

syms = mt5.symbols_get() or []
names = [s.name for s in syms]
out = {}
for canon, hints in WANT.items():
    b, why = best_match(hints, names)
    if b:
        try: mt5.symbol_select(b, True)
        except: pass
        info = mt5.symbol_info(b)
        out[canon] = {"broker": b, "visible": bool(info and info.visible), "why": why}
    else:
        out[canon] = {"broker": None, "visible": False, "why": "not_found"}

print(json.dumps(out))
mt5.shutdown()
