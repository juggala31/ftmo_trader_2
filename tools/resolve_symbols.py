from __future__ import annotations
import sys, pathlib, json, argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from settings import load_settings
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

PREFS = {
    "US30":  ["US30USD","US30","DJ","DOW"],
    "US100": ["NAS100USD","US100","NAS100","NDX","US100Cash"],
    "US500": ["SPX500USD","US500","SPX","US500Cash"],
    "BTC":   ["BTCUSD","XBTUSD","BTC","BTCUSDm"],
    "OIL":   ["WTICOUSD","USOIL","WTI","OILUSD","OIL","WTIUSD"],
    "GOLD":  ["XAUUSD","GOLD","XAUUSDm"]
}

def score_symbol(sym: str, prefs: list[str]) -> int:
    s = sym.upper()
    # higher is better; exact starts higher
    best = 0
    for i, p in enumerate(prefs):
        if s == p.upper():          best = max(best, 1000 - i)
        if p.upper() in s:          best = max(best, 500  - i)
    return best

def resolve_one(want: str, all_syms: list[str]) -> str|None:
    prefs = PREFS.get(want.upper(), [want.upper()])
    cands = [(score_symbol(s, prefs), s) for s in all_syms if any(p in s.upper() for p in prefs)]
    if not cands:
        # fallback: substring of the base token itself
        base = want.upper()
        cands = [(1, s) for s in all_syms if base in s.upper()]
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], len(x[1])))
    return cands[0][1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", default="US30,US100,US500,BTC,OIL,GOLD")
    args = ap.parse_args()
    wants = [w.strip().upper() for w in args.want.split(",") if w.strip()]

    if mt5 is None:
        raise SystemExit("Install MetaTrader5: pip install MetaTrader5")

    s = load_settings()
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    if not mt5.login(login=int(s.mt5_login) if str(s.mt5_login).isdigit() else s.mt5_login,
                     password=s.mt5_password, server=s.mt5_server):
        raise SystemExit(f"mt5.login() failed: {mt5.last_error()}")

    all_syms = [x.name for x in (mt5.symbols_get() or []) if (x.visible or mt5.symbol_select(x.name, True))]
    resolved = {}
    for w in wants:
        r = resolve_one(w, all_syms)
        resolved[w] = r

    print(json.dumps(resolved, indent=2))
    # also print a one-line csv list for easy copy
    csv = ",".join([v for v in resolved.values() if v])
    print(f"SYMBOLS={csv}")

if __name__ == "__main__":
    main()