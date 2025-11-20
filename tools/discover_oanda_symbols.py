from __future__ import annotations
import sys, pathlib, json
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from settings import load_settings
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

GROUP_PATTERNS = {
    "indices": {
        "path_sub": ["indice","index","indices","cfd index","stock index"],
        "name_sub": ["US30","NAS100","US100","SPX","US500","US2000","DJ","DE30","GER30","GER40",
                     "UK100","FTSE","FR40","CAC","JP225","NIK","HK50","HSI","AU200","ASX",
                     "ES35","IBEX","IT40","MIB","EU50","STOXX","CH20","SMI","CN50","SG30"]
    },
    "crypto": {
        "path_sub": ["crypto","cryptocurrency","digital asset"],
        "name_sub": ["BTC","XBT","ETH","LTC","BCH","XRP","ADA","SOL","DOT","DOGE","LINK","MATIC","XLM","BNB","TRX","ATOM"]
    },
    "metals": {
        "path_sub": ["metal","precious","commodity"],
        "name_sub": ["XAU","GOLD","XAG","SILV","XPT","PLAT","XPD","PALL","COPPER","XCU"]
    }
}

def _in_any(text: str, subs: List[str]) -> bool:
    t = (text or "").lower()
    return any(s.lower() in t for s in subs)

def group_of(si) -> str|None:
    path = getattr(si, "path", "") or ""
    name = getattr(si, "name", "") or ""
    for g, patt in GROUP_PATTERNS.items():
        if _in_any(path, patt["path_sub"]) or _in_any(name, patt["name_sub"]):
            return g
    return None

def _to_yaml(obj) -> str:
    try:
        import yaml
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    except Exception:
        # simple YAML writer for our dict
        s = "groups:\n"
        for k in ["indices","crypto","metals"]:
            s += f"  {k}:\n"
            for v in obj.get("groups", {}).get(k, []):
                s += f"    - {v}\n"
        return s

def main():
    if mt5 is None:
        raise SystemExit("Install MetaTrader5: pip install MetaTrader5")
    s = load_settings()
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    ok = mt5.login(login=int(s.mt5_login) if str(s.mt5_login).isdigit() else s.mt5_login,
                   password=s.mt5_password, server=s.mt5_server)
    if not ok:
        raise SystemExit(f"mt5.login() failed: {mt5.last_error()}")

    syms = mt5.symbols_get() or []
    groups: Dict[str, List[str]] = {"indices": [], "crypto": [], "metals": []}
    for si in syms:
        g = group_of(si)
        if g in groups:
            if si.visible or mt5.symbol_select(si.name, True):
                groups[g].append(si.name)

    for k in groups:
        groups[k] = sorted(set(groups[k]))

    cfg = {"groups": groups}
    outy = ROOT / "config" / "oanda_symbols.yml"
    outy.write_text(_to_yaml(cfg), encoding="utf-8")

    print(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))
    print(f"[OK] Wrote: {outy}")

if __name__ == "__main__":
    main()