#!/usr/bin/env python3
"""
mt5_try_servers.py
Brute-test possible OANDA MT5 servers for your login/password from ENV.

Run:
  python C:\ftmo_trader_2\tools\mt5_try_servers.py
"""

import os, importlib, time

def env(k, d=""): return os.environ.get(k, d)

LOGIN  = env("MT5_LOGIN","").strip()
PASS   = env("MT5_PASSWORD","").strip() or env("MT5_PASS","").strip()
MODE   = env("APP_MODE","live").lower()
LOGIN_INT = int(LOGIN) if LOGIN.isdigit() else LOGIN

if not LOGIN or not PASS:
    print("[ERR] Missing MT5_LOGIN / MT5_PASSWORD in environment.")
    raise SystemExit(2)

# Edit/extend if you know your exact region/broker routing:
CANDIDATES = [
    # your current env first
    env("MT5_SERVER","").strip(),
    # Common OANDA MT5 naming variants people hit
    "OANDA-Demo-1",
    "OANDA-Demo-2",
    "OANDA-Demo",
    "OANDA Global-MT5 Demo",
    "OANDA Global-MT5 Live-1",
    "OANDA Global-MT5 Live-2",
    "OANDA-MT5 Demo-1",
    "OANDA-MT5 Demo-2",
    "OANDA-MT5 Demo",
    "OANDA-MT5 Live-1",
    "OANDA-MT5 Live-2",
    "OANDA Europe-MT5 Demo",
    "OANDA Asia-MT5 Demo",
    "OANDA Corporation-MT5 Demo",
    "OANDA Corporation-MT5 Live",
]

def try_adapter(server):
    for modname in ("brokers.mt5_adapter","ftmo_trader_2.brokers.mt5_adapter"):
        try:
            mod = importlib.import_module(modname)
            Adapter = getattr(mod, "MT5Adapter")
            a = Adapter("live" if MODE=="live" else "dry", server)
            fn = getattr(a, "connect", None)
            if callable(fn):
                # prefer (login, password, server) then (login, password)
                try:
                    r = fn(LOGIN_INT, PASS, server)
                except TypeError:
                    r = fn(LOGIN_INT, PASS)
                ok = bool(getattr(r, "ok", False) if not isinstance(r, dict) else r.get("ok", False))
                msg = getattr(r, "message", None) if not isinstance(r, dict) else r.get("message")
                return ok, f"adapter:{modname}", msg or "ok"
        except Exception as e:
            continue
    return False, "adapter", "import/connection failed"

def try_mt5_module(server):
    try:
        mt5 = importlib.import_module("MetaTrader5")
    except Exception as e:
        return False, f"MetaTrader5 import failed: {e}"
    # Try initialize(login, password, server)
    try:
        r = mt5.initialize(login=LOGIN_INT, password=PASS, server=server)
        if r:
            mt5.shutdown()
            return True, "ok(init kwargs)"
        err = mt5.last_error()
        return False, f"init kwargs failed: {err}"
    except TypeError:
        try:
            r = mt5.initialize(LOGIN_INT, PASS, server)
            if r:
                mt5.shutdown()
                return True, "ok(init positional)"
            err = mt5.last_error()
            return False, f"init positional failed: {err}"
        except Exception as e:
            return False, f"init positional exception: {e}"
    except Exception as e:
        return False, f"init kwargs exception: {e}"

def main():
    print(f"Login={LOGIN} (int? {isinstance(LOGIN_INT,int)})  MODE={MODE}")
    tested = set()
    best = None
    for server in [s for s in CANDIDATES if s] :
        if server in tested: continue
        tested.add(server)
        print(f"\n=== TEST server: {server} ===")

        ok_a, tag_a, msg_a = try_adapter(server)
        print(f"[adapter] {tag_a}: ok={ok_a} msg={msg_a}")

        ok_m, msg_m = try_mt5_module(server)
        print(f"[mt5.py ] {msg_m}")

        if ok_a or ok_m:
            print(f"\n[FOUND] Server works: {server}  (adapter_ok={ok_a}, mt5_py_ok={ok_m})")
            best = server
            break
        time.sleep(0.2)

    if not best:
        print("\n[FAIL] No server in candidates authenticated. Please open MT5 desktop and copy the EXACT server name shown after a successful manual login, then set MT5_SERVER to that string and rerun.")
    else:
        # Write back to .env with the working server for convenience
        envfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        try:
            with open(envfile, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            lines = []
        new = []
        wrote = False
        for ln in lines:
            if ln.startswith("MT5_SERVER="):
                new.append(f"MT5_SERVER={best}")
                wrote = True
            else:
                new.append(ln)
        if not wrote:
            new.append(f"MT5_SERVER={best}")
        with open(envfile, "w", encoding="utf-8") as f:
            f.write("\n".join(new) + "\n")
        print(f"[OK] Updated .env with MT5_SERVER={best}")

if __name__ == "__main__":
    main()
