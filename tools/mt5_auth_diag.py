#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/mt5_auth_diag.py
- Forces C:\ftmo_trader_2 on sys.path
- Loads .env from project root
- Tries adapter import from brokers.mt5_adapter
- Tests MetaTrader5 raw init() + init(login,pass,server)
"""

from __future__ import annotations
import os, sys, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_env(dotenv_path: str):
    if os.path.isfile(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s: 
                        continue
                    k, v = s.split("=", 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass

load_env(os.path.join(ROOT, ".env"))

APP_MODE = os.environ.get("APP_MODE", "(unset)")
MT5_LOGIN= os.environ.get("MT5_LOGIN", "(unset)")
MT5_PASS = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER=os.environ.get("MT5_SERVER", "(unset)")
EXECUTE  = os.environ.get("EXECUTE", "(unset)")

print("=== ENV CHECK ===")
pw_txt = "<EMPTY>" if MT5_PASS == "" else f"<len={len(MT5_PASS)}>"
print(f"APP_MODE={APP_MODE}")
print(f"MT5_LOGIN={MT5_LOGIN}")
print(f"MT5_PASSWORD={pw_txt}")
print(f"MT5_SERVER={MT5_SERVER}")
print(f"EXECUTE={EXECUTE}")
print("=================\n")

# ---- Adapter import test ----
print("=== brokers.mt5_adapter ===")
adapter_ok = False
msg = ""
try:
    from brokers.mt5_adapter import MT5Adapter
    ad = MT5Adapter()
    adapter_ok = True
    # prefer 3-arg; fall back to 2-arg
    ok, msg = ad.connect(os.environ.get("MT5_LOGIN"), os.environ.get("MT5_PASSWORD"), os.environ.get("MT5_SERVER"))
    if not ok:
        ok2, msg2 = ad.connect_l_p(os.environ.get("MT5_LOGIN") or 0, os.environ.get("MT5_PASSWORD") or "")
        msg = msg if ok else msg2
    print("[OK] imported brokers.mt5_adapter.MT5Adapter")
    print(f"connect(l,p,s) -> ok={ok} message={msg}")
except Exception as e:
    print(f"[ERR] could not import your MT5 adapter: {e}")

print("\n=== MetaTrader5 module ===")
try:
    import MetaTrader5 as mt5
except Exception as e:
    mt5 = None
    print(f"MetaTrader5 not importable: {e}")

if mt5:
    try:
        print("version:", getattr(mt5, "version", lambda: (0,0,''))())
    except Exception:
        print("version: (unknown)")
    try:
        ok = mt5.initialize(login=int(os.environ.get("MT5_LOGIN", "0") or 0),
                            password=os.environ.get("MT5_PASSWORD", ""),
                            server=os.environ.get("MT5_SERVER", ""))
        print("initialize(login, password, server) ->", ok)
        try:
            print("last_error:", mt5.last_error())
        except Exception:
            pass
    except Exception as e:
        print("initialize(login, password, server) exception:", e)
    try:
        ok2 = mt5.initialize()
        print("initialize() ->", ok2)
        try:
            print("fallback last_error:", mt5.last_error())
        except Exception:
            pass
    except Exception as e:
        print("fallback init exception:", e)

print("\n=== SUMMARY ===")
print(f"adapter_ok={adapter_ok} mt5_py_ok={bool(mt5)}\n")
print(textwrap.dedent("""\
Likely causes if connect fails:
  • Wrong TRADING password (investor password will fail).
  • Wrong server string for this login (must match exact MT5 server).
  • Account is MT4 (use an MT5 account/login).
  • The password in the shell is stale (restart shell) or .env not loaded.
"""))
