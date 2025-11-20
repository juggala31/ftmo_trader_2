#!/usr/bin/env python
# Symbol Health Check — Trader 2.0
# - Reads SYMBOLS and MAP_* from .env (or .env.example fallback)
# - Shows canonical -> broker mapping
# - If MetaTrader5 is available, tries symbol_select() and prints tick status
# - Safe to run even without MT5/terminal

import os, sys, re, time
from pathlib import Path
from datetime import datetime

ROOT = Path(r"C:\ftmo_trader_2")
ENV_PATH = ROOT / ".env"
ENV_EXAMPLE_PATH = ROOT / ".env.example"

def load_env_lines():
    # Prefer .env, else fall back to .env.example
    path = ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE_PATH
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def parse_env():
    symbols = []
    mapping = {}
    mt5_login = mt5_password = mt5_server = None

    # Try python-dotenv if available (optional)
    used_dotenv = False
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE_PATH)
        used_dotenv = True
    except Exception:
        pass

    if used_dotenv:
        sy = os.getenv("SYMBOLS", "")
        if sy:
            symbols = [s.strip() for s in sy.split(",") if s.strip()]
        for k, v in os.environ.items():
            if k.startswith("MAP_") and v.strip():
                canonical = k[4:].strip()
                mapping[canonical] = v.strip()
        mt5_login = os.getenv("MT5_LOGIN") or None
        mt5_password = os.getenv("MT5_PASSWORD") or None
        mt5_server = os.getenv("MT5_SERVER") or None
    else:
        # Manual parse
        lines = load_env_lines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("SYMBOLS="):
                sy = line.split("=", 1)[1]
                symbols = [s.strip() for s in sy.split(",") if s.strip()]
            elif line.startswith("MAP_"):
                k, v = line.split("=", 1) if "=" in line else (line, "")
                canonical = k[4:].strip()
                mapping[canonical] = v.strip()
            elif line.startswith("MT5_LOGIN="):
                mt5_login = line.split("=", 1)[1].strip()
            elif line.startswith("MT5_PASSWORD="):
                mt5_password = line.split("=", 1)[1].strip()
            elif line.startswith("MT5_SERVER="):
                mt5_server = line.split("=", 1)[1].strip()

    # Defaults if empty
    if not symbols:
        symbols = ["XAUUSD","US30USD","NAS100USD","SPX500USD","BTCUSD","WTICOUSD"]

    return symbols, mapping, mt5_login, mt5_password, mt5_server

def try_mt5_init(login, password, server):
    try:
        import MetaTrader5 as mt5
    except Exception:
        return None, "MetaTrader5 module not available"

    # Initialize: if creds set, use them; else try default terminal
    ok = False
    if login and password and server:
        ok = mt5.initialize(login=int(login), password=password, server=server)
    else:
        ok = mt5.initialize()
    if not ok:
        err = mt5.last_error()
        return None, f"MT5 initialize failed: {err}"
    return mt5, "initialized"

def format_dt(ts):
    try:
        if isinstance(ts, (int, float)) and ts > 0:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return "-"

def main():
    symbols, mapping, mt5_login, mt5_password, mt5_server = parse_env()

    print("\n=== SYMBOL HEALTH CHECK ===")
    env_used = ".env" if ENV_PATH.exists() else (".env.example" if ENV_EXAMPLE_PATH.exists() else "(none)")
    print(f"Env file      : {env_used}")
    print(f"Canonical set : {', '.join(symbols)}")
    if mapping:
        print("Mappings      : " + ", ".join([f"{k}->{v}" for k,v in mapping.items()]))
    else:
        print("Mappings      : (none)")

    # Resolve canonical -> broker symbol
    resolved = []
    for s in symbols:
        broker = mapping.get(s, s)
        resolved.append((s, broker))

    # Try MT5
    mt5, status = try_mt5_init(mt5_login, mt5_password, mt5_server)
    print(f"\nMT5 status    : {status}")

    # Header
    print("\n{:<12}  {:<20}  {:<8}  {:<10}  {:<19}".format("CANONICAL","BROKER_SYMBOL","SELECT","TICK?", "LAST_TICK_TIME"))
    print("-"*75)

    if mt5 is None:
        # No MT5: just show mapping
        for can, bro in resolved:
            print("{:<12}  {:<20}  {:<8}  {:<10}  {:<19}".format(can, bro, "-", "-", "-"))
        print("\nNote: Install/enable MetaTrader5 Python API and ensure your terminal is running to test ticks.")
        return

    # With MT5: select and check ticks
    try:
        for can, bro in resolved:
            sel_ok = False
            tick_ok = False
            last_ts = "-"
            try:
                sel_ok = mt5.symbol_select(bro, True)
                tick = mt5.symbol_info_tick(bro)
                if tick:
                    # Any price field is enough to consider it "ticking"
                    tick_ok = (tick.ask > 0 or tick.bid > 0 or tick.last > 0)
                    # Try to use time_msc/time; varies by broker/build
                    last_ts_val = getattr(tick, "time_msc", None)
                    if not last_ts_val:
                        last_ts_val = getattr(tick, "time", None)
                    last_ts = format_dt(last_ts_val) if last_ts_val else "-"
            except Exception as e:
                pass
            print("{:<12}  {:<20}  {:<8}  {:<10}  {:<19}".format(
                can, bro, "YES" if sel_ok else "NO", "YES" if tick_ok else "NO", last_ts
            ))
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    print("\nDone.")

if __name__ == "__main__":
    main()