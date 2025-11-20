import time, MetaTrader5 as mt5
if not mt5.initialize(): print("[MT5] init failed:", mt5.last_error()); raise SystemExit
acc = mt5.account_info(); print(f"[MT5] attached: {acc.login} {acc.server}")
while True:
    acc = mt5.account_info()
    print(f"\n=== ACCOUNT ===  balance={acc.balance:.2f}  equity={acc.equity:.2f}  profit={acc.profit:.2f} {acc.currency}")
    poss = mt5.positions_get() or []
    if not poss: print("(no open positions)")
    else:
        for p in poss:
            print(f"{p.ticket}  {p.symbol:8s}  {p.type}  vol={p.volume:.2f}  price={p.price_open:.5f}  sl={p.sl:.5f}  tp={p.tp:.5f}")
    time.sleep(10)
