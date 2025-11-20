import os, time, json, MetaTrader5 as mt5
maps = {}
root = r"C:\ftmo_trader_2"
envp = os.path.join(root, ".env")
if os.path.exists(envp):
    txt = open(envp, "r", encoding="utf-8").read()
    import re
    for k in ["XAUUSD","BTCUSD","US30","US100","SPX500","WTICOUSD"]:
        m = re.search(rf"(?m)^\s*MAP_{k}\s*=\s*(.+)\s*$", txt)
        if m: maps[k] = m.group(1).strip()
syms = [maps.get("XAUUSD"), maps.get("BTCUSD"), maps.get("US30")]
syms = [s for s in syms if s]
print("[testlist]", syms)
vol = 0.01
if not mt5.initialize():
    print("[MT5] init failed:", mt5.last_error()); raise SystemExit
for sym in syms:
    info = mt5.symbol_info(sym)
    if not info or not info.visible:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if not info or not info.visible or not tick:
        print(f"[skip] {sym}: no info/tick"); continue
    print("[test] BUY", sym, "vol", vol, "ask", tick.ask)
    req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=sym, volume=vol, type=mt5.ORDER_TYPE_BUY,
               price=tick.ask, deviation=100, magic=24680, comment="PIPELINE_TEST",
               type_filling=mt5.ORDER_FILLING_FOK)
    r = mt5.order_send(req)
    print("[send]", r)
    if r and getattr(r,'retcode',None) in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        time.sleep(2)
        pos = mt5.positions_get(symbol=sym)
        if pos:
            print("[flatten] closing", sym)
            tick = mt5.symbol_info_tick(sym)
            close = dict(action=mt5.TRADE_ACTION_DEAL, symbol=sym, volume=pos[0].volume,
                         type=mt5.ORDER_TYPE_SELL, price=tick.bid, deviation=100, magic=24680,
                         comment="PIPELINE_TEST_CLOSE", type_filling=mt5.ORDER_FILLING_FOK)
            print("[close]", mt5.order_send(close))
        break
mt5.shutdown()
