import os, re, time, MetaTrader5 as mt5

root = r"C:\ftmo_trader_2"
envp = os.path.join(root, ".env")
maps = {}
if os.path.exists(envp):
    txt = open(envp, "r", encoding="utf-8").read()
    for k in ["US30","US100","SPX500","XAUUSD","BTCUSD","WTICOUSD"]:
        m = re.search(rf"(?m)^\s*MAP_{k}\s*=\s*(.+)\s*$", txt)
        if m: maps[k] = m.group(1).strip()

order_list = [maps.get("XAUUSD"), maps.get("BTCUSD"), maps.get("US30")]
order_list = [s for s in order_list if s]

print("[maps]", maps)
print("[order_try]", order_list)

if not mt5.initialize():
    print("[MT5] init failed:", mt5.last_error()); raise SystemExit

# ensure visible and show tick status
ok_syms = []
for sym in order_list:
    info = mt5.symbol_info(sym)
    if not info or not info.visible:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    print(f"[tick] {sym:10s} visible={bool(info and info.visible)} tick_ok={bool(tick)} bid={getattr(tick,'bid',None)} ask={getattr(tick,'ask',None)}")
    if info and info.visible and tick: ok_syms.append(sym)

# optional micro test: try first with a live tick
if ok_syms:
    sym = ok_syms[0]
    tick = mt5.symbol_info_tick(sym)
    vol  = 0.01
    print(f"[test] BUY {sym} vol {vol} ask {tick.ask}")
    req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=sym, volume=vol, type=mt5.ORDER_TYPE_BUY,
               price=tick.ask, deviation=100, magic=24680, comment="PIPELINE_TEST",
               type_filling=mt5.ORDER_FILLING_FOK)
    r = mt5.order_send(req)
    print("[send]", r)
    # flatten if opened
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
else:
    print("[note] No live ticks on these instruments right now; session likely closed.")

mt5.shutdown()
