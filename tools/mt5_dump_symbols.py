import sys, MetaTrader5 as mt5

def main():
    if not mt5.initialize():
        print("init failed:", mt5.last_error()); return
    try:
        infos = mt5.symbols_get()
        if not infos:
            print("no symbols"); return
        print("=== VISIBLE IN MARKET WATCH (selected=True) ===")
        count = 0
        for s in infos:
            if not s.visible: 
                continue
            tick = mt5.symbol_info_tick(s.name)
            has_tick = (tick is not None) and (getattr(tick, "bid", None) is not None or getattr(tick, "ask", None) is not None)
            bid = getattr(tick, "bid", None)
            ask = getattr(tick, "ask", None)
            print(f"{s.name:20s}  tick={'yes' if has_tick else 'no '}  bid={bid}  ask={ask}")
            count += 1
        print(f"--- total visible: {count} ---")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
