import os, sys, math
import datetime as dt
import MetaTrader5 as mt5

SYMS = ["XAUZ25.sim","US30Z25.sim","US100Z25.sim","US500Z25.sim","USOILZ25.sim","BTCX25.sim"]

def main():
    if not mt5.initialize(login=int(os.getenv("MT5_LOGIN","0") or 0),
                          password=os.getenv("MT5_PASSWORD") or "",
                          server=os.getenv("MT5_SERVER") or ""):
        print("[pos] mt5 init failed:", mt5.last_error()); return
    try:
        poss = mt5.positions_get()
        if not poss:
            print("[pos] no open positions"); return
        by_sym = {p.symbol: p for p in poss if p.symbol in SYMS}
        print(f"[pos] {len(by_sym)} tracked positions")
        for s in SYMS:
            p = by_sym.get(s)
            if not p:
                print(f"  {s:<12} : (no position)")
                continue
            tick = mt5.symbol_info_tick(s)
            si   = mt5.symbol_info(s)
            digits = si.digits if si else 2
            side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            price = tick.bid if p.type==mt5.POSITION_TYPE_BUY else tick.ask if tick else p.price_open
            # distances to SL/TP in points (if set)
            sl_dist = None
            tp_dist = None
            if p.sl and p.sl > 0:
                if p.type == mt5.POSITION_TYPE_BUY:
                    sl_dist = (p.price_open - p.sl) / (si.point if si else 0.01)
                else:
                    sl_dist = (p.sl - p.price_open) / (si.point if si else 0.01)
            if p.tp and p.tp > 0:
                if p.type == mt5.POSITION_TYPE_BUY:
                    tp_dist = (p.tp - p.price_open) / (si.point if si else 0.01)
                else:
                    tp_dist = (p.price_open - p.tp) / (si.point if si else 0.01)
            sl_str = f"{p.sl:.{digits}f}" if p.sl and p.sl>0 else "-"
            tp_str = f"{p.tp:.{digits}f}" if p.tp and p.tp>0 else "-"
            sld = f"{sl_dist:.0f}pt" if sl_dist is not None else "-"
            tpd = f"{tp_dist:.0f}pt" if tp_dist is not None else "-"
            print(f"  {s:<12} #{p.ticket} {side:<4} vol={p.volume:.2f} open={p.price_open:.{digits}f}  SL={sl_str} ({sld})  TP={tp_str} ({tpd})")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()