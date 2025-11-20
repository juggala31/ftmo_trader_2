import os, time, csv, datetime as dt
LOG = r"C:\ftmo_trader_2\logs\ai_signals.csv"
os.makedirs(os.path.dirname(LOG), exist_ok=True)
hdr = ["ts_iso","symbol","tf","side","confidence","ema_fast","ema_slow","rsi","placed","vol","reason"]
exists = os.path.exists(LOG)
with open(LOG, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if not exists:
        w.writerow(hdr)
        f.flush()
    syms = ["XAUUSD","US30","US100","SPX500","WTICOUSD","BTCUSD"]
    tf = "M1"
    while True:
        now = dt.datetime.utcnow().isoformat()+"Z"
        for s in syms:
            # Dummy eval row; proves file path/permissions. No orders placed.
            w.writerow([now,s,tf,"FLAT","0.00","0","0","0","0","0","AI/EVAL"])
        f.flush()
        print("[hb] wrote AI/EVAL rows @", now)
        time.sleep(10)
