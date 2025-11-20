import os, csv, time
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "ai_signals.csv"))
while not os.path.exists(path):
    print("waiting for logs\\ai_signals.csv …"); time.sleep(3)
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
    for r in rows[-20:]:
        try:
            ts,sym,tf,side,conf,ema_f,ema_s,rsi,placed,vol,reason = r
            print(f"{ts}  {sym:8s} {side:4s} conf={float(conf):.2f} placed={placed} vol={vol} reason={reason}")
        except Exception:
            print(",".join(r))
