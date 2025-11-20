from __future__ import annotations
import csv, json, sys, pathlib, datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "config" / "news_cache.json"

"""
CSV columns (header required):

date,time,tz,currency,impact,title
2025-11-05,13:30,UTC,USD,high,Nonfarm Payrolls
2025-11-05,15:00,UTC,USD,medium,ISM Services PMI
"""

def parse_row(r):
    d = r["date"].strip()
    t = r["time"].strip()
    tz = r.get("tz","UTC").strip() or "UTC"
    cur = r["currency"].strip().upper()
    imp = r["impact"].strip().lower()
    ttl = r.get("title","").strip()
    when = f"{d} {t} {tz}"
    try:
        # accept "UTC" only; other tz names require zoneinfo, keep simple here
        if tz != "UTC":
            raise ValueError("Only UTC supported in importer (set tz=UTC).")
        ts = dt.datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc).timestamp()
    except Exception as e:
        raise ValueError(f"Bad datetime: {when} ({e})")
    return {"ts_utc": ts, "currency": cur, "impact": imp, "title": ttl}

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/news_import.py path\\events.csv")
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            try:
                rows.append(parse_row(r))
            except Exception as e:
                print(f"[SKIP line {i+2}] {e}")
    old = []
    if CACHE.exists():
        try:
            old = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            old = []
    merged = old + rows
    # de-dup by (ts_utc,currency,impact,title)
    seen=set(); out=[]
    for ev in merged:
        k=(int(ev["ts_utc"]), ev["currency"], ev["impact"], ev.get("title",""))
        if k in seen: continue
        seen.add(k); out.append(ev)
    out.sort(key=lambda x: x["ts_utc"])
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] Imported {len(rows)} events; total cached: {len(out)}")
    print(f"[OK] cache -> {CACHE}")
if __name__ == "__main__":
    main()