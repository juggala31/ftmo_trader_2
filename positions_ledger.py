from __future__ import annotations
import csv, sqlite3, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
CSV = DATA / "positions.csv"
DB  = DATA / "positions.db"

class PositionsLedger:
    def __init__(self, csv_path: pathlib.Path = CSV, db_path: pathlib.Path = DB):
        self.csv_path = csv_path
        self.db_path  = db_path
        self._ensure_csv()
        self._ensure_db()

    def _ensure_csv(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ticket","open_ts","symbol","side","volume","entry","sl","tp","status","close_ts","close","realized_quote","realized_ccy","comment"
                ])

    def _ensure_db(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("""
            CREATE TABLE IF NOT EXISTS positions(
              ticket INTEGER PRIMARY KEY,
              open_ts REAL,
              symbol TEXT,
              side TEXT,
              volume REAL,
              entry REAL,
              sl REAL,
              tp REAL,
              status TEXT,          -- OPEN | CLOSED
              close_ts REAL,
              close REAL,
              realized_quote REAL,
              realized_ccy REAL,
              comment TEXT
            );
            """)
            con.commit()
        finally:
            con.close()

    def log_open(self, *, ticket:int, symbol:str, side:str, volume:float, entry:float,
                 sl:float|None, tp:float|None, comment:str=""):
        ts = time.time()
        # CSV append (OPEN row snapshot)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ticket, ts, symbol, side, volume, entry, sl, tp, "OPEN", "", "", "", comment])
        # DB upsert
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("""
              INSERT INTO positions(ticket,open_ts,symbol,side,volume,entry,sl,tp,status,comment)
              VALUES(?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(ticket) DO UPDATE SET
                open_ts=excluded.open_ts,symbol=excluded.symbol,side=excluded.side,
                volume=excluded.volume,entry=excluded.entry,sl=excluded.sl,tp=excluded.tp,
                status='OPEN', comment=excluded.comment
            """, (ticket, ts, symbol, side, float(volume), float(entry), (None if sl is None else float(sl)),
                  (None if tp is None else float(tp)), "OPEN", comment))
            con.commit()
        finally:
            con.close()

    def log_close_update(self, *, ticket:int, close_price:float, realized_quote:float, realized_ccy:float):
        ts = time.time()
        # CSV append (CLOSED row snapshot)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ticket, "", "", "", "", "", "", "", "CLOSED", ts, close_price, realized_quote, realized_ccy, ""])
        # DB update
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("""
              UPDATE positions
                 SET status='CLOSED',
                     close_ts=?,
                     close=?,
                     realized_quote=?,
                     realized_ccy=?
               WHERE ticket=?
            """, (ts, float(close_price), float(realized_quote), float(realized_ccy), ticket))
            con.commit()
        finally:
            con.close()