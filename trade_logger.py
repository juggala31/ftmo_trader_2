from __future__ import annotations
import csv, os, sqlite3, time, pathlib
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA / "trades.csv"
DB_PATH  = DATA / "trades.db"

class TradeLogger:
    def __init__(self, csv_path: pathlib.Path = CSV_PATH, db_path: pathlib.Path = DB_PATH):
        self.csv_path = csv_path
        self.db_path = db_path
        self._ensure_csv()
        self._ensure_db()

    def _ensure_csv(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "ts","ticket","symbol","side","volume",
                    "entry","close","realized_quote","realized_ccy","reason"
                ])

    def _ensure_db(self):
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("""
            CREATE TABLE IF NOT EXISTS trades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                ticket INTEGER,
                symbol TEXT,
                side TEXT,
                volume REAL,
                entry REAL,
                close REAL,
                realized_quote REAL,
                realized_ccy REAL,
                reason TEXT
            );
            """)
            con.commit()
        finally:
            con.close()

    def log_close(self, *, ticket:int, symbol:str, side:str, volume:float,
                  entry:float, close:float, realized_quote:float,
                  realized_ccy:float, reason:str="manual/auto"):
        ts = time.time()
        # CSV
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts, ticket, symbol, side, volume, entry, close, realized_quote, realized_ccy, reason])
        # SQLite
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("""
              INSERT INTO trades(ts,ticket,symbol,side,volume,entry,close,realized_quote,realized_ccy,reason)
              VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (ts, ticket, symbol, side, volume, entry, close, realized_quote, realized_ccy, reason))
            con.commit()
        finally:
            con.close()