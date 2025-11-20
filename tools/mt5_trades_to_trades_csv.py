#!/usr/bin/env python3
# tools/mt5_trades_to_trades_csv.py
#
# Watches MT5 account history and writes CLOSED deals into:
#   data/trades.csv  +  data/trades.db
# using the same schema that ai_auto_tuner.py expects.
#
# You can:
#   - run it ONCE to backfill recent history
#   - or run it with --loop to keep logging as you trade

from __future__ import annotations
import os, sys, time, json, argparse, datetime as dt, csv, sqlite3
from pathlib import Path

# ---- project root / sys.path ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_PATH = DATA_DIR / "mt5_trades_to_trades_csv_state.json"

# ---- .env loader (light) ----
def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception as e:
        print(f"[dotenv] load failed: {e}")

load_dotenv(ROOT / ".env")

# ---- imports from project ----
try:
    from trade_logger import TradeLogger
except Exception as e:
    print(f"[trades_logger] cannot import TradeLogger: {e}")
    sys.exit(1)

try:
    import MetaTrader5 as mt5
except Exception as e:
    print(f"[mt5] import failed: {e}")
    sys.exit(1)


# ---- MT5 connection ----
def ensure_mt5_connected() -> None:
    """Make sure MetaTrader5 is initialized and logged in."""
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    # First try plain initialize (attach to existing terminal)
    ok = mt5.initialize()
    if not ok:
        # Fall back to explicit login if creds are present
        login_i = 0
        try:
            if login:
                login_i = int(login)
        except Exception:
            login_i = 0

        if login_i and password:
            if server:
                ok = mt5.initialize(login=login_i, password=password, server=str(server))
            else:
                ok = mt5.initialize(login=login_i, password=password)
        else:
            ok = False

    if not ok:
        try:
            code, msg = mt5.last_error()
        except Exception:
            code, msg = ("?", "unknown")
        print(f"[mt5] initialize/login failed: {code} {msg}")
        sys.exit(1)

    info = mt5.account_info()
    print(f"[mt5] connected: login={getattr(info, 'login', None)} server={getattr(info, 'server', None)}")


# ---- state helpers (to avoid double-logging) ----
def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"last_ticket": 0}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[trades_logger] state load failed, resetting: {e}")
        return {"last_ticket": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


# ---- fetch closed deals from MT5 ----
def find_new_closed_deals(last_ticket: int, lookback_days: int):
    """
    Pull closed deals from MT5 history and filter to ticket > last_ticket.
    We treat DEAL_ENTRY_OUT / DEAL_ENTRY_INOUT / DEAL_ENTRY_OUT_BY as "closers".
    """
    utcnow = dt.datetime.now(dt.timezone.utc)
    start = utcnow - dt.timedelta(days=max(lookback_days, 1))

    deals = mt5.history_deals_get(start, utcnow) or []
    out = []

    for d in deals:
        try:
            ticket = int(getattr(d, "ticket", 0))
            if ticket <= last_ticket:
                continue

            symbol = getattr(d, "symbol", "")
            if not symbol:
                continue

            volume = float(getattr(d, "volume", 0.0))
            if volume <= 0.0:
                continue

            entry_kind = getattr(d, "entry", None)
            # Keep only closes
            allowed_entries = [
                getattr(mt5, "DEAL_ENTRY_OUT", None),
                getattr(mt5, "DEAL_ENTRY_INOUT", None),
                getattr(mt5, "DEAL_ENTRY_OUT_BY", None),
            ]
            if entry_kind not in allowed_entries:
                continue

            out.append(d)
        except Exception:
            continue

    out.sort(key=lambda d: int(getattr(d, "ticket", 0)))
    return out


# ---- log deals into data/trades.csv + data/trades.db ----
def log_deals_to_trades_csv(deals, last_ticket: int):
    """
    Write closed deals as rows into:
        data/trades.csv   (ts,ticket,symbol,side,volume,entry,close,realized_quote,realized_ccy,reason)
        data/trades.db    (same schema)
    We use deal.profit for realized_quote/realized_ccy.
    """
    # Ensure CSV/DB exist and get paths from TradeLogger
    tl = TradeLogger()
    csv_path = tl.csv_path
    db_path = tl.db_path

    logged = 0

    for d in deals:
        try:
            ticket = int(getattr(d, "ticket", 0))
            symbol = getattr(d, "symbol", "")

            # Map MT5 deal type -> BUY/SELL
            dtype = int(getattr(d, "type", 0))
            # These constants exist on mt5, but we fall back to "SELL" for non-BUY types
            buy_types = [
                getattr(mt5, "DEAL_TYPE_BUY", 0),
                getattr(mt5, "DEAL_TYPE_BUY_LIMIT", 2),
                getattr(mt5, "DEAL_TYPE_BUY_STOP", 4),
                getattr(mt5, "DEAL_TYPE_BUY_STOP_LIMIT", 8),
            ]
            side = "BUY" if dtype in buy_types else "SELL"

            volume = float(getattr(d, "volume", 0.0))
            price = float(getattr(d, "price", 0.0))
            profit = float(getattr(d, "profit", 0.0))
            comment = getattr(d, "comment", "") or "mt5_deal"

            # Use the actual close time from MT5 as ts
            tdt = getattr(d, "time", None)
            if hasattr(tdt, "timestamp"):
                ts = float(tdt.timestamp())
            else:
                ts = time.time()

            # For now we don't try to reconstruct the original entry price.
            # The auto-tuner only uses realized_ccy for stats, so it's OK if entry==close.
            entry_price = price
            close_price = price
            realized_quote = profit
            realized_ccy = profit

            # --- CSV append ---
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ts, ticket, symbol, side, volume,
                    entry_price, close_price,
                    realized_quote, realized_ccy,
                    comment
                ])

            # --- DB insert ---
            con = sqlite3.connect(db_path)
            try:
                con.execute(
                    """
                    INSERT INTO trades(ts,ticket,symbol,side,volume,entry,close,realized_quote,realized_ccy,reason)
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (ts, ticket, symbol, side, volume,
                     entry_price, close_price,
                     realized_quote, realized_ccy,
                     comment),
                )
                con.commit()
            finally:
                con.close()

            logged += 1
            if ticket > last_ticket:
                last_ticket = ticket

        except Exception as e:
            print(f"[trades_logger][warn] failed to log deal: {e!r}")

    return logged, last_ticket


# ---- one-shot run ----
def run_once(lookback_days: int):
    ensure_mt5_connected()
    state = load_state()
    last_ticket = int(state.get("last_ticket", 0))

    deals = find_new_closed_deals(last_ticket, lookback_days)
    if not deals:
        print(f"[trades_logger] no new closed deals (last_ticket={last_ticket})")
        return

    logged, last_ticket = log_deals_to_trades_csv(deals, last_ticket)
    state["last_ticket"] = last_ticket
    save_state(state)

    print(f"[trades_logger] logged {logged} new closed deals; last_ticket={last_ticket}")


# ---- CLI / main loop ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lookback-days", type=int, default=7,
                   help="history window when scanning for new deals")
    p.add_argument("--loop", action="store_true",
                   help="stay running and poll periodically")
    p.add_argument("--interval", type=float, default=30.0,
                   help="poll interval in seconds when --loop is enabled")
    args = p.parse_args()

    if not args.loop:
        run_once(args.lookback_days)
        return

    print(f"[trades_logger] starting loop (interval={args.interval}s, lookback_days={args.lookback_days})")
    try:
        while True:
            run_once(args.lookback_days)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[trades_logger] Ctrl-C, stopping.")
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()