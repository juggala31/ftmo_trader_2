import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List


ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / "ai"
DATA_DIR = ROOT / "data"

SIGNALS_PATH = AI_DIR / "ai_signals.csv"
TUNER_STATE_PATH = AI_DIR / "auto_tuner_state.json"
TRADES_PATH = DATA_DIR / "trades.csv"
PROFILE_PATH = AI_DIR / "ai_profile.json"


@dataclass
class CheckResult:
    name: str
    status: str  # OK / STALE / ERROR / MISSING / SKIP
    detail: str


def fmt_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 0:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def check_mt5() -> CheckResult:
    """
    Try to connect via brokers.mt5_adapter.MT5Adapter using env MT5_LOGIN/PASSWORD/SERVER.
    If env not set, we SKIP (likely not in live mode).
    """
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if not (login and password and server):
        return CheckResult(
            name="MT5",
            status="SKIP",
            detail="MT5_LOGIN/PASSWORD/SERVER not set in env (probably not live).",
        )

    try:
        from brokers.mt5_adapter import MT5Adapter  # type: ignore
    except Exception as e:
        return CheckResult(
            name="MT5",
            status="ERROR",
            detail=f"Import brokers.mt5_adapter failed: {e}",
        )

    try:
        adapter = MT5Adapter()
        ok, msg = adapter.connect(login=login, password=password, server=server)
        if ok:
            return CheckResult(name="MT5", status="OK", detail=f"Connected: {msg}")
        else:
            return CheckResult(name="MT5", status="ERROR", detail=f"Connect failed: {msg}")
    except Exception as e:
        return CheckResult(name="MT5", status="ERROR", detail=f"Exception: {e}")


def get_file_age(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    stat = path.stat()
    now = datetime.now(timezone.utc).timestamp()
    return now - stat.st_mtime


def check_file_age(name: str, path: Path, max_stale_sec: float) -> CheckResult:
    age = get_file_age(path)
    if age is None:
        return CheckResult(name=name, status="MISSING", detail=f"{path} not found.")
    status = "OK" if age <= max_stale_sec else "STALE"
    return CheckResult(
        name=name,
        status=status,
        detail=f"age={fmt_age(age)} (path={path})",
    )


def check_signals(max_stale_sec: float) -> CheckResult:
    """
    Look at ai_signals.csv:
      - Prefer using last numeric ts column if present.
      - If no usable ts, fall back to file mtime (age-based).
    """
    if not SIGNALS_PATH.exists():
        return CheckResult(
            name="AI_SIGNALS",
            status="MISSING",
            detail=f"{SIGNALS_PATH} not found.",
        )

    last_ts = None
    try:
        with SIGNALS_PATH.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            # Try to find a plausible timestamp column
            if r.fieldnames:
                candidates = [c for c in r.fieldnames if c.lower() in ("ts", "timestamp", "time")]
            else:
                candidates = []

            for row in r:
                for col in candidates:
                    ts_raw = (row.get(col) or "").strip()
                    if not ts_raw:
                        continue
                    try:
                        last_ts = float(ts_raw)
                    except Exception:
                        continue
    except Exception as e:
        return CheckResult(
            name="AI_SIGNALS",
            status="ERROR",
            detail=f"Failed to read {SIGNALS_PATH}: {e}",
        )

    now = datetime.now(timezone.utc).timestamp()

    # If we found a numeric ts, use that age
    if last_ts is not None:
        age = now - last_ts
        status = "OK" if age <= max_stale_sec else "STALE"
        return CheckResult(
            name="AI_SIGNALS",
            status=status,
            detail=f"last_ts_age={fmt_age(age)} (path={SIGNALS_PATH})",
        )

    # Fallback: no usable ts -> treat based on file age, not as a hard ERROR
    file_age = get_file_age(SIGNALS_PATH)
    if file_age is None:
        return CheckResult(
            name="AI_SIGNALS",
            status="MISSING",
            detail=f"{SIGNALS_PATH} not found (race).",
        )
    status = "OK" if file_age <= max_stale_sec else "STALE"
    return CheckResult(
        name="AI_SIGNALS",
        status=status,
        detail=f"no numeric ts; using file age={fmt_age(file_age)} (path={SIGNALS_PATH})",
    )


def check_ai_risk(max_risk_pct: float = 0.02) -> CheckResult:
    """
    Sanity-check AI profile + tuner state:

    - For each enabled live symbol:
        - If atr_risk_pct > max_risk_pct -> flag
        - If tuner bucket=weak or status in {observe,paper} or weak_streak>=3 -> flag

    Returns:
        OK if no issues, STALE if some live configs look risky/weak,
        MISSING/ERROR if files missing or unreadable.
    """
    if not PROFILE_PATH.exists():
        return CheckResult(
            name="AI_RISK",
            status="MISSING",
            detail=f"{PROFILE_PATH} not found.",
        )

    try:
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return CheckResult(
            name="AI_RISK",
            status="ERROR",
            detail=f"Failed to read profile: {e}",
        )

    state: dict
    if TUNER_STATE_PATH.exists():
        try:
            state = json.loads(TUNER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    else:
        state = {}

    weak_map = state.get("weak_streak", {}) if isinstance(state, dict) else {}
    bucket_map = state.get("bucket", {}) if isinstance(state, dict) else {}
    status_map = state.get("status", {}) if isinstance(state, dict) else {}

    issues = []
    live_symbols = []

    for sym, node in profile.items():
        if sym in ("global", "symbols"):
            continue
        if not isinstance(node, dict):
            continue

        enabled = node.get("enabled", False)
        trade_mode = node.get("trade_mode", "live")
        if not enabled or trade_mode != "live":
            continue

        # candidate live trading symbol
        live_symbols.append(sym)

        risk = node.get("risk", {}) or {}
        atr_risk_pct = risk.get("atr_risk_pct", None)
        bucket = bucket_map.get(sym, "?")
        tuner_status = status_map.get(sym, trade_mode)
        weak_streak = weak_map.get(sym, 0)

        # High risk?
        if atr_risk_pct is not None and atr_risk_pct > max_risk_pct:
            issues.append(f"{sym}: high risk {atr_risk_pct*100:.2f}%")

        # Weak / observe from tuner?
        if str(bucket).lower() == "weak":
            issues.append(f"{sym}: bucket=weak")
        if str(tuner_status).lower() in ("observe", "paper"):
            issues.append(f"{sym}: tuner_status={tuner_status}")
        if weak_streak >= 3:
            issues.append(f"{sym}: weak_streak={weak_streak}")

    if not live_symbols:
        return CheckResult(
            name="AI_RISK",
            status="OK",
            detail="No live symbols enabled in profile.",
        )

    if not issues:
        return CheckResult(
            name="AI_RISK",
            status="OK",
            detail=f"Live symbols={live_symbols} look sane (risk <= {max_risk_pct*100:.2f}%, no weak/observe flags).",
        )

    detail = f"Live symbols={live_symbols}; issues: " + "; ".join(issues)
    return CheckResult(
        name="AI_RISK",
        status="STALE",
        detail=detail,
    )


def run_checks(
    max_stale_signals_sec: float,
    max_stale_tuner_sec: float,
    max_stale_trades_sec: float,
    skip_mt5: bool,
    max_risk_pct: float,
) -> List[CheckResult]:
    results: List[CheckResult] = []

    if not skip_mt5:
        results.append(check_mt5())
    else:
        results.append(
            CheckResult(
                name="MT5",
                status="SKIP",
                detail="Skip MT5 check (per --no-mt5).",
            )
        )

    # AI signals heartbeat
    results.append(check_signals(max_stale_signals_sec))

    # Tuner state freshness (file mtime)
    results.append(
        check_file_age(
            name="TUNER_STATE",
            path=TUNER_STATE_PATH,
            max_stale_sec=max_stale_tuner_sec,
        )
    )

    # Trades CSV freshness
    results.append(
        check_file_age(
            name="TRADES_CSV",
            path=TRADES_PATH,
            max_stale_sec=max_stale_trades_sec,
        )
    )

    # AI risk/profile sanity
    results.append(check_ai_risk(max_risk_pct=max_risk_pct))

    return results


def print_report(results: List[CheckResult]) -> int:
    print("=== Health Watchdog ===")
    print("")
    print(f"{'CHECK':<12} {'STATUS':<8} DETAIL")
    print("-" * 72)
    worst = 0  # 0=OK/SKIP, 1=STALE, 2=ERROR/MISSING
    for r in results:
        print(f"{r.name:<12} {r.status:<8} {r.detail}")
        if r.status in ("ERROR", "MISSING"):
            worst = max(worst, 2)
        elif r.status == "STALE":
            worst = max(worst, 1)
    print("")
    if worst == 0:
        print("[health_watchdog] Overall: OK")
    elif worst == 1:
        print("[health_watchdog] Overall: STALE (something needs attention soon)")
    else:
        print("[health_watchdog] Overall: BAD (one or more checks failed)")

    return worst


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Basic health watchdog for MT5, AI signals, tuner state, trades CSV, and AI risk."
    )
    ap.add_argument(
        "--max-stale-signals-sec",
        type=float,
        default=15 * 60,
        help="Max age in seconds for last AI signal (default: 900 = 15min).",
    )
    ap.add_argument(
        "--max-stale-tuner-sec",
        type=float,
        default=36 * 3600,
        help="Max age in seconds for tuner state file mtime (default: 36h).",
    )
    ap.add_argument(
        "--max-stale-trades-sec",
        type=float,
        default=24 * 3600,
        help="Max age in seconds for trades.csv mtime (default: 24h).",
    )
    ap.add_argument(
        "--max-risk-pct",
        type=float,
        default=0.02,
        help="Max allowed atr_risk_pct for live symbols (default: 0.02 = 2%%).",
    )
    ap.add_argument(
        "--no-mt5",
        action="store_true",
        help="Skip MT5 connectivity check.",
    )

    args = ap.parse_args(argv)

    results = run_checks(
        max_stale_signals_sec=args.max_stale_signals_sec,
        max_stale_tuner_sec=args.max_stale_tuner_sec,
        max_stale_trades_sec=args.max_stale_trades_sec,
        skip_mt5=args.no_mt5,
        max_risk_pct=args.max_risk_pct,
    )
    worst = print_report(results)
    # Exit code: 0=OK, 1=STALE, 2=BAD
    raise SystemExit(worst)


if __name__ == "__main__":
    main()