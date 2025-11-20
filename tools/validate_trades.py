import csv
import json
import math
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRADES = ROOT / "data" / "trades.csv"
AI_PROFILE = ROOT / "ai" / "ai_profile.json"


def load_expected_symbols() -> List[str]:
    """
    Try to load expected symbols from ai_profile.json.
    Fallback to a static set of your main sim+live symbols.
    """
    symbols = set()
    try:
        if AI_PROFILE.exists():
            with AI_PROFILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                symbols.update(str(k) for k in data.keys())
    except Exception as e:
        print(f"[validate_trades][warn] failed to load {AI_PROFILE}: {e}", file=sys.stderr)

    if not symbols:
        # Fallback: main sim + live symbols
        symbols.update(
            {
                "US30Z25.sim",
                "US100Z25.sim",
                "US500Z25.sim",
                "XAUZ25.sim",
                "USOILZ25.sim",
                "BTCX25.sim",
                "US30USD",
                "NAS100USD",
                "SPX500USD",
                "XAUUSD",
                "WTICOUSD",
                "BTCUSD",
            }
        )

    return sorted(symbols)


REQUIRED_BASE_FIELDS = ["ts", "symbol", "side", "volume", "realized_ccy"]
ALT_ENTRY_FIELDS = ["entry_price", "entry"]
ALT_CLOSE_FIELDS = ["close_price", "close"]


def pick_field(row: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row and str(row[c]).strip() != "":
            return c
    return None


def parse_float(val: str) -> Optional[float]:
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def validate_trades(
    trades_path: Path,
    expected_symbols: List[str],
    strict: bool = False,
    max_rows: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Returns: (rows, warnings, errors)
    """
    if not trades_path.exists():
        print(f"[validate_trades][error] trades file not found: {trades_path}", file=sys.stderr)
        return 0, 0, 1

    expected_set = set(expected_symbols)
    warnings = 0
    errors = 0
    rows = 0

    last_ts: Optional[float] = None

    print(f"[validate_trades] checking {trades_path}")
    with trades_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        print(f"[validate_trades] header: {header}")

        # Quick header sanity
        missing_header_fields = [fld for fld in REQUIRED_BASE_FIELDS if fld not in header]
        if missing_header_fields:
            print(
                f"[validate_trades][warn] missing header fields: {missing_header_fields}",
                file=sys.stderr,
            )
            warnings += 1

        entry_field_name = None
        close_field_name = None

        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                break

            row_id = f"row#{rows}"

            # --- Required base fields ---
            for fld in REQUIRED_BASE_FIELDS:
                if fld not in row or str(row[fld]).strip() == "":
                    print(f"[error] {row_id}: missing required field '{fld}'", file=sys.stderr)
                    errors += 1

            # --- Symbol checks ---
            symbol = str(row.get("symbol", "")).strip()
            if symbol == "":
                print(f"[error] {row_id}: empty symbol", file=sys.stderr)
                errors += 1
            elif symbol not in expected_set:
                print(f"[warn]  {row_id}: unexpected symbol '{symbol}'", file=sys.stderr)
                warnings += 1

            # --- Entry / close fields ---
            if entry_field_name is None:
                entry_field_name = pick_field(row, ALT_ENTRY_FIELDS)
            if close_field_name is None:
                close_field_name = pick_field(row, ALT_CLOSE_FIELDS)

            ef = entry_field_name or pick_field(row, ALT_ENTRY_FIELDS)
            cf = close_field_name or pick_field(row, ALT_CLOSE_FIELDS)

            if ef is None:
                print(
                    f"[warn]  {row_id}: no entry or entry_price field found (ALT_ENTRY_FIELDS={ALT_ENTRY_FIELDS})",
                    file=sys.stderr,
                )
                warnings += 1
            else:
                ep = parse_float(row.get(ef, ""))
                if ep is None:
                    print(
                        f"[warn]  {row_id}: entry field '{ef}' could not be parsed as float: {row.get(ef)!r}",
                        file=sys.stderr,
                    )
                    warnings += 1

            if cf is None:
                print(
                    f"[warn]  {row_id}: no close or close_price field found (ALT_CLOSE_FIELDS={ALT_CLOSE_FIELDS})",
                    file=sys.stderr,
                )
                warnings += 1
            else:
                cp = parse_float(row.get(cf, ""))
                if cp is None:
                    print(
                        f"[warn]  {row_id}: close field '{cf}' could not be parsed as float: {row.get(cf)!r}",
                        file=sys.stderr,
                    )
                    warnings += 1

            # --- realized_ccy sanity ---
            pnl = parse_float(row.get("realized_ccy", ""))
            if pnl is None:
                print(
                    f"[warn]  {row_id}: realized_ccy not parseable as float: {row.get('realized_ccy')!r}",
                    file=sys.stderr,
                )
                warnings += 1
            else:
                if not math.isfinite(pnl):
                    print(f"[error] {row_id}: realized_ccy is not finite: {pnl}", file=sys.stderr)
                    errors += 1
                elif abs(pnl) > 1e7:
                    print(
                        f"[warn]  {row_id}: realized_ccy looks extremely large: {pnl}",
                        file=sys.stderr,
                    )
                    warnings += 1

            # --- timestamp sanity (ts) ---
            ts_raw = str(row.get("ts", "")).strip()
            if ts_raw != "":
                ts_val = parse_float(ts_raw)
                if ts_val is None:
                    # Maybe ISO datestamp – not fatal
                    print(
                        f"[warn]  {row_id}: ts='{ts_raw}' not parseable as float (epoch seconds?)",
                        file=sys.stderr,
                    )
                    warnings += 1
                else:
                    if last_ts is not None and ts_val < last_ts:
                        print(
                            f"[warn]  {row_id}: timestamp out of order (prev={last_ts}, current={ts_val})",
                            file=sys.stderr,
                        )
                        warnings += 1
                    last_ts = ts_val
            else:
                print(f"[warn]  {row_id}: empty ts field", file=sys.stderr)
                warnings += 1

    print(
        f"[validate_trades] done. rows={rows}, warnings={warnings}, errors={errors}"
    )

    if strict and (errors > 0 or warnings > 0):
        return rows, warnings, max(errors, 1)

    return rows, warnings, errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate trades.csv structure and basic sanity checks."
    )
    ap.add_argument(
        "--trades",
        type=str,
        default=str(DEFAULT_TRADES),
        help=f"Path to trades CSV (default: {DEFAULT_TRADES})",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if any warnings or errors are found.",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional max rows to validate (for speed).",
    )
    args = ap.parse_args()

    trades_path = Path(args.trades)
    expected_symbols = load_expected_symbols()
    rows, warnings, errors = validate_trades(
        trades_path=trades_path,
        expected_symbols=expected_symbols,
        strict=args.strict,
        max_rows=args.max_rows,
    )

    # Exit code semantics:
    # - 0 if no errors (and no warnings in strict mode)
    # - 1 if errors (or warnings in strict mode)
    if errors > 0 or (args.strict and (warnings > 0 or errors > 0)):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()