import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

index_path = ROOT / "batch_index.csv"
if not index_path.exists():
    raise SystemExit(f"Missing {index_path}")

print(f"[INFO] Loading index: {index_path}")
idx = pd.read_csv(index_path)

# Filter to rows that actually have trades
if "has_trades_csv" in idx.columns:
    idx = idx[idx["has_trades_csv"] == True].copy()

exp_rows = []
all_trades = []

for _, row in idx.iterrows():
    exp_name = row["experiment"]
    trades_path = str(row["trades_path"])

    tpath = Path(trades_path)

    if not tpath.exists():
        print(f"[WARN] trades.csv missing for {exp_name}: {tpath}")
        continue

    try:
        df = pd.read_csv(tpath)
    except Exception as e:
        print(f"[WARN] Failed to read {tpath} for {exp_name}: {e}")
        continue

    if df.empty:
        print(f"[WARN] Empty trades.csv for {exp_name}: {tpath}")
        continue

    # Lowercase column map
    cols = {c.lower(): c for c in df.columns}

    # We only REQUIRE symbol + pnl; timeframe will be taken from experiment name
    sym_col = cols.get("symbol", None)
    pnl_col = cols.get("pnl", cols.get("profit", cols.get("pl", None)))

    if sym_col is None or pnl_col is None:
        print(f"[WARN] Missing symbol/pnl columns in {tpath}, skipping.")
        print(f"       Columns found: {list(df.columns)}")
        continue

    # Derive win flag if not present
    win_col = cols.get("is_win", None)
    if win_col is None:
        df["_is_win"] = df[pnl_col] > 0
        win_col = "_is_win"

    # Parse symbol & tf from experiment name, e.g. E0001-US30Z25.sim-M30-N_A-5y
    exp_parts = str(exp_name).split("-")
    exp_symbol = exp_parts[1] if len(exp_parts) >= 2 else None
    exp_tf     = exp_parts[2] if len(exp_parts) >= 3 else None

    # If trades.csv has symbol already, use it; otherwise fallback
    if df[sym_col].isna().all():
        df[sym_col] = exp_symbol

    # Create a "tf" column for grouping, based on experiment name
    df["tf"] = exp_tf

    # ---- experiment-level stats ----
    pnl = df[pnl_col].sum()
    trades = len(df)
    wins = df[win_col].sum()
    win_rate = wins / trades if trades > 0 else 0.0

    gross_profit = df.loc[df[pnl_col] > 0, pnl_col].sum()
    gross_loss = -df.loc[df[pnl_col] < 0, pnl_col].sum()
    if gross_loss == 0:
        pf = float("inf") if gross_profit > 0 else 1.0
    else:
        pf = gross_profit / gross_loss

    exp_rows.append({
        "experiment": exp_name,
        "symbol": exp_symbol,
        "tf": exp_tf,
        "pnl": pnl,
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "pf": pf,
    })

    # ---- stash trade-level rows for symbol×tf stats ----
    tmp = df[[sym_col, "tf", pnl_col]].copy()
    tmp.columns = ["symbol", "tf", "pnl"]
    tmp["experiment"] = exp_name
    all_trades.append(tmp)

if not exp_rows:
    raise SystemExit("[ERROR] No experiment rows built. Check trades.csv structure in your experiments.")

# === PF by experiment ===
exp_df = pd.DataFrame(exp_rows)
exp_df_sorted = exp_df.sort_values(
    by=["pf", "pnl", "win_rate", "trades"],
    ascending=[False, False, False, False],
)

exp_out = ROOT / "batch_pf_by_experiment.csv"
exp_df_sorted.to_csv(exp_out, index=False)
print(f"[OK] Wrote {exp_out} ({len(exp_df_sorted)} experiments).")

# === PF by symbol × timeframe (from all trade rows) ===
if not all_trades:
    raise SystemExit("[ERROR] No trade rows collected for symbol×tf stats.")

big = pd.concat(all_trades, ignore_index=True)

def agg_symtf(group: pd.DataFrame) -> pd.Series:
    pnl = group["pnl"].sum()
    trades = len(group)
    wins = (group["pnl"] > 0).sum()
    win_rate = wins / trades if trades > 0 else 0.0

    gross_profit = group.loc[group["pnl"] > 0, "pnl"].sum()
    gross_loss = -group.loc[group["pnl"] < 0, "pnl"].sum()
    if gross_loss == 0:
        pf = float("inf") if gross_profit > 0 else 1.0
    else:
        pf = gross_profit / gross_loss

    return pd.Series({
        "pnl": pnl,
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "pf": pf,
    })

symtf_df = big.groupby(["symbol", "tf"]).apply(agg_symtf).reset_index()

symtf_out = ROOT / "batch_pf_by_symbol_tf.csv"
symtf_df.to_csv(symtf_out, index=False)
print(f"[OK] Wrote {symtf_out} ({len(symtf_df)} symbol×tf rows).")