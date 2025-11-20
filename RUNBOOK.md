# FTMO Trader 2.0 RUNBOOK

Location: `C:\ftmo_trader_2`

This runbook describes how to operate the FTMO Trader 2.0 system in its current state:

- Live trading engine with FTMO style risk guard
- XGBoost based AI signal engine (H1 focus)
- Auto tuner for live and backtest data
- Regime aware risk controls
- Daily health and performance reporting

---

## 1. Environment and basic assumptions

### 1.1 Requirements

- Windows with Python installed (same version used to set up the project)
- MetaTrader 5 terminal configured and logged in to your OANDA Demo or prop firm account
- Project root: `C:\ftmo_trader_2`

### 1.2 Important folders

- `run_live.py` – main live controller
- `ai\` – models, AI profile, tuner state, signals  
  - `ai\ai_profile.json` – master AI and risk configuration  
  - `ai\auto_tuner_state.json` – tuner state per symbol  
  - `ai\ai_signals.csv` – recent AI signals for all symbols and timeframes
- `data\` – live trades and support files  
  - `data\trades.csv` – consolidated closed trades from MT5
- `reports\` – backtest trade CSVs and metrics
- `backtest\` – backtest runner
- `tools\` – scripts and helpers (AI, tuner, health, reports, PowerShell launchers)
- `logs\` – logs written by health watchdog and other tools

---

## 2. Daily live trading flow

This is the normal routine when you want the bot trading.

### 2.1 Before market session

1. **Start MT5 and log in**

   - Launch MetaTrader 5.
   - Confirm the connection is green and the correct account is selected.
   - Confirm required symbols are visible in Market Watch:
     - `XAUUSD`, `US30USD`, `NAS100USD`, `SPX500USD`, `BTCUSD`, `WTICOUSD`.

2. **Open a PowerShell window and go to the project**

       cd C:\ftmo_trader_2

3. **(Optional) Check MT5 connection and adapter**

       python tools\mt5_auth_diag.py

   You should see things like:

   - `[OK] imported brokers.mt5_adapter.MT5Adapter`
   - `initialize(login, password, server) -> True`

### 2.2 Start live AI trader (H1, multi symbol)

Use the H1 launcher that wires AI into `run_live.py`:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\start_ai_all_h1.ps1

You should see:

- `Starting Trader 2.0 AI (all symbols, H1, LIVE)...`
- `[ftmo] enforcer ready (daily_loss=..., max_dd=...)`
- `[controller] ready (paused). Use GUI ▶ RESUME to start.`

Then in the GUI:

- Press **RESUME** when you’re ready.
- AI thread starts and begins evaluating on H1.
- Orders are sent via MT5 according to AI profile + tuner state.

### 2.3 Background health loop (optional but recommended)

Run the continuous health watchdog loop in a separate PowerShell window:

- Without MT5 check:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\start_health_loop.ps1 -NoMt5

- With MT5 check:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\start_health_loop.ps1

The loop:

- Calls `tools\health_watchdog.py` every N seconds (default 120).
- Logs to `logs\health_watchdog.log`.
- Prints overall status: `OK`, `STALE`, or `BAD`.

Stop it with Ctrl+C or by closing the window.

---

## 3. Auto tuner: forward live tuning

The forward tuner adjusts AI trade modes and risk using **recent live trades** plus a regime filter.

### 3.1 One shot forward tuning cycle

From the project root:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\start_forward_tuner.ps1 `
         -SinceDays 7 `
         -MinTrades 10 `
         -RegimeSinceDays 7 `
         -RegimeMaxRisk 0.005

Steps performed:

1. Sync MT5 closed trades into `data\trades.csv`  
   - Runs `tools\mt5_trades_to_trades_csv.py`.

2. Validate trades CSV  
   - Runs `tools\validate_trades.py --trades data\trades.csv --strict`.

3. Auto tune per symbol  
   - Runs `tools\ai_auto_tuner.py` with:
     - `since_days=7`, `min_trades=10`
     - Classifies each symbol as `strong`, `neutral`, `weak`, or `insufficient`.
     - Adjusts:
       - `enabled` (live vs paper)
       - `trade_mode`
       - `risk` (within configured band, e.g. 0.5%–2.0%)
       - thresholds (long/short)
       - weak streak counter
     - Synchronizes `.sim` ↔ live aliases:
       - `US100Z25.sim` ↔ `NAS100USD`
       - `US30Z25.sim` ↔ `US30USD`
       - `US500Z25.sim` ↔ `SPX500USD`
       - `XAUZ25.sim` ↔ `XAUUSD`
       - `USOILZ25.sim` ↔ `WTICOUSD`
       - `BTCX25.sim` ↔ `BTCUSD`
     - Writes:
       - `ai\ai_profile.json`
       - `ai\auto_tuner_state.json`.

4. Apply regime risk caps  
   - Runs `tools\regime_risk_cap.py --tf H1 --since-days 7 --max-risk 0.005`.
   - Reads `ai\ai_signals.csv` for trend/vol regime.
   - Caps risk for “dangerous” regimes if needed.
   - Writes a backup in `ai\profile_backups\`.

### 3.2 Dry run forward tuning

To see planned changes without writing them:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\start_forward_tuner.ps1 `
         -SinceDays 7 `
         -MinTrades 10 `
         -RegimeSinceDays 7 `
         -RegimeMaxRisk 0.005 `
         -DryRun

---

## 4. Daily reporting tools

### 4.1 One shot health check

Without MT5:

       cd C:\ftmo_trader_2
       python tools\health_watchdog.py --no-mt5

Checks:

- Freshness of `ai\ai_signals.csv`
- Staleness of `ai\auto_tuner_state.json`
- Staleness of `data\trades.csv`
- Sanity of live symbols and risk in `ai_profile.json`

Overall status:

- `OK` – good to go
- `STALE` – something is old / needs a refresh
- `BAD` – fix before trading

### 4.2 Daily summary table

Compact symbol view:

       cd C:\ftmo_trader_2
       python tools\daily_summary.py --since-days 30 --regime-since-days 7

Shows per symbol:

- `EN` – enabled
- `MODE` – `live` or `paper`
- `RISK%` – per trade risk (fraction × 100)
- `TRD` – number of trades in the window
- `WR%` – win rate
- `PNL` + `PF` – profit and profit factor
- `TREND` – UP / DOWN / RANGE
- `VOL` – LOW / NORM / HIGH

### 4.3 Full daily brief

All in one:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\daily_brief.ps1 -NoMt5

or with MT5 check:

       powershell -ExecutionPolicy Bypass -File tools\daily_brief.ps1

Outputs:

- Health watchdog result
- Daily summary
- Detailed explain sections for all key symbols

---

## 5. Symbol drilldown and regime analysis

### 5.1 Explain a single symbol 

For example, `US100Z25.sim`:

       cd C:\ftmo_trader_2
       python tools\explain_symbol.py --symbol US100Z25.sim --since-days 365 --regime-since-days 30

Shows:

- Profile settings from `ai_profile.json`
- Tuner state (if present)
- Trade performance from `data\trades.csv`
- Market regime from `ai\ai_signals.csv` (trend, volatility, RSI)

### 5.2 Regime only

If you only want the market regime for a symbol:

       cd C:\ftmo_trader_2
       python tools\analyze_regime.py --symbol US100Z25.sim --tf H1 --since-days 30

Gives:

- trend regime (UP / DOWN / RANGE)
- volatility regime (LOW / NORM / HIGH)
- RSI median snapshot

---

## 6. Backtesting and tuning from historical CSVs

### 6.1 Single backtest

Example: backtest US30Z25.sim H1 CSV:

       cd C:\ftmo_trader_2
       python backtest\runner.py `
         --csv ai\datasets\csv\H1\US30Z25.sim.csv `
         --symbol US30Z25.sim `
         --signal demo `
         --out reports\bt_US30Z25_H1_trades.csv

Then inspect:

       python tools\report_trades.py --trades reports\bt_US30Z25_H1_trades.csv --since-days 0

### 6.2 Backtest and tune pipeline

Backtest one symbol and run the tuner on that backtest:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\run_bt_and_tune.ps1 `
         -Csv ai\datasets\csv\H1\US100Z25.sim.csv `
         -Symbol US100Z25.sim `
         -SinceDays 365 `
         -MinTrades 50 `
         -DryRun

### 6.3 Batch backtests for all H1 symbols

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\run_all_bt_and_tune_H1.ps1 `
         -SinceDays 365 `
         -MinTrades 50

---

## 7. Trade reporting

### 7.1 Live trades

Overall live stats from `data\trades.csv`:

       cd C:\ftmo_trader_2
       python tools\report_trades.py --trades data\trades.csv --since-days 30

Per symbol, e.g. `US100Z25.sim`:

       python tools\report_trades.py --symbol US100Z25.sim

### 7.2 Backtest trades

Per backtest file:

       python tools\report_trades.py --trades reports\bt_US100Z25.sim_trades.csv --since-days 0

---

## 8. AI configuration and tuner state

### 8.1 Dump trade modes

To see current enabled/mode/risk/bucket/weak streak:

       cd C:\ftmo_trader_2
       python tools\dump_trade_modes.py

### 8.2 Profile fix helpers

You have helper scripts that patch `ai_profile.json` safely and write backups into `ai\profile_backups\`:

- Normalize H1 risk:

       python tools\fix_ai_profile_h1_risk.py

- Fix `.sim` runner risk if needed:

       python tools\fix_sim_risk_pct.py

---

## 9. Health loop details

### 9.1 start_health_loop.ps1

Located in `tools\start_health_loop.ps1`.

Features:

- Runs `health_watchdog.py` on a timer (default 120 sec).
- Logs to `logs\health_watchdog.log`.
- Optional `-NoMt5` flag to skip MT5 connection checks.

Example:

       cd C:\ftmo_trader_2
       powershell -ExecutionPolicy Bypass -File tools\start_health_loop.ps1 -NoMt5

You can customize:

       powershell -ExecutionPolicy Bypass -File tools\start_health_loop.ps1 -IntervalSec 300 -NoMt5

---

## 10. Typical daily workflow

### Start of day

1. Start MT5 and confirm connection and symbols.
2. Open PowerShell:

       cd C:\ftmo_trader_2

3. Optionally check MT5:

       python tools\mt5_auth_diag.py

4. Start live AI trader:

       powershell -ExecutionPolicy Bypass -File tools\start_ai_all_h1.ps1

   Then press RESUME in the GUI.

5. Start health loop:

       powershell -ExecutionPolicy Bypass -File tools\start_health_loop.ps1 -NoMt5

### Mid day

- Run daily brief:

       powershell -ExecutionPolicy Bypass -File tools\daily_brief.ps1 -NoMt5

- Drill into symbols if needed:

       python tools\explain_symbol.py --symbol US100Z25.sim --since-days 365 --regime-since-days 30

### End of day

1. Run forward tuner:

       powershell -ExecutionPolicy Bypass -File tools\start_forward_tuner.ps1 `
         -SinceDays 7 `
         -MinTrades 10 `
         -RegimeSinceDays 7 `
         -RegimeMaxRisk 0.005

2. Check modes:

       python tools\dump_trade_modes.py

3. Optionally run daily brief again and save logs.

4. Stop AI / close MT5 when done.

---

## 11. Notes and troubleshooting

- If `AI_SIGNALS` is `STALE`:
  - Confirm `run_live.py` with AI is running.
  - Ensure `ai\ai_signals.csv` is being updated.

- If tuner reports insufficient trades:
  - Collect more live trades before relying on those decisions.

- If `ai_profile.json` looks broken:
  - Restore from latest backup in `ai\profile_backups\`.

This RUNBOOK is meant to be your “how to drive it” guide for the current FTMO Trader 2.0 setup.