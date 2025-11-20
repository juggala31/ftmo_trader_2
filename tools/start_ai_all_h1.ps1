Set-Location "C:\ftmo_trader_2"

# MT5 credentials
$env:MT5_LOGIN    = "1600046554"
$env:MT5_PASSWORD = "!$9EipA3Gx"
$env:MT5_SERVER   = "OANDA-Demo-1"

# App / AI mode
$env:APP_MODE     = "live"

# === AI dry-run mode ===
# 0 = real orders
# 1 = signal-only (no MT5 orders)
$env:DRY_RUN_AI = "0"

# AI universe: all 6 .sim symbols
$env:AI_SYMBOLS = "XAUZ25.sim,US30Z25.sim,US100Z25.sim,US500Z25.sim,USOILZ25.sim,BTCX25.sim"

# Timeframes processed by the AI loop (entry + confirm)
$env:AI_TFS = "M30,H1"

# Entry vs confirmation TF config
$env:AI_ENTRY_TF       = "M30"
$env:AI_CONFIRM_TF     = "H1"
$env:AI_CONFIRM_ENABLE = "1"

# Higher timeframe (trend) filter
$env:AI_MTF_ENABLE = "1"
$env:AI_MTF_TF     = "H4"

# XGBoost confidence thresholds (aligned with your tuned H1)
$env:AI_PUP_LONG  = "0.56"
$env:AI_PUP_SHORT = "0.44"

# Session window (MT5 server time) – allow all hours;
# per-symbol sessions are handled via ai\session_filters.json
$env:AI_SESSION_START_HOUR = "0"
$env:AI_SESSION_END_HOUR   = "24"

# Loop speed (bar-gated; we only act once per new bar)
$env:AI_LOOP_SLEEP_SEC = "10"

Write-Host "Starting Trader 2.0 AI (M30 entry + H1 confirm, LIVE)..." -ForegroundColor Cyan
python .\run_live.py