param(
    [switch]$NoMt5
)

$ErrorActionPreference = "Stop"

# Root is one level up from tools\
$ROOT = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $ROOT

Write-Host "[daily-brief] Root: $ROOT"

$PY_HEALTH  = "tools\health_watchdog.py"
$PY_SUMMARY = "tools\daily_summary.py"
$PY_EXPLAIN = "tools\explain_symbol.py"

# ----------------------
# 1) Health watchdog
# ----------------------
$healthArgs = @()
if ($NoMt5) { $healthArgs += "--no-mt5" }

Write-Host ""
Write-Host "===== HEALTH WATCHDOG =====" -ForegroundColor Cyan
python $PY_HEALTH @healthArgs

# ----------------------
# 2) Daily summary
# ----------------------
Write-Host ""
Write-Host "===== DAILY SUMMARY =====" -ForegroundColor Cyan
python $PY_SUMMARY --since-days 30 --regime-since-days 7

# ----------------------
# 3) Explain main symbols
# ----------------------
# You can edit this list anytime
$symbols = @(
    "US100Z25.sim",
    "XAUZ25.sim",
    "US30Z25.sim",
    "US500Z25.sim",
    "BTCX25.sim",
    "USOILZ25.sim"
)

foreach ($sym in $symbols) {
    Write-Host ""
    Write-Host ("===== EXPLAIN SYMBOL: {0} =====" -f $sym) -ForegroundColor Cyan
    python $PY_EXPLAIN --symbol $sym --since-days 365 --regime-since-days 30
}

Write-Host ""
Write-Host "[daily-brief] Done."