param(
    [int]$SinceDays = 14,
    [int]$MinTrades = 20,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $here "..")
Set-Location $root

Write-Host "[live-tuner] Root:       $root"
Write-Host "[live-tuner] SinceDays:  $SinceDays"
Write-Host "[live-tuner] MinTrades:  $MinTrades"
Write-Host "[live-tuner] DryRun:     $DryRun"
Write-Host ""

# 1) Refresh trades from MT5 into data\trades.csv
Write-Host "[1/4] Refreshing trades from MT5 -> data\\trades.csv..." -ForegroundColor Cyan
Write-Host "[RUN] python tools\\mt5_trades_to_trades_csv.py"
python tools\\mt5_trades_to_trades_csv.py

# 2) Validate trades
Write-Host ""
Write-Host "[2/4] Validating data\\trades.csv..." -ForegroundColor Cyan
Write-Host "[RUN] python tools\\validate_trades.py --trades data\\trades.csv --strict"
python tools\\validate_trades.py --trades data\\trades.csv --strict

# 3) Ensure profile is in LIVE mode (2% ATR_RISK cap)
Write-Host ""
Write-Host "[3/4] Setting ai_profile.json mode=live (2% ATR_RISK cap)..." -ForegroundColor Cyan
Write-Host "[RUN] python tools\\set_trading_mode.py --mode live"
python tools\\set_trading_mode.py --mode live

# 4) Run auto-tuner (live-aware logic)
Write-Host ""
Write-Host "[4/4] Running ai_auto_tuner.py (LIVE mode: wider risk band, slower to paper weak symbols)..." -ForegroundColor Cyan

$dryArg = @()
if ($DryRun) {
    $dryArg = @("--dry-run")
}

$cmd = @(
    "tools\\ai_auto_tuner.py",
    "--trades-csv", "data\\trades.csv",
    "--profile", "ai\\ai_profile.json",
    "--state", "ai\\auto_tuner_state.json",
    "--since-days", $SinceDays,
    "--min-trades", $MinTrades
) + $dryArg

Write-Host "[RUN] python $($cmd -join ' ')"
python @cmd

Write-Host ""
Write-Host "[live-tuner] Done." -ForegroundColor Green