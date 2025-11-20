$ROOT = "C:\ftmo_trader_2"
Set-Location $ROOT

$trades = Join-Path $ROOT "data\trades.csv"
$profile = Join-Path $ROOT "ai\ai_profile.json"
$state = Join-Path $ROOT "ai\auto_tuner_state.json"

# Lookback window in days
$sinceDays = 7

# Minimum trades per symbol to consider tuning
$minTrades = 10

# DRY-RUN for safety: change to $false when you are ready to let it edit ai_profile.json
$dryRun = $true

$dryArg = ""
if ($dryRun) {
    $dryArg = "--dry-run"
}

Write-Host "[RUN] python tools\ai_auto_tuner.py --trades-csv $trades --profile $profile --state $state --since-days $sinceDays --min-trades $minTrades $dryArg" -ForegroundColor Cyan
python tools\ai_auto_tuner.py --trades-csv $trades --profile $profile --state $state --since-days $sinceDays --min-trades $minTrades $dryArg