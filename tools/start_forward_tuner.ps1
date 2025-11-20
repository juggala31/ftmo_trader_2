param(
    [double]$SinceDays = 7,
    [int]$MinTrades = 10,
    [switch]$Loop,
    [double]$IntervalHours = 24,
    [double]$RegimeSinceDays = 7,
    [double]$RegimeMaxRisk = 0.005,  # 0.5% cap in RANGE/HIGH
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

Write-Host "[forward-tuner] Root: $ROOT"
Write-Host "[forward-tuner] SinceDays=$SinceDays MinTrades=$MinTrades Loop=$($Loop.IsPresent) IntervalHours=$IntervalHours" 
Write-Host "[forward-tuner] RegimeSinceDays=$RegimeSinceDays RegimeMaxRisk=$RegimeMaxRisk DryRun=$($DryRun.IsPresent)"
Write-Host ""

function Invoke-Once {
    param(
        [double]$SinceDays,
        [int]$MinTrades,
        [double]$RegimeSinceDays,
        [double]$RegimeMaxRisk,
        [switch]$DryRun
    )

    Write-Host "======================================="
    Write-Host "[forward-tuner] Cycle started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "======================================="
    Write-Host ""

    # 1) Pull latest trades from MT5 into data\\trades.csv
    Write-Host "[1/4] Updating trades from MT5 -> data\\trades.csv..."
    $cmd1 = "python tools\\mt5_trades_to_trades_csv.py"
    Write-Host "[RUN] $cmd1" -ForegroundColor Cyan
    Invoke-Expression $cmd1
    Write-Host ""

    # 2) Validate trades
    Write-Host "[2/4] Validating trades CSV..."
    $cmd2 = "python tools\\validate_trades.py --trades data\\trades.csv --strict"
    Write-Host "[RUN] $cmd2" -ForegroundColor Cyan
    Invoke-Expression $cmd2
    Write-Host ""

    # 3) Run auto-tuner
    Write-Host "[3/4] Running auto-tuner..."
    $dryArg = ""
    if ($DryRun) {
        $dryArg = "--dry-run"
    }
    $cmd3 = "python tools\\ai_auto_tuner.py --trades-csv data\\trades.csv --profile ai\\ai_profile.json --state ai\\auto_tuner_state.json --since-days $SinceDays --min-trades $MinTrades $dryArg"
    Write-Host "[RUN] $cmd3" -ForegroundColor Cyan
    Invoke-Expression $cmd3
    Write-Host ""

    # 4) Apply regime-based risk caps
    Write-Host "[4/4] Applying regime-based risk caps (RANGE/HIGH -> cap to $RegimeMaxRisk)..."
    $cmd4 = "python tools\\regime_risk_cap.py --tf H1 --since-days $RegimeSinceDays --max-risk $RegimeMaxRisk $dryArg"
    Write-Host "[RUN] $cmd4" -ForegroundColor Cyan
    Invoke-Expression $cmd4
    Write-Host ""

    Write-Host "[forward-tuner] Cycle complete."
    Write-Host ""
}

if ($Loop) {
    Write-Host "[forward-tuner] Loop mode enabled. IntervalHours=$IntervalHours"
    Write-Host ""
    while ($true) {
        Invoke-Once -SinceDays $SinceDays -MinTrades $MinTrades -RegimeSinceDays $RegimeSinceDays -RegimeMaxRisk $RegimeMaxRisk -DryRun:$DryRun
        $sec = [int]([double]::Parse($IntervalHours.ToString()) * 3600.0)
        Write-Host "[forward-tuner] Sleeping $sec seconds..."
        Start-Sleep -Seconds $sec
    }
} else {
    Invoke-Once -SinceDays $SinceDays -MinTrades $MinTrades -RegimeSinceDays $RegimeSinceDays -RegimeMaxRisk $RegimeMaxRisk -DryRun:$DryRun
}