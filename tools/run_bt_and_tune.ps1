param(
    [Parameter(Mandatory = $true)]
    [string]$Csv,
    [Parameter(Mandatory = $true)]
    [string]$Symbol,
    [string]$Tf = "H1",
    [string]$OutCsv = "",
    [int]$SinceDays = 365,
    [int]$MinTrades = 50,
    [ValidateSet("demo","ml","live")]
    [string]$Signal = "ml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Resolve project root as parent of this script's folder
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $here "..")
Set-Location $root

Write-Host "[bt+tuner] Root: $root"
Write-Host "[bt+tuner] Symbol: $Symbol  Tf=$Tf  Signal=$Signal"
Write-Host "[bt+tuner] Csv: $Csv"
Write-Host "[bt+tuner] SinceDays=$SinceDays  MinTrades=$MinTrades  DryRun=$DryRun"

if (-not (Test-Path $Csv)) {
    throw "[bt+tuner] CSV not found: $Csv"
}

# Decide output trades CSV
if ([string]::IsNullOrWhiteSpace($OutCsv)) {
    $safeSymbol = $Symbol -replace '[\\/:*?""<>|]', '_'
    $baseName = "bt_{0}_{1}_{2}.csv" -f $safeSymbol, $Tf, $Signal
    $OutCsv = Join-Path "reports" $baseName
}

$OutCsvFull = Resolve-Path -LiteralPath (Join-Path $root $OutCsv) -ErrorAction SilentlyContinue
if (-not $OutCsvFull) {
    $OutCsvFull = Join-Path $root $OutCsv
}

Write-Host "`n[1/3] Running backtest (ML-aligned if Signal=ml)..." -ForegroundColor Cyan

$btArgs = @(
    "backtest\\runner.py",
    "--csv", $Csv,
    "--symbol", $Symbol,
    "--tf", $Tf,
    "--signal", $Signal,
    "--out", $OutCsvFull
)

Write-Host "[RUN] python $($btArgs -join ' ')" -ForegroundColor DarkGray
python @btArgs

if (-not (Test-Path $OutCsvFull)) {
    throw "[bt+tuner] Backtest trades CSV not found after run: $OutCsvFull"
}

Write-Host "`n[2/3] Validating trades CSV..." -ForegroundColor Cyan

$valArgs = @(
    "tools\\validate_trades.py",
    "--trades", $OutCsvFull
)

Write-Host "[RUN] python $($valArgs -join ' ')" -ForegroundColor DarkGray
python @valArgs

Write-Host "`n[3/3] Running AI auto-tuner on backtest trades..." -ForegroundColor Cyan

$tunerArgs = @(
    "tools\\ai_auto_tuner.py",
    "--trades-csv", $OutCsvFull,
    "--profile", "ai\\ai_profile.json",
    "--state", "ai\\auto_tuner_state.json",
    "--since-days", $SinceDays,
    "--min-trades", $MinTrades
)

if ($DryRun) {
    $tunerArgs += "--dry-run"
}

Write-Host "[RUN] python $($tunerArgs -join ' ')" -ForegroundColor DarkGray
python @tunerArgs

Write-Host "`n[bt+tuner] Done. Trades: $OutCsvFull" -ForegroundColor Green