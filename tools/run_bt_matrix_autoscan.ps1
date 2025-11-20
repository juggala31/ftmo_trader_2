# Auto-scan exported CSVs (SYMBOL_TF_FROM_TO.csv) and run backtest\runner.py on each.
param(
  [string]$DataRoot = "C:\ftmo_trader_2\data\export",
  [string]$OutRoot  = "C:\ftmo_trader_2\reports\bt5y",
  [ValidateSet("demo","live")][string]$Signal = "demo"
)

# Risk knobs (reuse your defaults from start_backtest.ps1)
if (-not $env:MIN_CONF)        { $env:MIN_CONF = "0.01" }
if (-not $env:POSITION_SIZING) { $env:POSITION_SIZING = "ATR_RISK" }
if (-not $env:ATR_RISK_PCT)    { $env:ATR_RISK_PCT = "0.50" }

$runner = "C:\ftmo_trader_2\backtest\runner.py"

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot "logs") | Out-Null

# Find all exported CSVs under DataRoot
$csvs = Get-ChildItem -Path $DataRoot -Recurse -File -Filter *.csv
if (-not $csvs -or $csvs.Count -eq 0) {
  Write-Host ("[BT] No CSVs under {0} - export first." -f $DataRoot) -ForegroundColor Yellow
  exit 1
}

# Regex: SYMBOL_TF_YYYY-MM-DD_YYYY-MM-DD (symbol can contain dots/underscores)
# Groups: 1=symbol  2=tf  3=from  4=to
$rx = '^(.*)_([A-Z0-9]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$'

foreach ($c in $csvs) {
  $name = [System.IO.Path]::GetFileNameWithoutExtension($c.Name)
  $m = [regex]::Match($name, $rx)
  if (-not $m.Success) {
    Write-Host ("[BT] skip unexpected filename: {0}" -f $name) -ForegroundColor Yellow
    continue
  }

  $symbol   = $m.Groups[1].Value
  $tf       = $m.Groups[2].Value
  $fromDate = $m.Groups[3].Value
  $toDate   = $m.Groups[4].Value

  $outTrades = Join-Path $OutRoot ("{0}_{1}_trades.csv" -f $symbol, $tf)
  $logDir    = Join-Path $OutRoot "logs"
  $logFile   = Join-Path $logDir ("{0}_{1}.txt" -f $symbol, $tf)

  Write-Host ("[BT] {0} {1} -> {2}" -f $symbol, $tf, $outTrades)

  $args = @(
    "--csv", $c.FullName,
    "--symbol", $symbol,
    "--signal", $Signal,
    "--cooldown-bars", "12",
    "--min-conf", $env:MIN_CONF,
    "--sizing", $env:POSITION_SIZING,
    "--atr-risk-pct", $env:ATR_RISK_PCT,
    "--out", $outTrades
  )

  & python $runner @args *>&1 | Tee-Object -FilePath $logFile
}

Write-Host ("[BT] Done. Trades CSVs in {0}, logs in {1}" -f $OutRoot, (Join-Path $OutRoot "logs"))