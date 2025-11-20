param(
    [int]$SinceDays = 365,
    [int]$MinTrades = 50,
    [ValidateSet("demo","ml","live")]
    [string]$Signal = "ml",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $here "..")
Set-Location $root

Write-Host "[batch-bt+tuner] Root: $root"
Write-Host "[batch-bt+tuner] SinceDays=$SinceDays MinTrades=$MinTrades Signal=$Signal DryRun=$DryRun"

# List of H1 backtest datasets and their symbols
$items = @(
    @{ Symbol = "US30Z25.sim"; Csv = "ai\\datasets\\csv\\H1\\US30Z25.sim.csv" },
    @{ Symbol = "US100Z25.sim"; Csv = "ai\\datasets\\csv\\H1\\US100Z25.sim.csv" },
    @{ Symbol = "US500Z25.sim"; Csv = "ai\\datasets\\csv\\H1\\US500Z25.sim.csv" },
    @{ Symbol = "XAUZ25.sim";   Csv = "ai\\datasets\\csv\\H1\\XAUZ25.sim.csv"   },
    @{ Symbol = "USOILZ25.sim"; Csv = "ai\\datasets\\csv\\H1\\USOILZ25.sim.csv" },
    @{ Symbol = "BTCX25.sim";   Csv = "ai\\datasets\\csv\\H1\\BTCX25.sim.csv"   }
)

$runner = Join-Path $root "tools\\run_bt_and_tune.ps1"

foreach ($item in $items) {
    $sym = $item.Symbol
    $csv = $item.Csv

    Write-Host "`n===============================================" -ForegroundColor Yellow
    Write-Host "[batch-bt+tuner] Symbol: $sym" -ForegroundColor Yellow
    Write-Host "CSV: $csv"

    if (-not (Test-Path $csv)) {
        Write-Host "[SKIP] CSV not found: $csv" -ForegroundColor Red
        continue
    }

    $args = @(
        "-Csv", $csv,
        "-Symbol", $sym,
        "-Tf", "H1",
        "-SinceDays", $SinceDays,
        "-MinTrades", $MinTrades,
        "-Signal", $Signal
    )

    if ($DryRun) {
        $args += "-DryRun"
    }

    Write-Host "[CALL] tools\\run_bt_and_tune.ps1 $($args -join ' ')" -ForegroundColor DarkGray
    powershell -ExecutionPolicy Bypass -File $runner @args
}

Write-Host "`n[batch-bt+tuner] Complete." -ForegroundColor Green