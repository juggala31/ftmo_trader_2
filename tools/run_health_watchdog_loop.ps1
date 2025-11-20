param(
    [int]$IntervalSec = 300,
    [switch]$NoMt5
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root = Resolve-Path (Join-Path $here "..")
Set-Location $root

$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$logPath = Join-Path $logDir "health_watchdog.log"

Write-Host "[health-loop] Root: $root"
Write-Host "[health-loop] Log:  $logPath"
Write-Host "[health-loop] Interval: $IntervalSec sec"
if ($NoMt5) {
    Write-Host "[health-loop] MT5 check: DISABLED (--no-mt5)"
} else {
    Write-Host "[health-loop] MT5 check: ENABLED"
}
Write-Host ""

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ""
    Write-Host "[$ts] Running health_watchdog..." -ForegroundColor Cyan

    $args = @("tools\health_watchdog.py")
    if ($NoMt5) {
        $args += "--no-mt5"
    }

    # Capture output
    $output = & python @args 2>&1
    $exitCode = $LASTEXITCODE

    # Write to console
    $output | ForEach-Object { Write-Host $_ }

    # Append to log with timestamp prefix
    $linesToLog = $output | ForEach-Object { "[$ts] $_" }
    $linesToLog | Add-Content -Path $logPath -Encoding UTF8

    # Status line based on exit code
    switch ($exitCode) {
        0 {
            Write-Host "[$ts] STATUS: OK" -ForegroundColor Green
        }
        1 {
            Write-Host "[$ts] STATUS: STALE (check soon)" -ForegroundColor Yellow
        }
        2 {
            Write-Host "[$ts] STATUS: BAD (attention needed!)" -ForegroundColor Red
        }
        default {
            Write-Host "[$ts] STATUS: UNKNOWN exit code=$exitCode" -ForegroundColor Magenta
        }
    }

    Write-Host "[health-loop] Sleeping $IntervalSec seconds..."
    Start-Sleep -Seconds $IntervalSec
}