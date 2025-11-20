param(
    [int]$IntervalSec = 120,
    [switch]$NoMt5
)

$ROOT   = Split-Path $PSScriptRoot -Parent
$logDir = Join-Path $ROOT "logs"
$LOG    = Join-Path $logDir "health_watchdog.log"

if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if ($NoMt5) { $mt5Status = "DISABLED" } else { $mt5Status = "ENABLED" }

Write-Host "[health-loop] Root: $ROOT"
Write-Host "[health-loop] Log:  $LOG"
Write-Host "[health-loop] Interval: $IntervalSec sec"
Write-Host "[health-loop] MT5 check: $mt5Status"
Write-Host ""

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] Running health_watchdog..."

    $args = @()
    if ($NoMt5) { $args += "--no-mt5" }

    $cmdLine = "python tools\health_watchdog.py " + ($args -join " ")
    Add-Content -Path $LOG -Value "[$ts] $cmdLine"

    $output = & python tools\health_watchdog.py @args 2>&1
    $output | Tee-Object -FilePath $LOG -Append | Out-Host

    $statusLine = $output | Select-String "\[health_watchdog\] Overall:" | Select-Object -Last 1
    if ($statusLine) {
        $statusText = $statusLine.ToString()
        Write-Host "[$ts] STATUS: $statusText"
        Add-Content -Path $LOG -Value "[$ts] STATUS: $statusText"
    }

    Write-Host "[health-loop] Sleeping $IntervalSec seconds..."
    Add-Content -Path $LOG -Value "[$ts] Sleeping $IntervalSec seconds..."
    Start-Sleep -Seconds $IntervalSec
}