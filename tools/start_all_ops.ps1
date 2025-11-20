param(
    [switch]$NoMt5
)

$ROOT = "C:\ftmo_trader_2"
Set-Location $ROOT

Write-Host "[start-all] Root: $ROOT" -ForegroundColor Cyan

# -----------------------------
# 1) Start AI live (all H1)
# -----------------------------
if (Test-Path "$ROOT\tools\start_ai_all_h1.ps1") {
    $aiArgs = "-ExecutionPolicy Bypass -File `"$ROOT\tools\start_ai_all_h1.ps1`""
    Start-Process powershell -ArgumentList $aiArgs -WindowStyle Normal
    Write-Host "[start-all] Launched AI (tools\start_ai_all_h1.ps1)" -ForegroundColor Green
} else {
    Write-Host "[start-all] WARN: tools\start_ai_all_h1.ps1 not found." -ForegroundColor Yellow
}

# -----------------------------
# 2) Start health loop
# -----------------------------
if (Test-Path "$ROOT\tools\start_health_loop.ps1") {
    $healthArgs = "-ExecutionPolicy Bypass -File `"$ROOT\tools\start_health_loop.ps1`""
    if ($NoMt5) {
        $healthArgs += " -NoMt5"
    }
    Start-Process powershell -ArgumentList $healthArgs -WindowStyle Normal
    Write-Host "[start-all] Launched Health Loop (tools\start_health_loop.ps1)" -ForegroundColor Green
} else {
    Write-Host "[start-all] WARN: tools\start_health_loop.ps1 not found." -ForegroundColor Yellow
}

# -----------------------------
# 3) Print next-step helpers
# -----------------------------
Write-Host ""
Write-Host "Next manual tools (run as needed):" -ForegroundColor Yellow

Write-Host "  Daily brief (quick status snapshot):" -ForegroundColor DarkYellow
Write-Host "    powershell -ExecutionPolicy Bypass -File tools\daily_brief.ps1 -NoMt5"

Write-Host ""
Write-Host "  Forward tuner (end-of-day / once per day):" -ForegroundColor DarkYellow
Write-Host "    powershell -ExecutionPolicy Bypass -File tools\start_forward_tuner.ps1 -SinceDays 7 -MinTrades 10 -RegimeSinceDays 7 -RegimeMaxRisk 0.005"

Write-Host ""
Write-Host "[start-all] Done. AI + Health Loop are running in their own windows." -ForegroundColor Cyan