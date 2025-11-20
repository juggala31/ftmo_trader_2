$ErrorActionPreference = "Stop"

# Resolve repo root (.. from tools)
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ROOT = Join-Path $SCRIPT_DIR ".." | Resolve-Path
Set-Location $ROOT

Write-Host ""
Write-Host "======================================" -ForegroundColor Red
Write-Host "        PANIC BUTTON ACTIVATED        " -ForegroundColor Red
Write-Host "======================================" -ForegroundColor Red
Write-Host ""

# 1) Flatten all MT5 positions (best effort)
Write-Host "[PANIC] Step 1: Flattening all MT5 positions via tools\mt5_flatten_all.py..." -ForegroundColor Yellow
try {
    python .\tools\mt5_flatten_all.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[PANIC][WARN] mt5_flatten_all.py exited with code $LASTEXITCODE" -ForegroundColor DarkYellow
    } else {
        Write-Host "[PANIC] mt5_flatten_all.py completed." -ForegroundColor Green
    }
} catch {
    Write-Host "[PANIC][ERROR] Failed to run mt5_flatten_all.py: $_" -ForegroundColor Red
}

# 2) Disable all symbols in ai_profile.json
Write-Host ""
Write-Host "[PANIC] Step 2: Disabling all symbols in ai\ai_profile.json..." -ForegroundColor Yellow
try {
    python .\tools\panic_button.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[PANIC][ERROR] panic_button.py exited with code $LASTEXITCODE" -ForegroundColor Red
    } else {
        Write-Host "[PANIC] ai_profile.json updated (all symbols OFF)." -ForegroundColor Green
    }
} catch {
    Write-Host "[PANIC][ERROR] Failed to run panic_button.py: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "[PANIC] Completed." -ForegroundColor Green
Write-Host "[PANIC] All open positions should be closed (if flatten succeeded)," -ForegroundColor Green
Write-Host "        and no NEW trades should open because all symbols are OFF in ai_profile.json." -ForegroundColor Green
Write-Host "        To resume trading, restore a backup profile or re-enable symbols and restart run_live.py." -ForegroundColor Green