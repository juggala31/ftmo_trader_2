param(
    [string]$Login  = "1600046554",      # change if your MT5 login is different
    [string]$Server = "OANDA-Demo-1"     # change if your MT5 server is different
)

$ROOT = "C:\ftmo_trader_2"
Set-Location $ROOT

# Ask for MT5 password (hidden input)
$securePwd = Read-Host "Enter MT5 password" -AsSecureString
$pwdPtr    = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePwd)
$Password  = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pwdPtr)

# Core MT5 env for guardian
$env:APP_MODE     = "live"
$env:MT5_LOGIN    = $Login
$env:MT5_PASSWORD = $Password
$env:MT5_SERVER   = $Server

# --- Guardian on + loop ---
$env:AI_GUARDIAN_ENABLED = "1"
$env:AI_GUARD_LOOP_SEC   = "5"
$env:AI_GUARDIAN_VERBOSE = "1"   # set to "0" later if logs too noisy

# --- PRODUCTION THRESHOLDS ---

# Breakeven: trigger at +1.0R, keep +0.10R cushion
$env:AI_GUARD_BE_TRIGGER_R = "1.0"
$env:AI_GUARD_BE_OFFSET_R  = "0.10"

# Trailing: start at +1.5R, trail = max(1×ATR_fast, 0.75R)
$env:AI_GUARD_TRAIL_TRIGGER_R = "1.5"
$env:AI_GUARD_TRAIL_ATR_MULT  = "1.0"
$env:AI_GUARD_TRAIL_MIN_R     = "0.75"

# Panic: early big loss at -1.25R within first 1h
$env:AI_GUARD_PANIC_LOSS_R    = "1.25"
$env:AI_GUARD_PANIC_EARLY_SEC = "3600"   # 1 hour

# Zombie: treat trades as dead after 6h if near flat (-0.5R .. +0.5R)
$env:AI_GUARD_ZOMBIE_AGE_SEC = "21600"   # 6 hours
$env:AI_GUARD_ZOMBIE_NEAR_R  = "0.50"

# ATR timeframe for trailing
$env:AI_GUARD_ATR_TF      = "M15"
$env:AI_GUARD_ATR_PERIOD  = "14"

# For safety: keep DRY_RUN on unless explicitly turned off
if (-not $env:DRY_RUN_AI) {
    $env:DRY_RUN_AI = "1"
}

Write-Host "[RUN] python -m ai.guardian_entry (PRODUCTION thresholds)" -ForegroundColor Cyan
python -m ai.guardian_entry