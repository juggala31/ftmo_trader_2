# tools\start_live_phase.ps1
# Set env vars for THIS PowerShell process, then start run_live.py

# --- Disable the force-once helper (kills the threading error) ---
$env:AI_FORCE_ONCE = '0'
$env:AI_FORCE_SIDE = ''
$env:AI_FORCE_VOL  = ''

# --- Phase-driven runtime (cover multiple aliases) ---
$env:MIN_CONF             = '0.01'
$env:MIN_CONF_FLOOR       = '0.01'
$env:AI_MIN_CONF          = '0.01'
$env:SIGNAL_MIN_CONF      = '0.01'
$env:CONFIDENCE_FLOOR     = '0.01'
$env:PRED_CONF_MIN        = '0.01'

$env:COOLDOWN_S           = '60'
$env:COOLDOWN             = '60'
$env:ORDER_COOLDOWN_S     = '60'
$env:HEARTBEAT_COOLDOWN_S = '60'
$env:HB_COOLDOWN_S        = '60'
$env:GUI_COOLDOWN_S       = '60'
$env:CONTROLLER_COOLDOWN_S= '60'

$env:MAX_POSITIONS        = '5'
$env:MAX_POS              = '5'
$env:MAX_CONCURRENT       = '5'
$env:MAX_CONCURRENT_TRADES= '5'
$env:MAX_OPEN_TRADES      = '5'
$env:MAX_INFLIGHT         = '5'

# Optional: helpful for live logs
$env:PYTHONUNBUFFERED     = '1'
$env:PYTHONFAULTHANDLER   = '1'

Set-Location 'C:\ftmo_trader_2'
Write-Host "[ENV] MIN_CONF=$($env:MIN_CONF)  COOLDOWN_S=$($env:COOLDOWN_S)  MAX_POS=$($env:MAX_POSITIONS)"
python .\run_live.py