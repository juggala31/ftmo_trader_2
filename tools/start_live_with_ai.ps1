param(
  [switch]$NoGUI = $false,
  [switch]$DryRunAI = $false
)

$ErrorActionPreference = "Stop"

$root   = "C:\ftmo_trader_2"
$py     = (Get-Command python).Source
$live   = Join-Path $root "run_live.py"
$ailoop = "ai.alpha_loop"
$dotenv = Join-Path $root ".env"

if (-not (Test-Path $live)) { Write-Host "[ERR] Missing $live" -ForegroundColor Red; exit 1 }

function Load-DotEnv([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  Write-Host "[ENV] Loading $path"
  $lines = Get-Content -Path $path -Encoding UTF8
  foreach ($line in $lines) {
    $t = $line.Trim()
    if (-not $t) { continue }
    if ($t.StartsWith("#")) { continue }
    if ($t -notmatch "=") { continue }
    $parts = $t.Split("=",2)
    $k = $parts[0].Trim()
    $v = $parts[1].Trim()
    if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1, $v.Length-2) }
    if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1, $v.Length-2) }
    if ($k) { Set-Item -Path ("Env:{0}" -f $k) -Value $v }
  }
  return $true
}

# 1) Load .env (if present)
$loaded = Load-DotEnv $dotenv
if (-not $loaded) { Write-Host "[ENV] No .env found at $dotenv - continuing with current env" -ForegroundColor Yellow }

# 2) Optional override from flag
if ($DryRunAI) { Set-Item Env:DRY_RUN_AI "1" }

# 3) Show effective config
function _nz([string]$s, [string]$d) { if ([string]::IsNullOrEmpty($s)) { $d } else { $s } }
$cfg = "[CFG] MT5_SERVER={0}  LOGIN={1}  DRY_RUN_AI={2}  AI_LOTS={3}  AI_SL_MULT={4}  AI_TP_MULT={5}  AI_DEVIATION={6}" -f `
  (_nz $env:MT5_SERVER "n/a"), (_nz $env:MT5_LOGIN "n/a"), (_nz $env:DRY_RUN_AI "n/a"), `
  (_nz $env:AI_LOTS "n/a"), (_nz $env:AI_SL_MULT "n/a"), (_nz $env:AI_TP_MULT "n/a"), (_nz $env:AI_DEVIATION "n/a")
Write-Host $cfg

# 4) Start run_live.py
$liveArgs = @("$live")
if ($NoGUI) { $liveArgs += @("--no-gui") }
$liveProc = Start-Process -FilePath $py -ArgumentList $liveArgs -WorkingDirectory $root -PassThru -WindowStyle Normal
Start-Sleep -Milliseconds 200
Write-Host ("[RUN] run_live.py  -> PID {0}" -f $liveProc.Id)

# 5) Start AI loop (module form)
$aiArgs = @("-m", $ailoop)
$aiProc = Start-Process -FilePath $py -ArgumentList $aiArgs -WorkingDirectory $root -PassThru -WindowStyle Normal
Start-Sleep -Milliseconds 200
Write-Host ("[RUN] ai.alpha_loop -> PID {0}" -f $aiProc.Id)

# 6) Final notes
Write-Host ""
Write-Host "[OK] Both processes launched."
Write-Host ("  Live PID: {0}" -f $liveProc.Id)
Write-Host ("  AI   PID: {0}" -f $aiProc.Id)
Write-Host ""
Write-Host "Tips:"
Write-Host "  - To stop, close the two consoles or: Stop-Process -Id <LivePID>,<AiPID>"
Write-Host "  - Edit C:\ftmo_trader_2\.env to change AI_LOTS / SL/TP / deviation etc., then re-run this script."
Write-Host "  - Positions viewer:  python C:\ftmo_trader_2\tools\mt5_positions_view.py"
Write-Host "  - Fix missing SL/TP: python C:\ftmo_trader_2\tools\set_sltp_for_open_positions.py"