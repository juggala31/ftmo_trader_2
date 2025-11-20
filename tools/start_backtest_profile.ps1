param(
  [Parameter(Mandatory=$true)][string]$Csv,
  [Parameter(Mandatory=$true)][string]$Symbol,
  [ValidateSet("demo","live")][string]$Signal = "live"
)

$root = 'C:\ftmo_trader_2'
$bt   = Join-Path $root 'backtest'
$runner = Join-Path $bt 'runner.py'
$profile = Join-Path $bt 'symbol_costs.json'

if (!(Test-Path $runner)) { Write-Host "[ERR] Missing $runner" -ForegroundColor Red; exit 1 }
if (!(Test-Path $Csv))    { Write-Host "[ERR] Missing CSV: $Csv" -ForegroundColor Red; exit 1 }

# Load per-symbol costs
$comm = 0.0; $spr = 0.0; $slip = 0.0; $pv = 0.0
if (Test-Path $profile) {
  try {
    $map = Get-Content -Raw -Path $profile | ConvertFrom-Json
    if ($map.$Symbol) {
      $comm = [double]($map.$Symbol.commission_per_side)
      $spr  = [double]($map.$Symbol.spread_points)
      $slip = [double]($map.$Symbol.slippage_points)
      $pv   = [double]($map.$Symbol.point_value)
    }
  } catch { Write-Host "[WARN] Problem reading $profile — using zero costs" -ForegroundColor Yellow }
}

Write-Host "[BT] symbol=$Symbol  comm=$comm  spread=$spr  slip=$slip  pv=$pv"
python $runner --csv $Csv --symbol $Symbol --signal $Signal `
  --commission-per-side $comm --spread-points $spr --slippage-points $slip --point-value $pv