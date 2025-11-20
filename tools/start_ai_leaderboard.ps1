$ROOT = "C:\ftmo_trader_2"
Set-Location $ROOT

$csv   = Join-Path $ROOT "data\trades.csv"
$out   = Join-Path $ROOT "reports\ai_leaderboard"

# Optional: limit to last N days of trades. Uncomment to use.
# $sinceDays = 7
# $sinceArg  = "--since-days $sinceDays"
# Or leave blank to use all history:
$sinceArg = ""

Write-Host "[RUN] python tools\ai_leaderboard.py --csv $csv --out-dir $out $sinceArg" -ForegroundColor Cyan
python tools\ai_leaderboard.py --csv $csv --out-dir $out $sinceArg