param(
  [string]$Csv = "C:\ftmo_trader_2\data\sample_ohlc.csv",
  [string]$Symbol = "US30USD",
  [ValidateSet("demo","live")][string]$Signal = "demo"
)
if (-not $env:MIN_CONF)        { $env:MIN_CONF = '0.01' }
if (-not $env:POSITION_SIZING) { $env:POSITION_SIZING = 'ATR_RISK' }
if (-not $env:ATR_RISK_PCT)    { $env:ATR_RISK_PCT = '0.50' }

Write-Host "[BT] CSV=$Csv  SYMBOL=$Symbol  SIGNAL=$Signal"
python "C:\ftmo_trader_2\backtest\runner.py" `
  --csv "$Csv" `
  --symbol "$Symbol" `
  --signal $Signal `
  --cooldown-bars 12 `
  --min-conf $env:MIN_CONF `
  --sizing $env:POSITION_SIZING `
  --atr-risk-pct $env:ATR_RISK_PCT `
  --out "C:\ftmo_trader_2\reports\backtest_trades.csv"