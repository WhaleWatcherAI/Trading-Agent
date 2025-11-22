#!/bin/bash

# Run all ICT backtests on micro contracts (MNQ, MES, MGC, M6E)
# Date range: Last 3 months for meaningful results

START_DATE="2025-08-01T00:00:00Z"
END_DATE="2025-11-11T00:00:00Z"
SYMBOLS=("MNQZ5" "MESZ5" "MGCZ5" "M6EZ5")
SYMBOL_NAMES=("Micro Nasdaq" "Micro S&P 500" "Micro Gold" "Micro Euro")

echo "================================================================================"
echo "ICT STRATEGIES BACKTEST - ALL MICRO CONTRACTS"
echo "================================================================================"
echo "Date Range: $START_DATE -> $END_DATE"
echo "Symbols: ${SYMBOLS[@]}"
echo ""

# Strategy #1: Liquidity-Sweep + FVG Return
echo "################################################################################"
echo "STRATEGY #1: LIQUIDITY-SWEEP + FVG RETURN"
echo "################################################################################"
echo ""

for i in "${!SYMBOLS[@]}"; do
  symbol="${SYMBOLS[$i]}"
  name="${SYMBOL_NAMES[$i]}"

  echo "------------------------------------------------------------"
  echo "Running on $symbol ($name)..."
  echo "------------------------------------------------------------"

  ICT_SWEEP_SYMBOL=$symbol \
  ICT_SWEEP_START=$START_DATE \
  ICT_SWEEP_END=$END_DATE \
  npx tsx backtest-ict-liquidity-sweep-fvg.ts 2>&1 | \
  grep -E "BACKTEST SUMMARY|Total Trades|Win Rate|Net Realized PnL|Profit Factor|Max Drawdown|Exit Reasons"

  echo ""
done

# Strategy #2: BOS/CHOCH + FVG
echo ""
echo "################################################################################"
echo "STRATEGY #2: BOS/CHOCH + FVG (TREND-FOLLOWING)"
echo "################################################################################"
echo ""

for i in "${!SYMBOLS[@]}"; do
  symbol="${SYMBOLS[$i]}"
  name="${SYMBOL_NAMES[$i]}"

  echo "------------------------------------------------------------"
  echo "Running on $symbol ($name)..."
  echo "------------------------------------------------------------"

  ICT_BOS_SYMBOL=$symbol \
  ICT_BOS_START=$START_DATE \
  ICT_BOS_END=$END_DATE \
  npx tsx backtest-ict-bos-choch-fvg.ts 2>&1 | \
  grep -E "BACKTEST SUMMARY|Total Trades|Win Rate|Net Realized PnL|Profit Factor|Max Drawdown|Exit Reasons"

  echo ""
done

# Strategy #3: Power of Three (PO3)
echo ""
echo "################################################################################"
echo "STRATEGY #3: POWER OF THREE (PO3) LITE"
echo "################################################################################"
echo ""

for i in "${!SYMBOLS[@]}"; do
  symbol="${SYMBOLS[$i]}"
  name="${SYMBOL_NAMES[$i]}"

  echo "------------------------------------------------------------"
  echo "Running on $symbol ($name)..."
  echo "------------------------------------------------------------"

  ICT_PO3_SYMBOL=$symbol \
  ICT_PO3_START=$START_DATE \
  ICT_PO3_END=$END_DATE \
  npx tsx backtest-ict-po3-lite.ts 2>&1 | \
  grep -E "BACKTEST SUMMARY|Total Trades|Win Rate|Net Realized PnL|Profit Factor|Max Drawdown|Exit Reasons"

  echo ""
done

echo "================================================================================"
echo "ALL BACKTESTS COMPLETE"
echo "================================================================================"
