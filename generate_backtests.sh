#!/bin/bash

# Create backtests for MES, MGC, M6E, and M2K
# Based on MNQ winner strategy

SYMBOLS=("MES" "MGC" "M6E" "M2K")
DESCRIPTIONS=(
  "Micro S&P 500"
  "Micro Gold"
  "Micro Euro FX"
  "Micro Russell 2000"
)

for i in "${!SYMBOLS[@]}"; do
  SYMBOL="${SYMBOLS[$i]}"
  DESC="${DESCRIPTIONS[$i]}"
  FILENAME="backtest-topstepx-mean-reversion-${SYMBOL,,}-winner.ts"
  
  echo "Creating $FILENAME..."
  
  # Use sed to replace the default symbol and description
  sed "s/OPTIMIZED FOR: MNQ (Micro Nasdaq-100)/OPTIMIZED FOR: $SYMBOL ($DESC)/g; s/DEFAULT_MR_SYMBOL || 'MNQZ5'/DEFAULT_MR_SYMBOL || '${SYMBOL}Z5'/g" \
    backtest-topstepx-mean-reversion-mnq-winner.ts > "$FILENAME"
  
  chmod +x "$FILENAME"
  echo "✓ Created $FILENAME"
done

echo "All backtest files created successfully!"
