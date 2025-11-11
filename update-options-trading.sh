#!/bin/bash

# Update openLongPosition to accept option type
sed -i '' '914s/async function openLongPosition(symbol: string, priceContext: number, fastSma: number, squeezeOn: boolean)/async function openLongPosition(symbol: string, priceContext: number, fastSma: number, squeezeOn: boolean, optionType: '\''call'\'' | '\''put'\'' = '\''call'\'')/' run-live-sma-crossover.ts

# Update the call to openLongOptionPosition to openOptionPosition with optionType parameter
sed -i '' '921s/await openLongOptionPosition(symbol, priceContext, fastSma, null);/await openOptionPosition(symbol, priceContext, fastSma, null, optionType);/' run-live-sma-crossover.ts

# Update bullish cross to pass 'call'
sed -i '' '1344s/await openLongPosition(symbol, fastCurr, fastCurr, squeezeCurr.isOn);/await openLongPosition(symbol, fastCurr, fastCurr, squeezeCurr.isOn, '\''call'\'');/' run-live-sma-crossover.ts

# Add bearish cross entry logic after line 1345
sed -i '' '1345 a\
\
  // Bearish cross - enter PUT position\
  if (priorRelation === '\''above'\'' && currentRelation === '\''below'\'') {\
    // Only trade when squeeze is OFF (bands expanded)\
    if (!squeezeCurr || squeezeCurr.isOn) {\
      log(symbol, '\''Bearish cross detected but squeeze not expanded (OFF); skipping entry.'\'');\
      return;\
    }\
    // Determine squeeze direction from price vs BB midpoint\
    const squeezeDirection = currentPrice >= squeezeCurr.bbMidpoint ? '\''bullish'\'' : '\''bearish'\'';\
    if (squeezeDirection !== '\''bearish'\'') {\
      log(symbol, `Bearish cross detected but squeeze direction is ${squeezeDirection} (price ${currentPrice.toFixed(2)} vs BB mid ${squeezeCurr.bbMidpoint.toFixed(2)}); skipping entry.`);\
      return;\
    }\
    await openLongPosition(symbol, fastCurr, fastCurr, squeezeCurr.isOn, '\''put'\'');\
  }
' run-live-sma-crossover.ts

echo "Options trading updated successfully!"
