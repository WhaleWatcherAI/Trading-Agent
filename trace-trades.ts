import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

const SMA_PERIOD = 9;
const COMMISSION = 1.30;
const SLIPPAGE_PERCENT = 0.5;

async function traceFirstTrades() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  DETAILED TRADE TRACE - First 5 Trades');
  console.log('═══════════════════════════════════════════════════\n');

  const symbol = 'TSLA';
  const testDate = '2025-11-04';

  const history = await getHistoricalData(symbol, '1min', 7, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  console.log(`Total bars: ${history.length}`);
  console.log(`Start price: $${closes[0].toFixed(2)}`);
  console.log(`End price: $${closes[closes.length - 1].toFixed(2)}`);
  console.log(`Daily move: +$${(closes[closes.length - 1] - closes[0]).toFixed(2)}\n`);

  let tradeCount = 0;
  let inTrade = false;
  let entryBar = 0;
  let entryPrice = 0;
  let entrySma = 0;
  let entryType: 'call' | 'put' = 'call';
  let entryStrike = 0;
  let entryDelta = 0;
  let entryTheta = 0;
  let entryOptionPrice = 0;
  let entryUnderlyingPrice = 0;

  for (let i = SMA_PERIOD; i < history.length && tradeCount < 5; i++) {
    const bar = history[i];
    const prevClose = closes[i - 1];
    const currentClose = closes[i];
    const sma = smaValues[i - SMA_PERIOD];
    const prevSma = smaValues[i - SMA_PERIOD - 1];

    // Entry logic
    if (!inTrade) {
      const crossesUp = prevClose <= prevSma && bar.high > sma;
      const crossesDown = prevClose >= prevSma && bar.low < sma;

      if (crossesUp || crossesDown) {
        tradeCount++;
        inTrade = true;
        entryBar = i;
        entrySma = sma;
        entryType = crossesUp ? 'call' : 'put';

        console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
        console.log(`TRADE #${tradeCount} - ${entryType.toUpperCase()} ENTRY`);
        console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
        console.log(`Time: ${bar.date}`);
        console.log(`Bar: O=$${bar.open.toFixed(2)} H=$${bar.high.toFixed(2)} L=$${bar.low.toFixed(2)} C=$${bar.close.toFixed(2)}`);
        console.log(`Prev Close: $${prevClose.toFixed(2)}`);
        console.log(`SMA: $${sma.toFixed(2)} (prev: $${prevSma.toFixed(2)})`);
        console.log(`Signal: ${crossesUp ? 'Price crossed ABOVE SMA' : 'Price crossed BELOW SMA'}`);

        // Entry at SMA level
        entryUnderlyingPrice = crossesUp
          ? (bar.open > sma ? bar.open : sma)
          : (bar.open < sma ? bar.open : sma);

        console.log(`Entry underlying price: $${entryUnderlyingPrice.toFixed(2)}`);

        // Find option
        const relevantOptions = optionsChain.filter(o => {
          if (o.type !== entryType) return false;
          if (entryType === 'call') {
            return o.strike < entryUnderlyingPrice;
          } else {
            return o.strike > entryUnderlyingPrice;
          }
        });

        relevantOptions.sort((a, b) => {
          if (entryType === 'call') {
            return b.strike - a.strike;
          } else {
            return a.strike - b.strike;
          }
        });

        const option = relevantOptions[0];
        entryStrike = option.strike;
        entryDelta = option.greeks?.delta || 0;
        entryTheta = option.greeks?.theta || 0;

        // Adjust option price to entry point
        const priceDiff = entryUnderlyingPrice - currentClose;
        const premiumAdj = entryDelta * priceDiff;
        const adjustedPremium = option.premium - premiumAdj;
        entryOptionPrice = adjustedPremium * (1 + SLIPPAGE_PERCENT / 100);

        console.log(`\nSelected Option: ${entryType.toUpperCase()} $${entryStrike}`);
        console.log(`  Option premium at close: $${option.premium.toFixed(2)}`);
        console.log(`  Price diff (entry vs close): $${priceDiff.toFixed(2)}`);
        console.log(`  Premium adjustment: $${premiumAdj.toFixed(2)}`);
        console.log(`  Adjusted premium: $${adjustedPremium.toFixed(2)}`);
        console.log(`  Entry price (with slippage): $${entryOptionPrice.toFixed(2)}`);
        console.log(`  Delta: ${entryDelta.toFixed(3)}, Theta: ${entryTheta.toFixed(3)}`);

        entryPrice = currentClose;
      }
    }
    // Exit logic
    else {
      const oppositeCross = entryType === 'call'
        ? (prevClose >= prevSma && bar.low < sma)
        : (prevClose <= prevSma && bar.high > sma);

      if (oppositeCross || i === history.length - 1) {
        const barsHeld = i - entryBar;
        const exitReason = oppositeCross ? 'OPPOSITE CROSSOVER' : 'END OF DAY';

        console.log(`\n${exitReason} - EXIT`);
        console.log(`Time: ${bar.date}`);
        console.log(`Bars held: ${barsHeld}`);
        console.log(`Bar: O=$${bar.open.toFixed(2)} H=$${bar.high.toFixed(2)} L=$${bar.low.toFixed(2)} C=$${bar.close.toFixed(2)}`);
        console.log(`SMA: $${sma.toFixed(2)}`);

        // Calculate option exit price
        const underlyingMove = currentClose - entryUnderlyingPrice;
        const optionMoveFromDelta = underlyingMove * entryDelta;
        const thetaDecay = entryTheta * (barsHeld / 390);
        const exitOptionPrice = (entryOptionPrice + optionMoveFromDelta + thetaDecay) * (1 - SLIPPAGE_PERCENT / 100);

        console.log(`\nUnderlying: $${entryUnderlyingPrice.toFixed(2)} → $${currentClose.toFixed(2)} (${underlyingMove > 0 ? '+' : ''}$${underlyingMove.toFixed(2)})`);
        console.log(`Option move from delta: ${entryDelta.toFixed(3)} × $${underlyingMove.toFixed(2)} = $${optionMoveFromDelta.toFixed(2)}`);
        console.log(`Theta decay: ${entryTheta.toFixed(3)} × ${barsHeld}/390 = $${thetaDecay.toFixed(2)}`);
        console.log(`Exit option price (before slippage): $${(entryOptionPrice + optionMoveFromDelta + thetaDecay).toFixed(2)}`);
        console.log(`Exit option price (with slippage): $${Math.max(0.01, exitOptionPrice).toFixed(2)}`);

        const grossPnL = Math.max(0.01, exitOptionPrice) - entryOptionPrice;
        const netPnL = grossPnL - COMMISSION;

        console.log(`\nP&L BREAKDOWN:`);
        console.log(`  Entry: $${entryOptionPrice.toFixed(2)}`);
        console.log(`  Exit: $${Math.max(0.01, exitOptionPrice).toFixed(2)}`);
        console.log(`  Gross P&L: $${grossPnL.toFixed(2)}`);
        console.log(`  Commission: -$${COMMISSION.toFixed(2)}`);
        console.log(`  Net P&L: $${netPnL.toFixed(2)} (${((netPnL / entryOptionPrice) * 100).toFixed(1)}%)`);

        if (netPnL > 0) {
          console.log(`  ✅ WINNER`);
        } else {
          console.log(`  ❌ LOSER`);
        }

        inTrade = false;
      }
    }
  }

  console.log('\n═══════════════════════════════════════════════════\n');
}

traceFirstTrades()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Trace failed:', error);
    process.exit(1);
  });
