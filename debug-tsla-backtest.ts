import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

const SMA_PERIOD = 9;

async function debugTSLA() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  TSLA DEBUG - Detailed Trade Analysis');
  console.log('═══════════════════════════════════════════════════\n');

  const symbol = 'TSLA';
  const testDate = '2025-11-04';

  // Get historical price data
  console.log(`📊 Fetching 1-minute historical data for ${symbol}...`);
  const history = await getHistoricalData(symbol, '1min', 7, testDate);
  console.log(`✓ Got ${history.length} bars\n`);

  if (history.length < 20) {
    console.log('Sample bars:');
    history.slice(0, 10).forEach(bar => {
      console.log(`  ${bar.date}: O=${bar.open} H=${bar.high} L=${bar.low} C=${bar.close}`);
    });
  }

  // Get options chain
  console.log(`📊 Fetching options chain for ${symbol} on ${testDate}...`);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);
  console.log(`✓ Got ${optionsChain.length} options contracts\n`);

  // Show sample options
  console.log('Sample options contracts:');
  const sampleOptions = optionsChain.slice(0, 5);
  sampleOptions.forEach(opt => {
    console.log(`  ${opt.type.toUpperCase()} $${opt.strike} - Premium: $${opt.premium}, Delta: ${opt.greeks?.delta?.toFixed(3) || 'N/A'}, Theta: ${opt.greeks?.theta?.toFixed(3) || 'N/A'}`);
  });
  console.log('');

  // Calculate SMA
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  console.log('Price range:');
  console.log(`  Min: $${Math.min(...closes).toFixed(2)}`);
  console.log(`  Max: $${Math.max(...closes).toFixed(2)}`);
  console.log(`  Start: $${closes[0].toFixed(2)}`);
  console.log(`  End: $${closes[closes.length - 1].toFixed(2)}`);
  console.log(`  Change: $${(closes[closes.length - 1] - closes[0]).toFixed(2)} (${(((closes[closes.length - 1] - closes[0]) / closes[0]) * 100).toFixed(2)}%)\n`);

  // Find crossover signals
  console.log('═══════════════════════════════════════════════════');
  console.log('  CROSSOVER SIGNALS DETECTED');
  console.log('═══════════════════════════════════════════════════\n');

  let signalCount = 0;
  for (let i = SMA_PERIOD; i < Math.min(history.length, SMA_PERIOD + 50); i++) {
    const currentPrice = closes[i];
    const previousPrice = closes[i - 1];
    const currentSma = smaValues[i - SMA_PERIOD];
    const previousSma = smaValues[i - SMA_PERIOD - 1];

    const crossesUp = previousPrice <= previousSma && currentPrice > currentSma;
    const crossesDown = previousPrice >= previousSma && currentPrice < currentSma;

    if (crossesUp || crossesDown) {
      signalCount++;
      const direction = crossesUp ? 'BULLISH (Call)' : 'BEARISH (Put)';
      const optionType = crossesUp ? 'call' : 'put';

      console.log(`Signal #${signalCount} at ${history[i].date}`);
      console.log(`  Direction: ${direction}`);
      console.log(`  Price: $${currentPrice.toFixed(2)} (prev: $${previousPrice.toFixed(2)})`);
      console.log(`  SMA: $${currentSma.toFixed(2)} (prev: $${previousSma.toFixed(2)})`);

      // Find the option that would be selected
      const relevantOptions = optionsChain.filter(o => {
        if (o.type !== optionType) return false;
        if (optionType === 'call') {
          return o.strike < currentPrice;
        } else {
          return o.strike > currentPrice;
        }
      });

      if (relevantOptions.length > 0) {
        relevantOptions.sort((a, b) => {
          if (optionType === 'call') {
            return b.strike - a.strike;
          } else {
            return a.strike - b.strike;
          }
        });

        const selectedOption = relevantOptions[0];
        console.log(`  Selected Option: ${selectedOption.type.toUpperCase()} $${selectedOption.strike}`);
        console.log(`    Premium: $${selectedOption.premium.toFixed(2)}`);
        console.log(`    Delta: ${selectedOption.greeks?.delta?.toFixed(3) || 'N/A'}`);
        console.log(`    Theta: ${selectedOption.greeks?.theta?.toFixed(3) || 'N/A'}`);
        console.log(`    Volume: ${selectedOption.volume}, OI: ${selectedOption.openInterest}`);

        // Check next few bars to see what would happen
        if (i + 5 < history.length) {
          const priceIn5Bars = closes[i + 5];
          const priceMove = priceIn5Bars - currentPrice;
          console.log(`    5 bars later: $${priceIn5Bars.toFixed(2)} (${priceMove > 0 ? '+' : ''}${priceMove.toFixed(2)})`);

          if (selectedOption.greeks?.delta) {
            // Use correct delta (with sign)
            const estimatedOptionMove = priceMove * selectedOption.greeks.delta;
            const estimatedNewPrice = selectedOption.premium + estimatedOptionMove;
            console.log(`    Estimated option price: $${estimatedNewPrice.toFixed(2)} (${estimatedOptionMove > 0 ? '+' : ''}${estimatedOptionMove.toFixed(2)})`);
          }
        }
      } else {
        console.log(`  ⚠️ No suitable ${optionType} options found!`);
      }
      console.log('');

      if (signalCount >= 10) {
        console.log('... (showing first 10 signals only)\n');
        break;
      }
    }
  }

  console.log(`Total signals in first 50 bars: ${signalCount}`);
  console.log(`Total bars analyzed: ${history.length}`);
  console.log(`Estimated total signals: ~${Math.round((signalCount / Math.min(50, history.length - SMA_PERIOD)) * (history.length - SMA_PERIOD))}`);

  console.log('\n═══════════════════════════════════════════════════');
  console.log('Debug Complete!');
  console.log('═══════════════════════════════════════════════════\n');
}

debugTSLA()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Debug failed:', error);
    process.exit(1);
  });
