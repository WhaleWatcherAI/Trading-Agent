import { SMA } from 'technicalindicators';
import { getHistoricalData } from './lib/technicals';
import { getOptionsChain } from './lib/tradier';

const SMA_PERIOD = 9;

async function testSingleTrade() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  SINGLE TRADE TEST - Hold First Signal All Day');
  console.log('═══════════════════════════════════════════════════\n');

  const symbol = 'TSLA';
  const testDate = '2025-11-04';

  const history = await getHistoricalData(symbol, '1min', 7, testDate);
  const optionsChain = await getOptionsChain(symbol, undefined, testDate);

  console.log(`Total bars: ${history.length}`);
  console.log(`Options available: ${optionsChain.length}\n`);

  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: SMA_PERIOD });

  console.log(`Opening price: $${closes[0].toFixed(2)}`);
  console.log(`Closing price: $${closes[closes.length - 1].toFixed(2)}`);
  console.log(`Daily change: $${(closes[closes.length - 1] - closes[0]).toFixed(2)} (${(((closes[closes.length - 1] - closes[0]) / closes[0]) * 100).toFixed(2)}%)\n`);

  // Find first signal
  for (let i = SMA_PERIOD; i < history.length; i++) {
    const currentPrice = closes[i];
    const previousPrice = closes[i - 1];
    const currentSma = smaValues[i - SMA_PERIOD];
    const previousSma = smaValues[i - SMA_PERIOD - 1];

    const crossesUp = previousPrice <= previousSma && currentPrice > currentSma;
    const crossesDown = previousPrice >= previousSma && currentPrice < currentSma;

    if (crossesUp || crossesDown) {
      const direction = crossesUp ? 'BULLISH (Call)' : 'BEARISH (Put)';
      const optionType = crossesUp ? 'call' : 'put';

      console.log('═══════════════════════════════════════════════════');
      console.log(`FIRST SIGNAL: ${direction}`);
      console.log('═══════════════════════════════════════════════════');
      console.log(`Time: ${history[i].date} (bar ${i} of ${history.length})`);
      console.log(`Entry Price: $${currentPrice.toFixed(2)}`);
      console.log(`SMA: $${currentSma.toFixed(2)}\n`);

      // Find option
      const relevantOptions = optionsChain.filter(o => {
        if (o.type !== optionType) return false;
        if (optionType === 'call') {
          return o.strike < currentPrice;
        } else {
          return o.strike > currentPrice;
        }
      });

      relevantOptions.sort((a, b) => {
        if (optionType === 'call') {
          return b.strike - a.strike;
        } else {
          return a.strike - b.strike;
        }
      });

      const selectedOption = relevantOptions[0];

      console.log(`Selected Option: ${selectedOption.type.toUpperCase()} $${selectedOption.strike}`);
      console.log(`Entry Premium: $${selectedOption.premium.toFixed(2)}`);
      console.log(`Delta: ${selectedOption.greeks?.delta?.toFixed(3) || 'N/A'}`);
      console.log(`Theta: ${selectedOption.greeks?.theta?.toFixed(3) || 'N/A'}\n`);

      // Simulate holding until end of day
      const finalPrice = closes[closes.length - 1];
      const priceMove = finalPrice - currentPrice;

      console.log('═══════════════════════════════════════════════════');
      console.log('HOLDING UNTIL END OF DAY');
      console.log('═══════════════════════════════════════════════════');
      console.log(`Exit Price: $${finalPrice.toFixed(2)}`);
      console.log(`Underlying Move: $${priceMove.toFixed(2)} (${((priceMove / currentPrice) * 100).toFixed(2)}%)\n`);

      if (selectedOption.greeks?.delta) {
        const barsHeld = history.length - i;
        const minutesHeld = barsHeld;
        const optionMove = priceMove * selectedOption.greeks.delta;
        const thetaDecay = (selectedOption.greeks.theta || 0) * (minutesHeld / 390);

        const exitPremium = selectedOption.premium + optionMove + thetaDecay;

        console.log(`Option Move from Delta: $${optionMove.toFixed(2)}`);
        console.log(`Theta Decay (${minutesHeld} min): $${thetaDecay.toFixed(2)}`);
        console.log(`Exit Premium: $${Math.max(0.01, exitPremium).toFixed(2)}\n`);

        const grossPnL = exitPremium - selectedOption.premium;
        const commission = 1.30; // $0.65 × 2
        const slippage = (selectedOption.premium * 0.005) + (exitPremium * 0.005);
        const netPnL = grossPnL - commission - slippage;

        console.log('P&L BREAKDOWN:');
        console.log(`  Gross P&L: $${grossPnL.toFixed(2)}`);
        console.log(`  Commission: -$${commission.toFixed(2)}`);
        console.log(`  Slippage: -$${slippage.toFixed(2)}`);
        console.log(`  Net P&L: $${netPnL.toFixed(2)} (${((netPnL / selectedOption.premium) * 100).toFixed(1)}%)`);

        if (netPnL > 0) {
          console.log(`\n✅ WINNER! Made $${netPnL.toFixed(2)}`);
        } else {
          console.log(`\n❌ LOSER. Lost $${Math.abs(netPnL).toFixed(2)}`);
        }
      }

      break;
    }
  }

  console.log('\n═══════════════════════════════════════════════════');
}

testSingleTrade()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Test failed:', error);
    process.exit(1);
  });
