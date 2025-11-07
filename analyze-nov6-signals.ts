import 'dotenv/config';
import { getHistoricalTimesales } from './lib/tradier';
import { RSI, BollingerBands } from 'technicalindicators';

async function analyzeSignals() {
  console.log('🔍 Analyzing November 6, 2025 Trading Signals\n');

  const bars = await getHistoricalTimesales('SPY', '2025-11-06', 15);
  console.log(`✅ Loaded ${bars.length} 15-minute bars\n`);

  if (bars.length < 20) {
    console.log('❌ Not enough bars for technical analysis');
    return;
  }

  const prices = bars.map(b => Number(b.close));

  // Calculate Bollinger Bands (20-period, 2 std dev)
  const bbInput = {
    period: 20,
    stdDev: 2,
    values: prices,
  };
  const bbResults = BollingerBands.calculate(bbInput);

  // Calculate RSI (14-period)
  const rsiResults = RSI.calculate({
    values: prices,
    period: 14,
  });

  console.log('📊 TECHNICAL ANALYSIS - 15-MINUTE BARS\n');
  console.log('Time         | Price    | BB Upper | BB Middle | BB Lower | RSI   | Signal');
  console.log('─'.repeat(95));

  // Start from bar 20 where we have all indicators
  for (let i = 20; i < bars.length; i++) {
    const bar = bars[i];
    const price = prices[i];
    const bbIndex = i - (prices.length - bbResults.length);
    const rsiIndex = i - (prices.length - rsiResults.length);

    if (bbIndex >= 0 && rsiIndex >= 0) {
      const bb = bbResults[bbIndex];
      const rsi = rsiResults[rsiIndex];

      const time = bar.time.split('T')[1] || bar.time;

      // Check mean reversion signals
      // LONG signal: price at/below lower BB AND RSI <= 30
      // SHORT signal: price at/above upper BB AND RSI >= 70
      let signal = '';
      const bbThreshold = 0.005; // 0.5% threshold

      const lowerBBDistance = (price - bb.lower) / bb.lower;
      const upperBBDistance = (bb.upper - price) / bb.upper;

      if (lowerBBDistance <= bbThreshold && rsi <= 30) {
        signal = '🔵 LONG ENTRY';
      } else if (upperBBDistance <= bbThreshold && rsi >= 70) {
        signal = '🔴 SHORT ENTRY';
      } else if (price <= bb.lower) {
        signal = `near lower BB (RSI ${rsi.toFixed(1)} not ≤30)`;
      } else if (price >= bb.upper) {
        signal = `near upper BB (RSI ${rsi.toFixed(1)} not ≥70)`;
      } else if (rsi <= 30) {
        signal = `oversold RSI (not at BB)`;
      } else if (rsi >= 70) {
        signal = `overbought RSI (not at BB)`;
      }

      console.log(
        `${time.padEnd(12)} | ${price.toFixed(2).padEnd(8)} | ${bb.upper.toFixed(2).padEnd(8)} | ${bb.middle.toFixed(2).padEnd(9)} | ${bb.lower.toFixed(2).padEnd(8)} | ${rsi.toFixed(1).padEnd(5)} | ${signal}`
      );
    }
  }

  // Summary statistics
  const lastBB = bbResults[bbResults.length - 1];
  const avgRSI = rsiResults.reduce((sum, val) => sum + val, 0) / rsiResults.length;
  const minRSI = Math.min(...rsiResults);
  const maxRSI = Math.max(...rsiResults);

  console.log('\n');
  console.log('═'.repeat(95));
  console.log('📈 SUMMARY');
  console.log('═'.repeat(95));
  console.log(`RSI Range: ${minRSI.toFixed(1)} - ${maxRSI.toFixed(1)} (Avg: ${avgRSI.toFixed(1)})`);
  console.log(`Final BB: Lower ${lastBB.lower.toFixed(2)} | Middle ${lastBB.middle.toFixed(2)} | Upper ${lastBB.upper.toFixed(2)}`);
  console.log(`\nEntry Conditions:`);
  console.log(`  LONG:  Price ≤ BB Lower + 0.5% threshold AND RSI ≤ 30`);
  console.log(`  SHORT: Price ≥ BB Upper - 0.5% threshold AND RSI ≥ 70`);
}

analyzeSignals().catch(console.error);
