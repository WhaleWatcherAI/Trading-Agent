import 'dotenv/config';
import { getHistoricalTimesales } from './lib/tradier';
import { RSI, BollingerBands } from 'technicalindicators';

async function fullAnalysis() {
  console.log('🔍 Full Analysis - November 6, 2025\n');

  const bars = await getHistoricalTimesales('SPY', '2025-11-06', 15);
  console.log(`✅ Loaded ${bars.length} 15-minute bars\n`);

  const prices = bars.map(b => Number(b.close));

  // Calculate indicators
  const bbResults = BollingerBands.calculate({
    period: 20,
    stdDev: 2,
    values: prices,
  });

  const rsiResults = RSI.calculate({
    values: prices,
    period: 14,
  });

  console.log('📊 COMPLETE TECHNICAL ANALYSIS\n');
  console.log('#  | Time     | Price    | BB Upper | BB Mid   | BB Lower | BB% | RSI   | Potential Signal');
  console.log('─'.repeat(115));

  // Show ALL bars with indicators
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const price = prices[i];
    const bbIndex = i - (prices.length - bbResults.length);
    const rsiIndex = i - (prices.length - rsiResults.length);

    let bbUpper = '-', bbMiddle = '-', bbLower = '-', bbPct = '-';
    let rsi = '-';
    let signal = '';

    if (bbIndex >= 0) {
      const bb = bbResults[bbIndex];
      bbUpper = bb.upper.toFixed(2);
      bbMiddle = bb.middle.toFixed(2);
      bbLower = bb.lower.toFixed(2);

      // Calculate BB% (where price is within the bands)
      const bbRange = bb.upper - bb.lower;
      const pricePosition = (price - bb.lower) / bbRange;
      bbPct = (pricePosition * 100).toFixed(0) + '%';

      if (rsiIndex >= 0) {
        rsi = rsiResults[rsiIndex].toFixed(1);
        const rsiVal = rsiResults[rsiIndex];

        // Check entry signals
        const bbThreshold = 0.005; // 0.5%
        const lowerBBDistance = (price - bb.lower) / bb.lower;
        const upperBBDistance = (bb.upper - price) / bb.upper;

        if (lowerBBDistance <= bbThreshold && rsiVal <= 30) {
          signal = '🟢 LONG ENTRY';
        } else if (upperBBDistance <= bbThreshold && rsiVal >= 70) {
          signal = '🔴 SHORT ENTRY';
        } else if (price <= bb.lower && rsiVal > 30) {
          signal = `Below BB but RSI ${rsiVal.toFixed(1)} > 30`;
        } else if (price >= bb.upper && rsiVal < 70) {
          signal = `Above BB but RSI ${rsiVal.toFixed(1)} < 70`;
        } else if (rsiVal <= 30 && price > bb.lower * 1.005) {
          signal = `Oversold RSI but price ${((price - bb.lower) / bb.lower * 100).toFixed(2)}% above lower BB`;
        } else if (rsiVal >= 70 && price < bb.upper * 0.995) {
          signal = `Overbought RSI but price ${((bb.upper - price) / bb.upper * 100).toFixed(2)}% below upper BB`;
        }
      }
    }

    if (rsiIndex >= 0 && bbIndex < 0) {
      rsi = rsiResults[rsiIndex].toFixed(1);
    }

    const time = (bar.time.split('T')[1] || bar.time).substring(0, 8);
    console.log(
      `${String(i).padStart(2)} | ${time} | ${price.toFixed(2).padEnd(8)} | ${String(bbUpper).padEnd(8)} | ${String(bbMiddle).padEnd(8)} | ${String(bbLower).padEnd(8)} | ${String(bbPct).padEnd(3)} | ${String(rsi).padEnd(5)} | ${signal}`
    );
  }

  console.log('\n═'.repeat(115));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(115));

  const avgRSI = rsiResults.reduce((sum, val) => sum + val, 0) / rsiResults.length;
  const minRSI = Math.min(...rsiResults);
  const maxRSI = Math.max(...rsiResults);
  const minRSIIndex = rsiResults.indexOf(minRSI);
  const maxRSIIndex = rsiResults.indexOf(maxRSI);

  console.log(`\nRSI Statistics:`);
  console.log(`  Range: ${minRSI.toFixed(1)} (bar ${minRSIIndex + 14}) - ${maxRSI.toFixed(1)} (bar ${maxRSIIndex + 14})`);
  console.log(`  Average: ${avgRSI.toFixed(1)}`);

  const lastBB = bbResults[bbResults.length - 1];
  console.log(`\nBollinger Bands (final):`);
  console.log(`  Upper: $${lastBB.upper.toFixed(2)}`);
  console.log(`  Middle: $${lastBB.middle.toFixed(2)}`);
  console.log(`  Lower: $${lastBB.lower.toFixed(2)}`);
  console.log(`  Width: $${(lastBB.upper - lastBB.lower).toFixed(2)}`);

  console.log(`\n🎯 Entry Conditions (Mean Reversion Strategy):`);
  console.log(`  LONG:  Price ≤ Lower BB + 0.5% AND RSI ≤ 30`);
  console.log(`  SHORT: Price ≥ Upper BB - 0.5% AND RSI ≥ 70`);

  // Check if any bars met the criteria
  let potentialEntries = 0;
  for (let i = 20; i < bars.length; i++) {
    const price = prices[i];
    const bbIndex = i - (prices.length - bbResults.length);
    const rsiIndex = i - (prices.length - rsiResults.length);

    if (bbIndex >= 0 && rsiIndex >= 0) {
      const bb = bbResults[bbIndex];
      const rsi = rsiResults[rsiIndex];
      const lowerBBDistance = (price - bb.lower) / bb.lower;
      const upperBBDistance = (bb.upper - price) / bb.upper;

      if ((lowerBBDistance <= 0.005 && rsi <= 30) || (upperBBDistance <= 0.005 && rsi >= 70)) {
        potentialEntries++;
      }
    }
  }

  console.log(`\n✅ Result: ${potentialEntries} valid entry signals found`);
}

fullAnalysis().catch(console.error);
