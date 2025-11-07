import 'dotenv/config';
import { getHistoricalTimesales } from './lib/tradier';
import { RSI, BollingerBands } from 'technicalindicators';

async function analyze5MinBars() {
  console.log('🔍 November 6, 2025 - 5-MINUTE BAR ANALYSIS\n');

  const bars = await getHistoricalTimesales('SPY', '2025-11-06', 5);
  console.log(`✅ Loaded ${bars.length} 5-minute bars\n`);

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

  console.log('📊 5-MINUTE BARS WITH SIGNALS\n');
  console.log('#   | Time     | Price    | BB Upper | BB Mid   | BB Lower | RSI   | Signal');
  console.log('─'.repeat(105));

  const signals = [];

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const price = prices[i];
    const bbIndex = i - (prices.length - bbResults.length);
    const rsiIndex = i - (prices.length - rsiResults.length);

    let bbUpper = '-', bbMiddle = '-', bbLower = '-';
    let rsi = '-';
    let signal = '';

    if (bbIndex >= 0 && rsiIndex >= 0) {
      const bb = bbResults[bbIndex];
      const rsiVal = rsiResults[rsiIndex];

      bbUpper = bb.upper.toFixed(2);
      bbMiddle = bb.middle.toFixed(2);
      bbLower = bb.lower.toFixed(2);
      rsi = rsiVal.toFixed(1);

      // Check entry signals
      const bbThreshold = 0.005; // 0.5%
      const lowerBBDistance = (price - bb.lower) / bb.lower;
      const upperBBDistance = (bb.upper - price) / bb.upper;

      if (lowerBBDistance <= bbThreshold && rsiVal <= 30) {
        signal = '🟢 LONG ENTRY ✅';
        signals.push({
          time: bar.time,
          type: 'LONG',
          price,
          rsi: rsiVal,
          bb,
        });
      } else if (upperBBDistance <= bbThreshold && rsiVal >= 70) {
        signal = '🔴 SHORT ENTRY ✅';
        signals.push({
          time: bar.time,
          type: 'SHORT',
          price,
          rsi: rsiVal,
          bb,
        });
      } else if (price <= bb.lower && rsiVal > 30) {
        signal = `Below BB, RSI ${rsiVal.toFixed(1)}`;
      } else if (price >= bb.upper && rsiVal < 70) {
        signal = `Above BB, RSI ${rsiVal.toFixed(1)}`;
      } else if (rsiVal <= 30) {
        signal = `RSI oversold, ${((price - bb.lower) / bb.lower * 100).toFixed(2)}% above lower BB`;
      } else if (rsiVal >= 70) {
        signal = `RSI overbought, ${((bb.upper - price) / bb.upper * 100).toFixed(2)}% below upper BB`;
      }
    }

    const time = (bar.time.split('T')[1] || bar.time).substring(0, 8);
    console.log(
      `${String(i).padStart(3)} | ${time} | ${price.toFixed(2).padEnd(8)} | ${String(bbUpper).padEnd(8)} | ${String(bbMiddle).padEnd(8)} | ${String(bbLower).padEnd(8)} | ${String(rsi).padEnd(5)} | ${signal}`
    );
  }

  console.log('\n═'.repeat(105));
  console.log('📊 SUMMARY');
  console.log('═'.repeat(105));

  const avgRSI = rsiResults.reduce((sum, val) => sum + val, 0) / rsiResults.length;
  const minRSI = Math.min(...rsiResults);
  const maxRSI = Math.max(...rsiResults);

  console.log(`\nRSI Statistics:`);
  console.log(`  Range: ${minRSI.toFixed(1)} - ${maxRSI.toFixed(1)}`);
  console.log(`  Average: ${avgRSI.toFixed(1)}`);

  if (bbResults.length > 0) {
    const lastBB = bbResults[bbResults.length - 1];
    console.log(`\nBollinger Bands (final):`);
    console.log(`  Upper: $${lastBB.upper.toFixed(2)}`);
    console.log(`  Middle: $${lastBB.middle.toFixed(2)}`);
    console.log(`  Lower: $${lastBB.lower.toFixed(2)}`);
    console.log(`  Width: $${(lastBB.upper - lastBB.lower).toFixed(2)}`);
  }

  console.log(`\n🎯 Valid Entry Signals: ${signals.length}`);

  if (signals.length > 0) {
    console.log('\n📍 ENTRY SIGNALS DETAIL:\n');
    signals.forEach((sig, idx) => {
      const time = (sig.time.split('T')[1] || sig.time).substring(0, 8);
      console.log(`Signal ${idx + 1}: ${sig.type} at ${time}`);
      console.log(`  Price: $${sig.price.toFixed(2)}`);
      console.log(`  RSI: ${sig.rsi.toFixed(1)}`);
      console.log(`  BB: Lower $${sig.bb.lower.toFixed(2)} | Middle $${sig.bb.middle.toFixed(2)} | Upper $${sig.bb.upper.toFixed(2)}`);
      console.log('');
    });
  } else {
    console.log('\n❌ No valid entry signals met the criteria:');
    console.log('   LONG:  Price ≤ Lower BB + 0.5% AND RSI ≤ 30');
    console.log('   SHORT: Price ≥ Upper BB - 0.5% AND RSI ≥ 70');
  }
}

analyze5MinBars().catch(console.error);
