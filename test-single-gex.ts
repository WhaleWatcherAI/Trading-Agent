import { calculateGexForSymbol } from './lib/gexCalculator';

async function test() {
  console.log('Testing GEX calculation for SPY on specific dates:\n');

  // Test with recent dates (last few days - current expirations should still exist)
  const dates = ['2025-11-01', '2025-11-04', '2025-11-05'];

  for (const date of dates) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Date: ${date}`);
    console.log('='.repeat(60));

    const gex = await calculateGexForSymbol('SPY', 'intraday', date);

    console.log(`  Stock Price: $${gex.stockPrice.toFixed(2)}`);
    console.log(`  Call GEX: $${(gex.summary.totalCallGex / 1_000_000).toFixed(1)}M`);
    console.log(`  Put GEX: $${(gex.summary.totalPutGex / 1_000_000).toFixed(1)}M`);
    console.log(`  Net GEX: $${(gex.summary.netGex / 1_000_000).toFixed(1)}M`);
    console.log(`  Regime: ${gex.summary.netGex < 0 ? 'NEGATIVE (trending)' : 'POSITIVE (pinned)'}`);
  }
}

test().catch(console.error);
