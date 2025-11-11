#!/usr/bin/env tsx
import { backtestKeltnerScalp } from './lib/keltnerScalpBacktester';

async function runES() {
  console.log('\n=== Running Keltner Scalp on ES (Full Contract) ===\n');

  const result = await backtestKeltnerScalp(
    'CON.F.US.EP.Z25',
    '2025-10-01T00:00:00Z',
    '2025-10-31T23:59:59Z',
    5,
    0.62  // ES commission
  );

  console.log('\n=== KELTNER SCALP - ES RESULTS ===');
  console.log('Trades:', result.summary.totalTrades);
  console.log('Win Rate:', result.summary.winRate.toFixed(1) + '%');
  console.log('Net PnL: $' + result.summary.totalNetProfit.toFixed(2));
  console.log('ADX Rejections:', result.summary.skippedByADX);
  console.log('Avg Win: $' + result.summary.avgWin.toFixed(2));
  console.log('Avg Loss: $' + result.summary.avgLoss.toFixed(2));
}

runES().catch(console.error);
