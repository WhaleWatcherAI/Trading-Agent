#!/usr/bin/env tsx
import { backtestKeltnerScalp } from './lib/keltnerScalpBacktester';

async function runFullOctoberTest() {
  console.log('\n=== KELTNER SCALP - FULL OCTOBER 2025 (5-MIN BARS) ===\n');

  const result = await backtestKeltnerScalp(
    'CON.F.US.MES.Z25',
    '2025-10-01T00:00:00Z',
    '2025-10-31T23:59:59Z',
    5,  // 5-minute bars
    0.37
  );

  const { summary, trades } = result;

  console.log('\n' + '='.repeat(80));
  console.log('KELTNER SCALP RESULTS - FULL OCTOBER 2025 (MES)');
  console.log('='.repeat(80));
  console.log(`\nTotal Trades: ${summary.totalTrades}`);
  console.log(`  Wins: ${summary.winCount} | Losses: ${summary.lossCount}`);
  console.log(`  Win Rate: ${summary.winRate.toFixed(1)}%`);
  console.log(`  ADX Filter Rejections: ${summary.skippedByADX} setups avoided`);

  console.log(`\nP&L:`);
  console.log(`  Gross Profit: $${summary.totalGrossProfit.toFixed(2)}`);
  console.log(`  Commission: -$${summary.totalCommission.toFixed(2)}`);
  console.log(`  Net Profit: $${summary.totalNetProfit.toFixed(2)}`);
  console.log(`  Max Drawdown: -$${summary.maxDrawdown.toFixed(2)}`);

  console.log(`\nPer Trade:`);
  console.log(`  Avg Win: $${summary.avgWin.toFixed(2)}`);
  console.log(`  Avg Loss: $${summary.avgLoss.toFixed(2)}`);
  console.log(`  Profit Factor: ${summary.profitFactor.toFixed(2)}`);
  console.log(`  Avg Duration: ${summary.avgDurationMinutes.toFixed(0)} minutes`);

  console.log('\n' + '='.repeat(80));
  console.log('STRATEGY COMPARISON - FULL OCTOBER 2025 (MES)');
  console.log('='.repeat(80));
  console.log('\nSMA Crossover (15-min):');
  console.log('  Trades: 54 | Win Rate: 42.6% | PnL: $2,722.45');
  console.log('\nMean Reversion (5-min):');
  console.log('  Trades: 245 | Win Rate: 50.6% | PnL: $1,752.16');
  console.log('\nKeltner Scalp (5-min):');
  console.log(`  Trades: ${summary.totalTrades} | Win Rate: ${summary.winRate.toFixed(1)}% | PnL: $${summary.totalNetProfit.toFixed(2)}`);
  console.log('='.repeat(80));
}

runFullOctoberTest().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
