import { backtestMeanReversionMultiple } from './lib/meanReversionBacktester';
import { promises as fs } from 'fs';

async function run() {
  const symbol = 'SPX';
  const dates = [
    '2025-10-27',
    '2025-10-28',
    '2025-10-29',
    '2025-10-30',
    '2025-10-31',
  ];

  console.log(`🚀 Running mean reversion backtest for ${symbol}`);
  console.log(`📅 Dates: ${dates.join(', ')}`);

  const results = await backtestMeanReversionMultiple([symbol], dates, 'intraday');

  const outputPath = `backtest_mean_reversion_${symbol}_${new Date().toISOString().split('T')[0]}.json`;
  await fs.writeFile(outputPath, JSON.stringify(results, null, 2));

  const traded = results.filter(r => r.trades.length > 0);
  const skipped = results.length - traded.length;

  console.log(`\nSummary for ${symbol}:`);
  console.log(`  Days processed: ${results.length}`);
  console.log(`  Days traded: ${traded.length}`);
  console.log(`  Days skipped: ${skipped}`);

  const trades = traded.flatMap(r => r.trades);
  const wins = trades.filter(t => t.stock.profit > 0);
  const losses = trades.filter(t => t.stock.profit <= 0);

  const totalProfit = trades.reduce((sum, t) => sum + t.stock.profit, 0);
  const grossProfit = wins.reduce((sum, t) => sum + t.stock.profit, 0);
  const grossLoss = Math.abs(losses.reduce((sum, t) => sum + t.stock.profit, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

  console.log(`  Total trades: ${trades.length}`);
  console.log(`  Win rate: ${trades.length ? ((wins.length / trades.length) * 100).toFixed(2) : '0.00'}%`);
  console.log(`  Total profit: $${totalProfit.toFixed(2)}`);
  console.log(`  Profit factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`\n💾 Results saved to ${outputPath}`);
}

run().catch(error => {
  console.error('Backtest failed:', error);
  process.exit(1);
});
