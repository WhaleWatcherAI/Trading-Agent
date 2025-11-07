import { backtestMeanReversionMultiple } from './lib/meanReversionBacktester';
import { promises as fs } from 'fs';

async function run() {
  const symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'TSLA', 'META', 'GOOGL', 'AMD'];
  const dates = [
    '2025-10-27',
    '2025-10-28',
    '2025-10-29',
    '2025-10-30',
    '2025-10-31',
  ];

  console.log(`🚀 Running mean reversion backtest for top symbols: ${symbols.join(', ')}`);
  console.log(`📅 Dates: ${dates.join(', ')}`);

  const results = await backtestMeanReversionMultiple(symbols, dates, 'intraday', 15);

  const outputPath = `backtest_mean_reversion_top10_${new Date().toISOString().split('T')[0]}.json`;
  await fs.writeFile(outputPath, JSON.stringify(results, null, 2));

  const tradedBySymbol = new Map<string, typeof results>();
  results.forEach(result => {
    if (!tradedBySymbol.has(result.symbol)) {
      tradedBySymbol.set(result.symbol, []);
    }
    tradedBySymbol.get(result.symbol)!.push(result);
  });

  console.log('\n📈 BACKTEST RESULTS SUMMARY');
  console.log('='.repeat(80));

  tradedBySymbol.forEach((symbolResults, symbol) => {
    const tradedDays = symbolResults.filter(r => r.trades.length > 0);
    const totalTrades = tradedDays.flatMap(r => r.trades);
    const wins = totalTrades.filter(t => t.stock.profit > 0);
    const losses = totalTrades.filter(t => t.stock.profit <= 0);
    const totalProfit = totalTrades.reduce((sum, t) => sum + t.stock.profit, 0);
    const grossProfit = wins.reduce((sum, t) => sum + t.stock.profit, 0);
    const grossLoss = Math.abs(losses.reduce((sum, t) => sum + t.stock.profit, 0));
    const winRate = totalTrades.length ? (wins.length / totalTrades.length) * 100 : 0;
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

    console.log(`\n${symbol}:`);
    console.log(`  Trades: ${totalTrades.length} | Win rate: ${winRate.toFixed(2)}% | Profit factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
    console.log(`  Total Profit: $${totalProfit.toFixed(2)} | Gross +$${grossProfit.toFixed(2)} / -$${grossLoss.toFixed(2)}`);
  });

  console.log('\n💾 Results saved to', outputPath);
}

run().catch(error => {
  console.error('Backtest failed:', error);
  process.exit(1);
});
