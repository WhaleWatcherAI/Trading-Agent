import 'dotenv/config';
import { backtestMeanReversionMultiple, MeanReversionBacktestResult } from './lib/meanReversionBacktester';

const symbols = ['AAPL', 'QQQ', 'SPY', 'NVDA'];
const tradingDayCount = Number(process.env.TRADING_DAYS || '7');

function getRecentTradingDays(count: number): string[] {
  const dates: string[] = [];
  const cursor = new Date();
  cursor.setHours(0, 0, 0, 0);
  cursor.setDate(cursor.getDate() - 1);

  while (dates.length < count) {
    const day = cursor.getDay();
    if (day !== 0 && day !== 6) {
      dates.push(cursor.toISOString().slice(0, 10));
    }
    cursor.setDate(cursor.getDate() - 1);
  }

  return dates;
}

function summarize(results: MeanReversionBacktestResult[]) {
  const bySymbol = new Map<string, MeanReversionBacktestResult[]>();
  for (const result of results) {
    if (!bySymbol.has(result.symbol)) {
      bySymbol.set(result.symbol, []);
    }
    bySymbol.get(result.symbol)!.push(result);
  }

  for (const [symbol, symbolResults] of bySymbol.entries()) {
    const tradedDays = symbolResults.filter(r => r.trades.length > 0);
    const trades = tradedDays.flatMap(r => r.trades);
    const wins = trades.filter(t => (t.stock?.profit ?? 0) > 0);
    const losses = trades.filter(t => (t.stock?.profit ?? 0) <= 0);

    const totalProfit = trades.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0);
    const grossProfit = wins.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0);
    const grossLoss = Math.abs(losses.reduce((sum, t) => sum + (t.stock?.profit ?? 0), 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
    const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0;

    console.log(`\n${symbol}`);
    console.log('─'.repeat(symbol.length));
    console.log(`  Days processed: ${symbolResults.length}`);
    console.log(`  Days traded:   ${tradedDays.length}`);
    console.log(`  Trades:        ${trades.length}`);
    console.log(`  Win rate:      ${winRate.toFixed(2)}%`);
    console.log(`  Total P&L:     $${totalProfit.toFixed(2)}`);
    console.log(`  Profit factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);

    if (trades.length > 0) {
      const avgDuration = trades.reduce((sum, t) => sum + t.durationMinutes, 0) / trades.length;
      console.log(`  Avg duration:  ${avgDuration.toFixed(0)} minutes`);
    }

    const exitReasons = new Map<string, number>();
    for (const trade of trades) {
      exitReasons.set(trade.exitReason, (exitReasons.get(trade.exitReason) ?? 0) + 1);
    }

    if (exitReasons.size > 0) {
      console.log('  Exit reasons:');
      for (const [reason, count] of exitReasons.entries()) {
        const pct = trades.length > 0 ? (count / trades.length) * 100 : 0;
        console.log(`    ${reason}: ${count} (${pct.toFixed(1)}%)`);
      }
    }
  }
}

async function main() {
  const dates = getRecentTradingDays(tradingDayCount);
  console.log('🚀 5-Min Mean Reversion Backtest');
  console.log(`Symbols: ${symbols.join(', ')}`);
  console.log(`Dates:   ${dates.join(', ')}`);

  const results = await backtestMeanReversionMultiple(symbols, dates, 'intraday', 5, 1);
  summarize(results);
}

main().catch(error => {
  console.error('❌ Backtest failed:', error);
  process.exit(1);
});
