import 'dotenv/config';
import { backtestMeanReversionMultiple, MeanReversionTrade } from '../lib/meanReversionBacktester';

async function getRecentTradingDays(count: number) {
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

(async () => {
  const dates = await getRecentTradingDays(7);
  const [result] = await backtestMeanReversionMultiple(['QQQ'], dates, 'intraday', 5, 1);
  console.log(`Trades for QQQ across ${dates.join(', ')}`);
  for (const trade of result.trades) {
    const { entryTimestamp, exitTimestamp, stock, exitReason, entryPrice, stopLoss, target } = trade;
    console.log(`\n${entryTimestamp} -> ${exitTimestamp}`);
    console.log(`  Entry ${entryPrice.toFixed(2)} Stop ${trade.stopLoss?.toFixed(2) ?? 'n/a'} Target ${trade.target?.toFixed(2) ?? 'n/a'}`);
    console.log(`  Exit ${stock.exitPrice.toFixed(2)} (${exitReason}) PnL ${stock.profit.toFixed(2)}`);
  }
  console.log('\nNotes:');
  result.notes.forEach(note => console.log('  ' + note));
})();
