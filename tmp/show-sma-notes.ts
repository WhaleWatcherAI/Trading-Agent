import 'dotenv/config';
import { backtestMeanReversionMultiple } from '../lib/meanReversionBacktester';

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
  const symbols = ['AAPL', 'QQQ', 'SPY', 'NVDA'];
  const dates = await getRecentTradingDays(7);
  const results = await backtestMeanReversionMultiple(symbols, dates, 'intraday', 5, 1);
  const bySymbol = new Map<string, { date: string; count: number; notes: string[] }[]>();
  for (const res of results) {
    const breakevenNotes = res.notes.filter(note => note.includes('breakeven'));
    if (!bySymbol.has(res.symbol)) bySymbol.set(res.symbol, []);
    bySymbol.get(res.symbol)!.push({ date: res.date, count: breakevenNotes.length, notes: breakevenNotes });
  }

  for (const [symbol, entries] of bySymbol.entries()) {
    const total = entries.reduce((sum, entry) => sum + entry.count, 0);
    console.log(`\n${symbol}: ${total} SMA breakeven moves across ${entries.length} days`);
    entries
      .filter(entry => entry.count > 0)
      .forEach(entry => {
        console.log(`  ${entry.date}: ${entry.count}`);
        entry.notes.forEach(note => console.log(`    ${note}`));
      });
  }
})();
