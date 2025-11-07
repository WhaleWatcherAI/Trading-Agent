import 'dotenv/config';
import { backtestMeanReversionMultiple } from '../lib/meanReversionBacktester';

(async () => {
  const date = '2025-11-06';
  const [result] = await backtestMeanReversionMultiple(['QQQ'], [date], 'intraday', 5, 1);
  console.log(`Trades for QQQ on ${date}`);
  for (const trade of result.trades) {
    console.log(`\nEntry ${trade.entryTimestamp} -> Exit ${trade.exitTimestamp}`);
    console.log(
      `  Entry ${trade.stock.entryPrice.toFixed(2)} Stop ${trade.stopLoss?.toFixed(2) ?? 'n/a'} Target ${trade.target?.toFixed(2) ?? 'n/a'} ` +
        `BB Lower ${trade.signal.bbLower?.toFixed(2) ?? 'n/a'} Middle ${trade.signal.bbMiddle?.toFixed(2) ?? 'n/a'} Upper ${trade.signal.bbUpper?.toFixed(2) ?? 'n/a'} ` +
        `RSI ${trade.signal.rsi?.toFixed(1) ?? 'n/a'}`,
    );
    console.log(`  Exit ${trade.stock.exitPrice.toFixed(2)} (${trade.exitReason}) PnL ${trade.stock.profit.toFixed(2)}`);
  }
  console.log('\nNotes:');
  result.notes.forEach(note => console.log('  ' + note));
})();
