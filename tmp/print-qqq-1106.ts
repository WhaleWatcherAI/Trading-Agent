import 'dotenv/config';
import { backtestMeanReversionMultiple } from '../lib/meanReversionBacktester';

function toET(iso: string): string {
  const localString = new Date(iso).toLocaleString('en-US', { timeZone: 'America/New_York' });
  const localDate = new Date(localString);
  return localDate.toISOString().slice(11, 19);
}

(async () => {
  const date = '2025-11-06';
  const [result] = await backtestMeanReversionMultiple(['QQQ'], [date], 'intraday', 5, 1);
  console.log(`Trades for QQQ on ${date}`);
  for (const trade of result.trades) {
    console.log(`\n${toET(trade.entryTimestamp)} -> ${toET(trade.exitTimestamp)}`);
    console.log(
      `  Entry ${trade.stock.entryPrice.toFixed(2)} | Stop ${trade.stopLoss?.toFixed(2) ?? 'n/a'} | Target ${trade.target?.toFixed(2) ?? 'n/a'}`,
    );
    console.log(
      `  Exit ${trade.stock.exitPrice.toFixed(2)} (${trade.exitReason}) | PnL ${trade.stock.profit.toFixed(2)}`,
    );
  }
})();
