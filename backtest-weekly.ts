import axios from 'axios';

// Top 20 liquid stocks
const SYMBOLS = [
  'SPY', 'QQQ', 'TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'META',
  'GOOGL', 'NFLX', 'AMD', 'SPOT', 'MA', 'V', 'DIA',
  'JPM', 'BAC', 'UNH', 'PFE', 'CAT'
];

// Past week of trading days (excluding weekends)
const DATES = [
  '2025-10-28', // Monday
  '2025-10-29', // Tuesday
  '2025-10-30', // Wednesday
  '2025-10-31', // Thursday
  '2025-11-01', // Friday
  '2025-11-04', // Monday
];

const API_URL = 'http://localhost:3004/api/backtest-single';

interface DayResult {
  date: string;
  totalTrades: number;
  totalPnL: number;
  winners: number;
  losers: number;
  stocksTraded: number;
}

async function backtestDay(date: string): Promise<DayResult> {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`  TESTING ${date}`);
  console.log('='.repeat(60));

  let totalTrades = 0;
  let totalPnL = 0;
  let winners = 0;
  let losers = 0;
  let stocksTraded = 0;

  for (const symbol of SYMBOLS) {
    try {
      const response = await axios.get(API_URL, {
        params: { symbol, date },
        timeout: 30000
      });

      const stats = response.data.stats;

      if (stats.totalTrades > 0) {
        console.log(`  ${symbol.padEnd(6)} - ${stats.totalTrades} trades, P&L: $${stats.totalPnL.toFixed(2)}`);
        totalTrades += stats.totalTrades;
        totalPnL += stats.totalPnL;
        stocksTraded++;

        if (stats.totalPnL > 0) winners++;
        else if (stats.totalPnL < 0) losers++;
      }

      // Small delay to avoid overwhelming API
      await new Promise(resolve => setTimeout(resolve, 300));
    } catch (error: any) {
      console.error(`  ❌ Error testing ${symbol}: ${error.message}`);
    }
  }

  console.log(`\n  📊 Day Summary: ${totalTrades} trades, $${totalPnL.toFixed(2)} P&L, ${stocksTraded} stocks traded`);

  return {
    date,
    totalTrades,
    totalPnL,
    winners,
    losers,
    stocksTraded,
  };
}

async function runWeeklyBacktest() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  WEEKLY BACKTEST - GEX ZONE + 20-SMA STRATEGY');
  console.log('═══════════════════════════════════════════════════════════');
  console.log(`Testing ${SYMBOLS.length} stocks across ${DATES.length} days\n`);

  const results: DayResult[] = [];

  for (const date of DATES) {
    const dayResult = await backtestDay(date);
    results.push(dayResult);
  }

  console.log('\n\n');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  WEEKLY SUMMARY');
  console.log('═══════════════════════════════════════════════════════════\n');

  console.log('Daily Results:');
  console.log('-'.repeat(60));
  results.forEach(r => {
    const emoji = r.totalPnL >= 0 ? '✅' : '❌';
    console.log(`${emoji} ${r.date} - ${r.totalTrades.toString().padStart(3)} trades | P&L: $${r.totalPnL.toFixed(2).padStart(10)} | ${r.stocksTraded} stocks`);
  });

  const totalTrades = results.reduce((sum, r) => sum + r.totalTrades, 0);
  const totalPnL = results.reduce((sum, r) => sum + r.totalPnL, 0);
  const avgDailyPnL = totalPnL / results.length;
  const profitableDays = results.filter(r => r.totalPnL > 0).length;
  const losingDays = results.filter(r => r.totalPnL < 0).length;
  const avgTradesPerDay = totalTrades / results.length;

  console.log('\n' + '='.repeat(60));
  console.log('AGGREGATE STATS:');
  console.log('='.repeat(60));
  console.log(`Total Days Tested: ${results.length}`);
  console.log(`Profitable Days: ${profitableDays} | Losing Days: ${losingDays}`);
  console.log(`Total Trades: ${totalTrades}`);
  console.log(`Avg Trades/Day: ${avgTradesPerDay.toFixed(1)}`);
  console.log(`Total P&L: $${totalPnL.toFixed(2)}`);
  console.log(`Avg Daily P&L: $${avgDailyPnL.toFixed(2)}`);
  console.log(`P&L per Trade: $${totalTrades > 0 ? (totalPnL / totalTrades).toFixed(2) : '0.00'}`);

  const bestDay = results.reduce((best, r) => r.totalPnL > best.totalPnL ? r : best, results[0]);
  const worstDay = results.reduce((worst, r) => r.totalPnL < worst.totalPnL ? r : worst, results[0]);

  console.log(`\nBest Day: ${bestDay.date} ($${bestDay.totalPnL.toFixed(2)})`);
  console.log(`Worst Day: ${worstDay.date} ($${worstDay.totalPnL.toFixed(2)})`);

  console.log('\n' + '='.repeat(60));
  console.log('Backtest Complete!');
  console.log('='.repeat(60));
}

runWeeklyBacktest()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Weekly backtest failed:', error);
    process.exit(1);
  });
