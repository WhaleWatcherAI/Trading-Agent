import axios from 'axios';

const SYMBOLS = ['MSFT', 'AMD', 'SPY', 'QQQ', 'META', 'NFLX', 'PLTR'];
const DATE = '2025-11-04';
const API_URL = 'http://localhost:3004/api/backtest-single';

async function analyzeGexExits() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  GEX WALL CROSSING EXIT ANALYSIS');
  console.log('═══════════════════════════════════════════════════\n');

  let totalGexExits = 0;
  let totalGexExitPnL = 0;
  let gexExitWins = 0;
  let gexExitLosses = 0;

  for (const symbol of SYMBOLS) {
    try {
      const response = await axios.get(API_URL, {
        params: { symbol, date: DATE }
      });

      const trades = response.data.trades || [];
      const gexExits = trades.filter((t: any) => t.exit.reason === 'GEX Wall Cross');

      if (gexExits.length > 0) {
        console.log(`\n${symbol}: ${gexExits.length} GEX wall exits`);

        gexExits.forEach((t: any) => {
          const profitEmoji = t.optionPnL >= 0 ? '✅' : '❌';
          console.log(`  ${profitEmoji} Trade #${t.tradeNumber} ${t.direction}: $${t.optionPnL.toFixed(2)} (${t.entry.time} → ${t.exit.time})`);

          totalGexExits++;
          totalGexExitPnL += t.optionPnL;
          if (t.optionPnL >= 0) gexExitWins++;
          else gexExitLosses++;
        });
      }
    } catch (error) {
      console.error(`Error analyzing ${symbol}`);
    }
  }

  console.log('\n═══════════════════════════════════════════════════');
  console.log('  SUMMARY');
  console.log('═══════════════════════════════════════════════════');
  console.log(`Total GEX Wall Exits: ${totalGexExits}`);
  console.log(`Winners: ${gexExitWins} | Losers: ${gexExitLosses}`);
  console.log(`Win Rate: ${totalGexExits > 0 ? ((gexExitWins / totalGexExits) * 100).toFixed(1) : 0}%`);
  console.log(`Total P&L from GEX Exits: $${totalGexExitPnL.toFixed(2)}`);
  console.log(`Avg P&L per GEX Exit: $${totalGexExits > 0 ? (totalGexExitPnL / totalGexExits).toFixed(2) : 0}`);
  console.log('═══════════════════════════════════════════════════');
}

analyzeGexExits()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Analysis failed:', error);
    process.exit(1);
  });
