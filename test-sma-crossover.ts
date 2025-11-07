import { runSmaCrossoverStrategy } from './lib/smaCrossoverAgent';
import { TradeSignal } from './types';

// Top 20 major tickers by market cap and liquidity
const TOP_20_TICKERS = [
  'SPY',   // S&P 500 ETF
  'QQQ',   // Nasdaq 100 ETF
  'AAPL',  // Apple
  'MSFT',  // Microsoft
  'GOOGL', // Alphabet
  'AMZN',  // Amazon
  'NVDA',  // Nvidia
  'TSLA',  // Tesla
  'META',  // Meta
  'BRK.B', // Berkshire Hathaway
  'V',     // Visa
  'JPM',   // JPMorgan
  'JNJ',   // Johnson & Johnson
  'WMT',   // Walmart
  'MA',    // Mastercard
  'PG',    // Procter & Gamble
  'XOM',   // Exxon Mobil
  'UNH',   // UnitedHealth
  'HD',    // Home Depot
  'CVX',   // Chevron
];

async function testSmaCrossoverOnAllTickers() {
  console.log('Testing 9-SMA Crossover Strategy on Top 20 Tickers');
  console.log('===================================================\n');

  const results: { symbol: string; signal: TradeSignal | null; error?: string }[] = [];

  for (const symbol of TOP_20_TICKERS) {
    try {
      console.log(`Testing ${symbol}...`);
      const signal = await runSmaCrossoverStrategy({ symbol });
      results.push({ symbol, signal });

      if (signal) {
        console.log(`✓ ${symbol}: SIGNAL FOUND - ${signal.type.toUpperCase()} ${signal.contract}`);
        console.log(`  Price: $${signal.currentPrice}, Strike: $${signal.strike}`);
        console.log(`  Reasoning: ${signal.reasoning}\n`);
      } else {
        console.log(`  No crossover signal\n`);
      }
    } catch (error: any) {
      console.error(`✗ ${symbol}: ERROR - ${error.message}\n`);
      results.push({ symbol, signal: null, error: error.message });
    }
  }

  // Summary
  console.log('\n===================================================');
  console.log('SUMMARY');
  console.log('===================================================\n');

  const signalsFound = results.filter(r => r.signal !== null);
  const errors = results.filter(r => r.error);

  console.log(`Total tickers tested: ${TOP_20_TICKERS.length}`);
  console.log(`Signals found: ${signalsFound.length}`);
  console.log(`Errors: ${errors.length}`);
  console.log(`No signal: ${results.length - signalsFound.length - errors.length}\n`);

  if (signalsFound.length > 0) {
    console.log('SIGNALS DETECTED:');
    signalsFound.forEach(({ symbol, signal }) => {
      console.log(`  ${symbol}: ${signal!.type.toUpperCase()} - ${signal!.contract}`);
    });
  }

  if (errors.length > 0) {
    console.log('\nERRORS:');
    errors.forEach(({ symbol, error }) => {
      console.log(`  ${symbol}: ${error}`);
    });
  }

  return results;
}

// Run the test
testSmaCrossoverOnAllTickers()
  .then(() => {
    console.log('\nTest completed!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('Test failed:', error);
    process.exit(1);
  });
