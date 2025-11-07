import axios from 'axios';

// 60 liquid stocks with good options liquidity
const SYMBOLS = [
  // Mega Cap Tech
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
  // Large Cap Tech
  'AMD', 'INTC', 'QCOM', 'AVGO', 'CRM', 'ORCL', 'ADBE', 'CSCO',
  // Communication & Media
  'DIS', 'CMCSA', 'T', 'VZ', 'NFLX', 'SNAP', 'PINS', 'SPOT',
  // Financials
  'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'PYPL',
  // Consumer
  'AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD',
  // Healthcare
  'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'CVS',
  // Energy
  'XOM', 'CVX', 'COP', 'SLB', 'OXY',
  // Industrials
  'BA', 'CAT', 'GE', 'UPS', 'LMT',
  // ETFs & Index
  'SPY', 'QQQ', 'IWM', 'DIA'
];

const DATE = '2025-11-04';
const API_URL = 'http://localhost:3004/api/backtest-single';

interface BacktestResult {
  symbol: string;
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
  profitFactor: number;
  callPutRatio: number;
  bias: string;
  netGex: number;
  hasNegativeGex: boolean;
}

async function backtestStock(symbol: string): Promise<BacktestResult | null> {
  try {
    console.log(`\n🔍 Testing ${symbol}...`);
    const response = await axios.get(API_URL, {
      params: { symbol, date: DATE }
    });

    const stats = response.data.stats;

    console.log(`✅ ${symbol}: ${stats.totalTrades} trades, Net GEX: ${stats.netGex.toFixed(2)}, C/P: ${stats.callPutRatio.toFixed(2)}`);

    return {
      symbol: stats.symbol,
      totalTrades: stats.totalTrades,
      winRate: stats.winRate,
      totalPnL: stats.totalPnL,
      avgPnL: stats.avgPnL,
      profitFactor: stats.profitFactor,
      callPutRatio: stats.callPutRatio,
      bias: stats.bias,
      netGex: stats.netGex,
      hasNegativeGex: stats.hasNegativeGex,
    };
  } catch (error: any) {
    console.error(`❌ Error testing ${symbol}:`, error.message);
    return null;
  }
}

async function runMultipleBacktests() {
  console.log('═══════════════════════════════════════════════════');
  console.log(`  MULTI-STOCK BACKTEST - ${DATE}`);
  console.log('═══════════════════════════════════════════════════');
  console.log(`Testing ${SYMBOLS.length} stocks...`);
  console.log('');

  const results: BacktestResult[] = [];

  // Run backtests sequentially to avoid overwhelming the API
  for (const symbol of SYMBOLS) {
    const result = await backtestStock(symbol);
    if (result) {
      results.push(result);
    }
    // Small delay to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  console.log('\n═══════════════════════════════════════════════════');
  console.log('  SUMMARY');
  console.log('═══════════════════════════════════════════════════\n');

  // Sort by total P&L
  const sorted = [...results].sort((a, b) => b.totalPnL - a.totalPnL);

  // Stocks with trades
  const tradedStocks = sorted.filter(r => r.totalTrades > 0);
  const noTradeStocks = sorted.filter(r => r.totalTrades === 0);

  console.log(`📊 STOCKS WITH TRADES (${tradedStocks.length}/${results.length}):\n`);

  if (tradedStocks.length > 0) {
    tradedStocks.forEach(r => {
      const emoji = r.totalPnL >= 0 ? '✅' : '❌';
      console.log(`${emoji} ${r.symbol.padEnd(6)} - ${r.totalTrades} trades | P&L: $${r.totalPnL.toFixed(2).padStart(10)} | Win Rate: ${r.winRate.toFixed(1)}% | GEX: ${(r.netGex / 1000000).toFixed(1)}M`);
    });

    const totalTrades = tradedStocks.reduce((sum, r) => sum + r.totalTrades, 0);
    const totalPnL = tradedStocks.reduce((sum, r) => sum + r.totalPnL, 0);
    const avgWinRate = tradedStocks.reduce((sum, r) => sum + r.winRate, 0) / tradedStocks.length;

    console.log('\n📈 AGGREGATE STATS:');
    console.log(`   Total Trades: ${totalTrades}`);
    console.log(`   Total P&L: $${totalPnL.toFixed(2)}`);
    console.log(`   Avg Win Rate: ${avgWinRate.toFixed(1)}%`);
    console.log(`   Stocks Traded: ${tradedStocks.length}`);
  } else {
    console.log('   No stocks had trades (all blocked by GEX filter)');
  }

  console.log(`\n🚫 STOCKS BLOCKED BY GEX FILTER (${noTradeStocks.length}):\n`);
  noTradeStocks.forEach(r => {
    const gexStatus = r.hasNegativeGex ? 'NEGATIVE' : 'POSITIVE';
    console.log(`   ${r.symbol.padEnd(6)} - GEX: ${(r.netGex / 1000000).toFixed(1)}M (${gexStatus})`);
  });

  console.log('\n═══════════════════════════════════════════════════');
  console.log('Backtest Complete!');
  console.log('═══════════════════════════════════════════════════');
}

runMultipleBacktests()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Backtest failed:', error);
    process.exit(1);
  });
