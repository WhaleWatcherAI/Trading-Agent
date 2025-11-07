#!/usr/bin/env node

/**
 * Test regime backtester on very recent dates where GEX data is available
 */

const dates = [
  '2024-11-01',
  '2024-10-31',
  '2024-10-30',
  '2024-10-29',
  '2024-10-28',
  '2024-10-25',
  '2024-10-24',
  '2024-10-23',
  '2024-10-22',
  '2024-10-21',
];

async function testDate(date) {
  const url = `http://localhost:3002/api/regime/backtest-v2?date=${date}&mode=swing`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (data.error) {
      return {
        date,
        error: data.error,
        signals: 0,
        trades: 0,
      };
    }

    const signals = data.regimeAnalyses?.reduce((sum, a) => sum + (a.tradeSignals?.length || 0), 0) || 0;
    const trades = data.trades || [];

    return {
      date,
      signals,
      trades: trades.length,
      tradeDetails: trades.map(t => ({
        symbol: t.symbol,
        direction: t.direction,
        entry: t.entryPrice.toFixed(2),
        exit: t.exitPrice.toFixed(2),
        profitPct: (t.profitPct * 100).toFixed(2),
        exitReason: t.exitReason,
      })),
    };
  } catch (error) {
    return {
      date,
      error: error.message,
      signals: 0,
      trades: 0,
    };
  }
}

async function main() {
  console.log('Testing regime backtester on recent dates with swing mode...\n');

  for (const date of dates) {
    console.log(`Testing ${date}...`);
    const result = await testDate(date);

    if (result.error) {
      console.log(`  ❌ Error: ${result.error}`);
    } else if (result.trades > 0) {
      console.log(`  ✅ Signals: ${result.signals}, Trades: ${result.trades}`);
      result.tradeDetails.forEach((t, i) => {
        console.log(`     ${i + 1}. ${t.symbol} ${t.direction}: Entry $${t.entry} → Exit $${t.exit} (${t.profitPct}%, ${t.exitReason})`);
      });
    } else {
      console.log(`  📊 Signals: ${result.signals}, Trades: 0`);
    }
  }
}

main().catch(console.error);
