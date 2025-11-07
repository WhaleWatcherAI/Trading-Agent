#!/usr/bin/env node

/**
 * Test regime backtester across multiple dates
 */

const dates = [
  '2024-10-28',
  '2024-10-25',
  '2024-10-24',
  '2024-10-23',
  '2024-10-22',
  '2024-10-21',
  '2024-10-18',
  '2024-10-17',
  '2024-10-16',
  '2024-10-15',
  '2024-10-14',
  '2024-10-11',
  '2024-10-10',
  '2024-10-09',
  '2024-10-08',
  '2024-10-07',
  '2024-10-04',
  '2024-10-03',
  '2024-10-02',
  '2024-10-01',
];

async function testDate(date) {
  const url = `http://localhost:3002/api/regime/backtest-v2?date=${date}&mode=scalp`;

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

    return {
      date,
      signals: data.regimeAnalyses?.reduce((sum, a) => sum + (a.tradeSignals?.length || 0), 0) || 0,
      trades: data.trades?.length || 0,
      totalProfit: data.metrics?.totalProfit || 0,
      winRate: data.metrics?.winRate || 0,
      notes: data.notes?.join('; ') || '',
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
  console.log('Testing regime backtester across 20 dates...\n');

  const results = [];

  for (const date of dates) {
    console.log(`Testing ${date}...`);
    const result = await testDate(date);
    results.push(result);

    if (result.error) {
      console.log(`  ❌ Error: ${result.error}`);
    } else {
      console.log(`  📊 Signals: ${result.signals}, Trades: ${result.trades}, Profit: $${result.totalProfit.toFixed(2)}, Win Rate: ${(result.winRate * 100).toFixed(1)}%`);
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('AGGREGATE RESULTS');
  console.log('='.repeat(80));

  const successfulTests = results.filter(r => !r.error);
  const totalSignals = successfulTests.reduce((sum, r) => sum + r.signals, 0);
  const totalTrades = successfulTests.reduce((sum, r) => sum + r.trades, 0);
  const totalProfit = successfulTests.reduce((sum, r) => sum + r.totalProfit, 0);
  const avgWinRate = successfulTests.length > 0
    ? successfulTests.reduce((sum, r) => sum + r.winRate, 0) / successfulTests.length
    : 0;

  console.log(`Total Dates Tested: ${dates.length}`);
  console.log(`Successful Tests: ${successfulTests.length}`);
  console.log(`Failed Tests: ${results.filter(r => r.error).length}`);
  console.log(`Total Signals Generated: ${totalSignals}`);
  console.log(`Total Trades Executed: ${totalTrades}`);
  console.log(`Signal-to-Trade Ratio: ${totalSignals > 0 ? ((totalTrades / totalSignals) * 100).toFixed(1) : 0}%`);
  console.log(`Total Profit: $${totalProfit.toFixed(2)}`);
  console.log(`Average Win Rate: ${(avgWinRate * 100).toFixed(1)}%`);

  console.log('\n' + '='.repeat(80));
  console.log('DETAILED RESULTS');
  console.log('='.repeat(80));

  successfulTests.forEach(r => {
    console.log(`\n${r.date}:`);
    console.log(`  Signals: ${r.signals}`);
    console.log(`  Trades: ${r.trades}`);
    if (r.trades > 0) {
      console.log(`  Profit: $${r.totalProfit.toFixed(2)}`);
      console.log(`  Win Rate: ${(r.winRate * 100).toFixed(1)}%`);
    }
  });
}

main().catch(console.error);
