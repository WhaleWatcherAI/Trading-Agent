#!/usr/bin/env tsx
/**
 * Keltner Channel + RSI Scalping Backtest
 *
 * Strategy:
 * - Entry: RSI < 25 (long) or > 75 (short) + price outside Keltner(20, 1.5)
 * - Exit: Middle Keltner band OR RSI back to 50
 * - Filter: Skip if ADX > 25 (trending market)
 * - Ideal for: NQ/MES scalps in ranging markets
 */

import { backtestKeltnerScalp } from './lib/keltnerScalpBacktester';

async function runKeltnerScalpBacktests() {
  console.log('\n' + '='.repeat(80));
  console.log('KELTNER CHANNEL + RSI SCALPING BACKTEST');
  console.log('='.repeat(80));
  console.log('\nStrategy:');
  console.log('  Entry: RSI < 25 (long) or RSI > 75 (short) + price outside Keltner(20, 1.5)');
  console.log('  Exit: Middle Keltner band OR RSI returns to 50');
  console.log('  Filter: Skip if ADX > 25 (trending market)');
  console.log('  Stop Loss: 0.2% from entry (tight scalping stop)');
  console.log('='.repeat(80));

  const config = {
    contractId: process.env.TOPSTEPX_CONTRACT_ID || 'CON.F.US.MES.Z25',
    startDate: process.env.TOPSTEPX_START_DATE || '2025-11-01T00:00:00Z',
    endDate: process.env.TOPSTEPX_END_DATE || '2025-11-07T23:59:59Z',
    commission: Number(process.env.TOPSTEPX_FUTURES_COMMISSION || '0.37'),
  };

  const results: any[] = [];

  // Test different timeframes
  const timeframes = [
    { label: '1-Minute', minutes: 1 },
    { label: '3-Minute', minutes: 3 },
    { label: '5-Minute', minutes: 5 },
  ];

  for (const tf of timeframes) {
    console.log(`\n⚡ Running ${tf.label} Bar Backtest...`);
    try {
      const result = await backtestKeltnerScalp(
        config.contractId,
        config.startDate,
        config.endDate,
        tf.minutes,
        config.commission,
      );
      results.push({ barSize: tf.minutes, ...result });

      printResults(result, tf.minutes);
    } catch (error: any) {
      console.error(`\n✗ ${tf.label} backtest failed: ${error.message}`);
    }
  }

  // Comparison
  if (results.length >= 2) {
    console.log('\n' + '='.repeat(80));
    console.log('COMPARISON: BEST TIMEFRAME');
    console.log('='.repeat(80));

    results.forEach(result => {
      console.log(`\n${result.barSize}-Minute Bars:`);
      console.log(`  Trades: ${result.summary.totalTrades}`);
      console.log(`  Win Rate: ${result.summary.winRate.toFixed(1)}%`);
      console.log(`  Net P&L: $${result.summary.totalNetProfit.toFixed(2)}`);
      console.log(`  Avg Duration: ${result.summary.avgDurationMinutes.toFixed(0)} min`);
      console.log(`  ADX Filter Rejections: ${result.summary.skippedByADX}`);
    });

    const best = results.reduce((prev, curr) =>
      curr.summary.totalNetProfit > prev.summary.totalNetProfit ? curr : prev,
    );

    console.log(`\n✓ Best Performance: ${best.barSize}-MIN bars ($${best.summary.totalNetProfit.toFixed(2)})`);
    console.log('='.repeat(80));
  }
}

function printResults(result: any, barSize: number) {
  console.log('\n' + '-'.repeat(80));
  console.log(`RESULTS: ${barSize}-Minute Bars`);
  console.log('-'.repeat(80));

  const { summary, trades } = result;

  if (trades.length === 0) {
    console.log('\n✗ No trades generated');
    console.log('\nPossible reasons:');
    console.log('  - ADX filter rejected all setups (market was trending)');
    console.log('  - RSI/Keltner conditions were never met');
    console.log('  - Try a different date range or more volatile market');
    return;
  }

  console.log(`\nTotal Trades: ${summary.totalTrades}`);
  console.log(`  Wins: ${summary.winCount} | Losses: ${summary.lossCount}`);
  console.log(`  Win Rate: ${summary.winRate.toFixed(1)}%`);
  console.log(`  ADX Filter Rejections: ${summary.skippedByADX} (avoided trending setups)`);

  console.log(`\nP&L:`);
  console.log(`  Gross Profit: $${summary.totalGrossProfit.toFixed(2)}`);
  console.log(`  Commission: -$${summary.totalCommission.toFixed(2)}`);
  console.log(`  Net Profit: $${summary.totalNetProfit.toFixed(2)}`);
  console.log(`  Max Drawdown: -$${summary.maxDrawdown.toFixed(2)}`);

  console.log(`\nPer Trade:`);
  console.log(`  Avg Win: $${summary.avgWin.toFixed(2)}`);
  console.log(`  Avg Loss: $${summary.avgLoss.toFixed(2)}`);
  console.log(`  Profit Factor: ${summary.profitFactor.toFixed(2)}`);
  console.log(`  Avg Duration: ${summary.avgDurationMinutes.toFixed(0)} minutes (${(summary.avgDurationMinutes / 60).toFixed(1)} hours)`);

  // Exit reason breakdown
  const exitReasons = {
    target: trades.filter((t: any) => t.exitReason === 'target').length,
    stop: trades.filter((t: any) => t.exitReason === 'stop').length,
    rsi_neutral: trades.filter((t: any) => t.exitReason === 'rsi_neutral').length,
    end_of_session: trades.filter((t: any) => t.exitReason === 'end_of_session').length,
  };

  console.log(`\nExit Reasons:`);
  console.log(`  Target (Middle Keltner): ${exitReasons.target} (${((exitReasons.target / trades.length) * 100).toFixed(0)}%)`);
  console.log(`  RSI Neutral: ${exitReasons.rsi_neutral} (${((exitReasons.rsi_neutral / trades.length) * 100).toFixed(0)}%)`);
  console.log(`  Stop Loss: ${exitReasons.stop} (${((exitReasons.stop / trades.length) * 100).toFixed(0)}%)`);
  console.log(`  End of Session: ${exitReasons.end_of_session}`);

  // Sample trades
  console.log(`\nSample Trades (last 5):`);
  const lastTrades = trades.slice(-5);
  lastTrades.forEach((trade: any, idx: number) => {
    const sign = trade.netProfit >= 0 ? '+' : '';
    console.log(`\n  ${trades.length - 5 + idx + 1}. ${trade.direction.toUpperCase()} - ${trade.exitReason}`);
    console.log(`     Entry: ${trade.entryTimestamp.substring(0, 16)} @ ${trade.entryPrice.toFixed(2)}`);
    console.log(`     RSI: ${trade.entryRSI?.toFixed(1) || 'N/A'} | ADX: ${trade.entryADX?.toFixed(1) || 'N/A'}`);
    console.log(`     Exit:  ${trade.exitTimestamp.substring(0, 16)} @ ${trade.exitPrice.toFixed(2)}`);
    console.log(`     Ticks: ${trade.ticksGained.toFixed(1)} | P&L: ${sign}$${trade.netProfit.toFixed(2)} | Duration: ${trade.durationMinutes.toFixed(0)}min`);
  });

  console.log('\n' + '-'.repeat(80));
}

// Run the backtests
runKeltnerScalpBacktests().catch(err => {
  console.error('\n✗ Backtest failed:', err.message);
  if (err.stack) {
    console.error(err.stack);
  }
  process.exit(1);
});
