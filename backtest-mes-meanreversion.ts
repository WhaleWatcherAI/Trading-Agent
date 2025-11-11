#!/usr/bin/env tsx
/**
 * Mean Reversion Backtest for MES Futures
 * Uses EXACT same scaling logic as stock backtest:
 * - Entry: 2 units at RSI extreme + BB extreme
 * - First target: BB middle - scale out 50%, move stop to just inside middle
 * - Second target: Opposite outer BB - close remaining 50%
 */

import { backtestFuturesMeanReversion } from './lib/meanReversionBacktesterFutures';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

async function runMESBacktests() {
  console.log('\n' + '='.repeat(80));
  console.log('MES MEAN REVERSION BACKTEST - SCALING STRATEGY');
  console.log('='.repeat(80));
  console.log('\nStrategy:');
  console.log('  Entry: 2 units when RSI<30 (long) or RSI>70 (short) + at BB extreme');
  console.log('  First Target: BB Middle - scale out 1 unit, move stop to middle band');
  console.log('  Second Target: Opposite BB - close remaining 1 unit');
  console.log('  Stop Loss: 0.1% from entry (moves after first target)');
  console.log('='.repeat(80));

  const defaultContractId = process.env.TOPSTEPX_CONTRACT_ID || 'CON.F.US.MES.Z25';
  const config = {
    contractId: defaultContractId,
    startDate: process.env.TOPSTEPX_START_DATE || '2025-11-01T00:00:00Z',
    endDate: process.env.TOPSTEPX_END_DATE || '2025-11-07T23:59:59Z',
    commission: process.env.TOPSTEPX_FUTURES_COMMISSION
      ? Number(process.env.TOPSTEPX_FUTURES_COMMISSION)
      : inferFuturesCommissionPerSide(defaultContractId, 0.35),  // Defaults to MES $0.70 RT
  };

  const results: any[] = [];

  // Test 5-minute bars
  console.log('\n📊 Running 5-Minute Bar Backtest...');
  try {
    const result5min = await backtestFuturesMeanReversion(
      config.contractId,
      config.startDate,
      config.endDate,
      5,  // 5-minute bars
      config.commission
    );
    results.push({ barSize: 5, ...result5min });

    printResults(result5min, 5);
  } catch (error: any) {
    console.error(`\n✗ 5-minute backtest failed: ${error.message}`);
  }

  // Test 15-minute bars
  console.log('\n📊 Running 15-Minute Bar Backtest...');
  try {
    const result15min = await backtestFuturesMeanReversion(
      config.contractId,
      config.startDate,
      config.endDate,
      15,  // 15-minute bars
      config.commission
    );
    results.push({ barSize: 15, ...result15min });

    printResults(result15min, 15);
  } catch (error: any) {
    console.error(`\n✗ 15-minute backtest failed: ${error.message}`);
  }

  // Test 30-minute bars
  console.log('\n📊 Running 30-Minute Bar Backtest...');
  try {
    const result30min = await backtestFuturesMeanReversion(
      config.contractId,
      config.startDate,
      config.endDate,
      30,  // 30-minute bars
      config.commission
    );
    results.push({ barSize: 30, ...result30min });

    printResults(result30min, 30);
  } catch (error: any) {
    console.error(`\n✗ 30-minute backtest failed: ${error.message}`);
  }

  // Test 60-minute bars
  console.log('\n📊 Running 60-Minute Bar Backtest...');
  try {
    const result60min = await backtestFuturesMeanReversion(
      config.contractId,
      config.startDate,
      config.endDate,
      60,  // 60-minute bars
      config.commission
    );
    results.push({ barSize: 60, ...result60min });

    printResults(result60min, 60);
  } catch (error: any) {
    console.error(`\n✗ 60-minute backtest failed: ${error.message}`);
  }

  // Comparison
  if (results.length >= 2) {
    console.log('\n' + '='.repeat(80));
    console.log('COMPARISON: TOP BAR SIZES');
    console.log('='.repeat(80));

    results.forEach(result => {
      console.log(`\n${result.barSize}-Minute Bars:`);
      console.log(`  Trades: ${result.summary.totalTrades} (${result.summary.scaledTrades} scaled)`);
      console.log(`  Win Rate: ${result.summary.winRate.toFixed(1)}%`);
      console.log(`  Net P&L: $${result.summary.totalNetProfit.toFixed(2)}`);
      console.log(`  Avg Duration: ${result.summary.avgDurationMinutes.toFixed(0)} min`);
    });

    const best = results.reduce((prev, curr) => (
      curr.summary.totalNetProfit > prev.summary.totalNetProfit ? curr : prev
    ));

    console.log(`\n✓ Better Performance: ${best.barSize}-MIN bars ($${best.summary.totalNetProfit.toFixed(2)})`);
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
    console.log('  - No data returned from TopstepX for this period');
    console.log('  - RSI/BB conditions were never met');
    console.log('  - Try a different date range (more recent data)');
    return;
  }

  console.log(`\nTotal Trades: ${summary.totalTrades}`);
  console.log(`  Wins: ${summary.winCount} | Losses: ${summary.lossCount}`);
  console.log(`  Win Rate: ${summary.winRate.toFixed(1)}%`);
  console.log(`  Scaled Trades: ${summary.scaledTrades} (${((summary.scaledTrades / summary.totalTrades) * 100).toFixed(0)}% hit first target)`);

  console.log(`\nP&L:`);
  console.log(`  Gross Profit: $${summary.totalGrossProfit.toFixed(2)}`);
  console.log(`  Commission: -$${summary.totalCommission.toFixed(2)}`);
  console.log(`  Net Profit: $${summary.totalNetProfit.toFixed(2)}`);
  console.log(`  Max Drawdown: -$${summary.maxDrawdown.toFixed(2)}`);

  console.log(`\nPer Trade:`);
  console.log(`  Avg Win: $${summary.avgWin.toFixed(2)}`);
  console.log(`  Avg Loss: $${summary.avgLoss.toFixed(2)}`);
  console.log(`  Profit Factor: ${summary.profitFactor.toFixed(2)}`);
  console.log(`  Avg Duration: ${summary.avgDurationMinutes.toFixed(0)} minutes`);

  // Exit reason breakdown
  const exitReasons = {
    scale: trades.filter((t: any) => t.exitReason === 'scale').length,
    target: trades.filter((t: any) => t.exitReason === 'target' && !t.isScaledTrade).length,
    stop: trades.filter((t: any) => t.exitReason === 'stop').length,
    end_of_session: trades.filter((t: any) => t.exitReason === 'end_of_session').length,
  };

  console.log(`\nExit Reasons:`);
  console.log(`  First Target (Scaled): ${exitReasons.scale}`);
  console.log(`  Second Target: ${exitReasons.target}`);
  console.log(`  Stop Loss: ${exitReasons.stop}`);
  console.log(`  End of Session: ${exitReasons.end_of_session}`);

  // Sample trades
  console.log(`\nSample Trades (last 5):`);
  const lastTrades = trades.slice(-5);
  lastTrades.forEach((trade: any, idx: number) => {
    const sign = trade.netProfit >= 0 ? '+' : '';
    const scaleInfo = trade.isScaledTrade ? ' [SCALED 1U]' : trade.units === 1 ? ' [REMAINING 1U]' : ' [2U]';
    console.log(`\n  ${trades.length - 5 + idx + 1}. ${trade.direction.toUpperCase()} - ${trade.exitReason}${scaleInfo}`);
    console.log(`     Entry: ${trade.entryTimestamp.substring(0, 16)} @ ${trade.entryPrice.toFixed(2)} (RSI: ${trade.entryRSI?.toFixed(1) || 'N/A'})`);
    console.log(`     Exit:  ${trade.exitTimestamp.substring(0, 16)} @ ${trade.exitPrice.toFixed(2)} (RSI: ${trade.exitRSI?.toFixed(1) || 'N/A'})`);
    console.log(`     Ticks: ${trade.ticksGained.toFixed(1)} | P&L: ${sign}$${trade.netProfit.toFixed(2)} | Duration: ${trade.durationMinutes.toFixed(0)}min`);
  });

  console.log('\n' + '-'.repeat(80));
}

// Run the backtests
runMESBacktests().catch(err => {
  console.error('\n✗ Backtest failed:', err.message);
  if (err.stack) {
    console.error(err.stack);
  }
  process.exit(1);
});
