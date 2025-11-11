#!/usr/bin/env tsx
/**
 * Test both SMA Crossover and Mean Reversion strategies on Micro Euro (M6E)
 * Full October 2025
 */

import { backtestFuturesMeanReversion } from './lib/meanReversionBacktesterFutures';
import { spawnSync } from 'child_process';

const CONTRACT_ID = 'CON.F.US.M6E.Z25';  // Micro Euro
const START_DATE = '2025-10-01T00:00:00Z';
const END_DATE = '2025-10-31T23:59:59Z';
const COMMISSION = 0.37;  // Micro Euro commission per side

async function runEuroBacktests() {
  console.log('\n' + '='.repeat(80));
  console.log('MICRO EURO (M6E) STRATEGY BACKTESTS - OCTOBER 2025');
  console.log('='.repeat(80));

  // 1. SMA Crossover Strategy (15-minute bars)
  console.log('\n🔄 Running SMA Crossover Strategy (15-min bars)...\n');

  const smaResult = spawnSync('npx', ['tsx', 'backtest-topstepx-sma.ts'], {
    env: {
      ...process.env,
      TOPSTEPX_CONTRACT_ID: CONTRACT_ID,
      TOPSTEPX_SMA_START: START_DATE,
      TOPSTEPX_SMA_END: END_DATE,
      TOPSTEPX_SMA_INTERVAL: '15m',
      TOPSTEPX_SMA_PERIOD: '20',
      TOPSTEPX_SMA_COMMISSION: COMMISSION.toString(),
      TOPSTEPX_STOP_LOSS_PERCENT: '0.001',
      TOPSTEPX_TAKE_PROFIT_PERCENT: '0.011',
    },
    stdio: 'inherit',
    shell: true,
  });

  if (smaResult.status !== 0) {
    console.error('\n✗ SMA Crossover backtest failed');
  }

  // 2. Mean Reversion Strategy (5-minute bars)
  console.log('\n\n📊 Running Mean Reversion Strategy (5-min bars)...\n');

  try {
    const meanRevResult = await backtestFuturesMeanReversion(
      CONTRACT_ID,
      START_DATE,
      END_DATE,
      5,  // 5-minute bars
      COMMISSION
    );

    console.log('\n' + '-'.repeat(80));
    console.log('MEAN REVERSION RESULTS - 5-MIN BARS');
    console.log('-'.repeat(80));
    console.log(`\nTotal Trades: ${meanRevResult.summary.totalTrades}`);
    console.log(`  Wins: ${meanRevResult.summary.winCount} | Losses: ${meanRevResult.summary.lossCount}`);
    console.log(`  Win Rate: ${meanRevResult.summary.winRate.toFixed(1)}%`);
    console.log(`  Scaled Trades: ${meanRevResult.summary.scaledTrades}`);
    console.log(`\nP&L:`);
    console.log(`  Gross Profit: $${meanRevResult.summary.totalGrossProfit.toFixed(2)}`);
    console.log(`  Commission: -$${meanRevResult.summary.totalCommission.toFixed(2)}`);
    console.log(`  Net Profit: $${meanRevResult.summary.totalNetProfit.toFixed(2)}`);
    console.log(`  Max Drawdown: -$${meanRevResult.summary.maxDrawdown.toFixed(2)}`);
    console.log(`\nPer Trade:`);
    console.log(`  Avg Win: $${meanRevResult.summary.avgWin.toFixed(2)}`);
    console.log(`  Avg Loss: $${meanRevResult.summary.avgLoss.toFixed(2)}`);
    console.log(`  Profit Factor: ${meanRevResult.summary.profitFactor.toFixed(2)}`);
    console.log('-'.repeat(80));
  } catch (error: any) {
    console.error('\n✗ Mean Reversion backtest failed:', error.message);
  }

  console.log('\n' + '='.repeat(80));
  console.log('MICRO EURO BACKTESTS COMPLETE');
  console.log('='.repeat(80));
}

runEuroBacktests().catch(console.error);
