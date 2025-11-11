#!/usr/bin/env tsx
/**
 * SMA Crossover - Everything on 5-min bars
 * - SMA on 5-min bars
 * - RSI on 5-min bars
 * - Entry/Exit on 5-min closes
 */

import { RSI } from 'technicalindicators';
import { fetchTopstepXFuturesBars, fetchTopstepXFuturesMetadata } from './lib/topstepx';

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  exitReason: 'target' | 'stop' | 'signal' | 'session' | 'end_of_data';
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isTradingAllowed(timestamp: string | Date) {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 6) return false;
  if (day === 0 && minutes < WEEKEND_REOPEN_MINUTES) return false;
  if (day === 5 && minutes >= CUT_OFF_MINUTES) return false;

  const dailyOpen = minutes < CUT_OFF_MINUTES || minutes >= REOPEN_MINUTES;
  return dailyOpen;
}

function calculateSMA(values: number[], period: number): number | null {
  if (values.length < period) return null;
  const sum = values.slice(-period).reduce((a, b) => a + b, 0);
  return sum / period;
}

async function runSMA5MinAllBacktest(
  contractId: string,
  startDate: string,
  endDate: string,
  smaPeriod: number = 10,   // 10 x 5-min = 50 minutes
  rsiPeriod: number = 14,   // 14 x 5-min = 70 minutes
  stopLossPercent: number = 0.001,
  takeProfitPercent: number = 0.011,
  commission: number = 0.37,
) {
  console.log('\n' + '='.repeat(80));
  console.log('REAL-TIME SMA BACKTEST - ALL ON 5-MIN BARS');
  console.log('='.repeat(80));
  console.log(`Contract: ${contractId}`);
  console.log(`Period: ${startDate} to ${endDate}`);
  console.log(`SMA Period: ${smaPeriod} (5-min bars = ${smaPeriod * 5} minutes)`);
  console.log(`RSI Period: ${rsiPeriod} (5-min bars = ${rsiPeriod * 5} minutes)`);
  console.log('='.repeat(80));

  // Fetch contract metadata
  const metadata = await fetchTopstepXFuturesMetadata(contractId);
  if (!metadata) {
    throw new Error(`Contract ${contractId} not found`);
  }

  const pointValue = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : 50;

  console.log(`\nContract: ${metadata.name}`);
  console.log(`Point Value: $${pointValue}`);

  // Fetch 5-min bars
  const bars5min = await fetchTopstepXFuturesBars({
    contractId,
    startTime: startDate,
    endTime: endDate,
    unit: 2,
    unitNumber: 5,
  });

  if (!bars5min.length) {
    throw new Error('No 5-min bars returned');
  }

  bars5min.reverse();
  console.log(`\nLoaded ${bars5min.length} 5-min bars`);

  // Calculate RSI on 5-min bars
  const closes5min = bars5min.map(b => b.close);
  const rsiValues = RSI.calculate({ values: closes5min, period: rsiPeriod });

  // Backtest loop
  const closesForSMA: number[] = [];
  let position: 'long' | 'short' | null = null;
  let entryPrice = 0;
  let entryTime = '';
  let prevSMA: number | null = null;
  let prevClose: number | null = null;
  let prevRSI: number | null = null;
  const trades: TradeRecord[] = [];
  let realizedPnL = 0;

  const exitPosition = (exitPrice: number, exitTime: string, reason: TradeRecord['exitReason']) => {
    if (!position) return;
    const direction = position === 'long' ? 1 : -1;
    const rawPnl = (exitPrice - entryPrice) * direction * pointValue;
    const commissionCost = commission * 2;
    const pnl = rawPnl - commissionCost;
    trades.push({
      entryTime,
      exitTime,
      side: position,
      entryPrice,
      exitPrice,
      pnl,
      exitReason: reason,
    });
    realizedPnL += pnl;
    position = null;
    entryPrice = 0;
    entryTime = '';
  };

  for (let i = 0; i < bars5min.length; i++) {
    const bar = bars5min[i];
    closesForSMA.push(bar.close);

    // Calculate SMA
    const currentSMA = calculateSMA(closesForSMA, smaPeriod);
    if (!currentSMA) continue;

    // Get RSI (offset due to initial calculation period)
    const rsiOffset = closes5min.length - rsiValues.length;
    const rsiIndex = i - rsiOffset;
    if (rsiIndex < 0 || rsiIndex >= rsiValues.length) continue;
    const currentRSI = rsiValues[rsiIndex];

    // Check stop/target if in position
    if (position && isTradingAllowed(bar.timestamp)) {
      const direction = position === 'long' ? 1 : -1;
      const target = direction === 1
        ? entryPrice * (1 + takeProfitPercent)
        : entryPrice * (1 - takeProfitPercent);
      const stop = direction === 1
        ? entryPrice * (1 - stopLossPercent)
        : entryPrice * (1 + stopLossPercent);

      if (direction === 1 && bar.high >= target) {
        exitPosition(target, bar.timestamp, 'target');
      } else if (direction === 1 && bar.low <= stop) {
        exitPosition(stop, bar.timestamp, 'stop');
      } else if (direction === -1 && bar.low <= target) {
        exitPosition(target, bar.timestamp, 'target');
      } else if (direction === -1 && bar.high >= stop) {
        exitPosition(stop, bar.timestamp, 'stop');
      }
    }

    // Exit if session ends
    if (position && !isTradingAllowed(bar.timestamp)) {
      exitPosition(bar.close, bar.timestamp, 'session');
    }

    if (position) continue;

    // Need previous values
    if (prevSMA === null || prevClose === null || !prevRSI) {
      prevSMA = currentSMA;
      prevClose = bar.close;
      prevRSI = currentRSI;
      continue;
    }

    // Check for crossover
    const crossedUp = prevClose <= prevSMA && bar.close > currentSMA;
    const crossedDown = prevClose >= prevSMA && bar.close < currentSMA;

    const rsiBullish = currentRSI > 50 && currentRSI > prevRSI;
    const rsiBearish = currentRSI < 50 && currentRSI < prevRSI;

    // Enter on crossover + RSI confirmation
    if (crossedUp && rsiBullish && isTradingAllowed(bar.timestamp)) {
      position = 'long';
      entryPrice = bar.close;
      entryTime = bar.timestamp;
    } else if (crossedDown && rsiBearish && isTradingAllowed(bar.timestamp)) {
      position = 'short';
      entryPrice = bar.close;
      entryTime = bar.timestamp;
    }

    // Update previous values
    prevSMA = currentSMA;
    prevClose = bar.close;
    prevRSI = currentRSI;
  }

  // Close any open position
  if (position) {
    const lastBar = bars5min[bars5min.length - 1];
    exitPosition(lastBar.close, lastBar.timestamp, 'end_of_data');
  }

  // Calculate stats
  const wins = trades.filter(t => t.pnl > 0).length;
  const losses = trades.filter(t => t.pnl < 0).length;
  const avgWin = wins ? trades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0) / wins : 0;
  const avgLoss = losses ? trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0) / losses : 0;

  let runningPnL = 0;
  let peakPnL = 0;
  let maxDrawdown = 0;
  trades.forEach(trade => {
    runningPnL += trade.pnl;
    if (runningPnL > peakPnL) peakPnL = runningPnL;
    const drawdown = peakPnL - runningPnL;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  });

  console.log('\n' + '='.repeat(80));
  console.log('RESULTS');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${trades.length}`);
  console.log(`Wins: ${wins} | Losses: ${losses}`);
  console.log(`Win Rate: ${trades.length ? ((wins / trades.length) * 100).toFixed(1) : '0'}%`);
  console.log(`\nRealized PnL: $${realizedPnL.toFixed(2)}`);
  console.log(`Avg Win: $${avgWin.toFixed(2)}`);
  console.log(`Avg Loss: $${avgLoss.toFixed(2)}`);
  console.log(`Max Drawdown: $${maxDrawdown.toFixed(2)}`);

  console.log('\n' + '='.repeat(80));
  console.log('SAMPLE TRADES (Last 5)');
  console.log('='.repeat(80));
  trades.slice(-5).forEach((trade, idx) => {
    const sign = trade.pnl >= 0 ? '+' : '';
    console.log(`${trades.length - 5 + idx + 1}. ${trade.side.toUpperCase()} - ${trade.exitReason}`);
    console.log(`   Entry: ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)}`);
    console.log(`   Exit:  ${trade.exitTime} @ ${trade.exitPrice.toFixed(2)}`);
    console.log(`   PnL: ${sign}$${trade.pnl.toFixed(2)}`);
  });

  console.log('\n' + '='.repeat(80));

  return { trades, summary: { totalTrades: trades.length, wins, losses, winRate: trades.length ? (wins / trades.length) * 100 : 0, totalPnL: realizedPnL, avgWin, avgLoss, maxDrawdown } };
}

async function main() {
  const START = '2025-10-01T00:00:00Z';
  const END = '2025-10-31T23:59:59Z';

  console.log('\nTESTING: MICRO NASDAQ (MNQ) - SMA(10) + RSI(14) on 5-min\n');
  await runSMA5MinAllBacktest('CON.F.US.MNQ.Z25', START, END, 10, 14, 0.001, 0.011, 0.37);

  console.log('\n\nTESTING: MICRO E-MINI S&P (MES) - SMA(10) + RSI(14) on 5-min\n');
  await runSMA5MinAllBacktest('CON.F.US.MES.Z25', START, END, 10, 14, 0.001, 0.011, 0.37);
}

main().catch(err => {
  console.error('\n✗ Backtest failed:', err.message);
  process.exit(1);
});
