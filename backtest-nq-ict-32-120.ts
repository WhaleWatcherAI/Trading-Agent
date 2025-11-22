#!/usr/bin/env tsx
/**
 * NQ ICT Strategy Backtest - 1-Minute Bars
 *
 * Strategy Logic (matching live-topstepx-nq-ict.ts):
 * - Entry Signal: Wicked Candle + BOS (Break of Structure) SAME BAR + SAME DIRECTION
 * - Wicked Candle Detection:
 *   - Bullish: Bottom wick >60% of range, close in top 40%, top wick <20%
 *   - Bearish: Top wick >60% of range, close in bottom 40%, bottom wick <20%
 * - BOS Detection:
 *   - Bullish: Close > max(last 3 bars high)
 *   - Bearish: Close < min(last 3 bars low)
 * - Entry: Market entry on signal bar close
 *
 * Risk Parameters:
 * - Symbol: NQZ5 (full-size Nasdaq)
 * - Stop Loss: 32 ticks
 * - Take Profit: 120 ticks (single exit, no scaling)
 * - Contracts: 3
 * - Tick Size: 0.25
 * - Tick Value: $5 per tick
 *
 * Backtest Settings:
 * - Date Range: Last 14 days (Nov 1-14, 2025)
 * - No session filters
 * - No HTF confirmation
 * - Includes slippage from slip-config.json
 * - Includes commissions
 */

import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';

// Configuration
const SYMBOL = 'NQZ5';
const STOP_LOSS_TICKS = 32;
const TAKE_PROFIT_TICKS = 120;
const NUM_CONTRACTS = 3;
const TICK_SIZE = 0.25;
const TICK_VALUE = 5; // $5 per tick for NQ

// Date range: Last 14 days (Nov 1-14, 2025)
const END_DATE = new Date('2025-11-14T23:59:59Z');
const START_DATE = new Date('2025-11-01T00:00:00Z');

// Load slippage configuration
interface SlippageConfig {
  tickSize: Record<string, number>;
  slipAvg: {
    entry: Record<string, number>;
    tp: Record<string, number>;
    stop: Record<string, number>;
  };
  avgSpreadTicks: Record<string, number>;
  feesPerSideUSD: Record<string, number>;
  p_tp_passive: Record<string, number>;
}

const loadSlippageConfig = (): SlippageConfig => {
  const configPath = path.join(__dirname, 'slip-config.json');
  const configData = fs.readFileSync(configPath, 'utf-8');
  return JSON.parse(configData);
};

const SLIP_CONFIG = loadSlippageConfig();

// Get base symbol for slippage lookup
const getBaseSymbol = (fullSymbol: string): string => {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
};

// Realistic fill simulation functions
function fillEntry(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const S = 0.5 * t * SLIP_CONFIG.avgSpreadTicks[sym];
  const sigma = SLIP_CONFIG.slipAvg.entry[sym] * t;
  return side === 'buy' ? mid + S + sigma : mid - S - sigma;
}

function fillTP(sym: string, side: 'buy' | 'sell', mid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const S_ticks = SLIP_CONFIG.avgSpreadTicks[sym];
  const sigma_tp_ticks = SLIP_CONFIG.slipAvg.tp[sym];
  const p_passive = SLIP_CONFIG.p_tp_passive[sym];
  const E_tp_ticks = (1 - p_passive) * (S_ticks + sigma_tp_ticks);
  return side === 'sell' ? mid - E_tp_ticks * t : mid + E_tp_ticks * t;
}

function fillStop(sym: string, side: 'buy' | 'sell', triggerMid: number): number {
  const t = SLIP_CONFIG.tickSize[sym];
  const sigma = SLIP_CONFIG.slipAvg.stop[sym] * t;
  return side === 'buy' ? triggerMid + sigma : triggerMid - sigma;
}

function addFees(sym: string, contracts: number): number {
  return SLIP_CONFIG.feesPerSideUSD[sym] * contracts;
}

function roundToTick(price: number, tickSize: number): number {
  return Math.round(price / tickSize) * tickSize;
}

// Pattern Detection Functions
const BOS_SWING_LOOKBACK = 3;

function findLastSwingHigh(
  bars: TopstepXFuturesBar[],
  index: number,
  lookback: number,
): { level: number; pivotIndex: number } | null {
  if (index < lookback * 2 + 1) {
    return null;
  }

  for (let i = index - lookback; i >= lookback; i -= 1) {
    const candidate = bars[i];
    let isSwingHigh = true;

    for (let j = i - lookback; j <= i + lookback; j += 1) {
      if (bars[j].high > candidate.high) {
        isSwingHigh = false;
        break;
      }
    }

    if (isSwingHigh) {
      return { level: candidate.high, pivotIndex: i };
    }
  }

  return null;
}

function findLastSwingLow(
  bars: TopstepXFuturesBar[],
  index: number,
  lookback: number,
): { level: number; pivotIndex: number } | null {
  if (index < lookback * 2 + 1) {
    return null;
  }

  for (let i = index - lookback; i >= lookback; i -= 1) {
    const candidate = bars[i];
    let isSwingLow = true;

    for (let j = i - lookback; j <= i + lookback; j += 1) {
      if (bars[j].low < candidate.low) {
        isSwingLow = false;
        break;
      }
    }

    if (isSwingLow) {
      return { level: candidate.low, pivotIndex: i };
    }
  }

  return null;
}

function detectBOS(bars: TopstepXFuturesBar[], index: number): { type: 'bullish' | 'bearish'; level: number } | null {
  if (index < BOS_SWING_LOOKBACK * 2 + 1) {
    return null;
  }

  const swingHigh = findLastSwingHigh(bars, index, BOS_SWING_LOOKBACK);
  const swingLow = findLastSwingLow(bars, index, BOS_SWING_LOOKBACK);
  const currentBar = bars[index];

  // Bullish BOS: close breaks above last confirmed swing high, with bullish candle body
  if (swingHigh && currentBar.close > swingHigh.level && currentBar.close > currentBar.open) {
    return { type: 'bullish', level: swingHigh.level };
  }

  // Bearish BOS: close breaks below last confirmed swing low, with bearish candle body
  if (swingLow && currentBar.close < swingLow.level && currentBar.close < currentBar.open) {
    return { type: 'bearish', level: swingLow.level };
  }

  return null;
}

function detectWickedCandle(bar: TopstepXFuturesBar): { type: 'bullish' | 'bearish'; strength: number } | null {
  const range = bar.high - bar.low;
  if (range < 0.01) return null;

  const closePercent = (bar.close - bar.low) / range;
  const topWickPercent = (bar.high - Math.max(bar.open, bar.close)) / range;
  const bottomWickPercent = (Math.min(bar.open, bar.close) - bar.low) / range;

  // Bullish: Bottom wick >60%, close in top 40%, top wick <20%
  if (topWickPercent < 0.2 && bottomWickPercent > 0.6 && closePercent > 0.6) {
    return { type: 'bullish', strength: bottomWickPercent };
  }

  // Bearish: Top wick >60%, close in bottom 40%, bottom wick <20%
  if (bottomWickPercent < 0.2 && topWickPercent > 0.6 && closePercent < 0.4) {
    return { type: 'bearish', strength: topWickPercent };
  }

  return null;
}

// Trade Record Interface
interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'stop' | 'target';
  pattern: string;
}

// Main Backtest Function
async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('NQ ICT STRATEGY BACKTEST (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Period: ${START_DATE.toISOString()} -> ${END_DATE.toISOString()}`);
  console.log(`Strategy: Wicked Candle + BOS (same bar, same direction)`);
  console.log(`Stop Loss: ${STOP_LOSS_TICKS} ticks | Take Profit: ${TAKE_PROFIT_TICKS} ticks`);
  console.log(`Contracts: ${NUM_CONTRACTS} | Tick Size: ${TICK_SIZE} | Tick Value: $${TICK_VALUE}`);
  console.log('='.repeat(80));

  const baseSymbol = getBaseSymbol(SYMBOL);

  // Fetch metadata
  const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${SYMBOL}`);
  }

  console.log(`Resolved contract: ${metadata.name} (${metadata.id})`);
  console.log(`Slippage (${baseSymbol}): Entry=${SLIP_CONFIG.slipAvg.entry[baseSymbol]} ticks, TP=${SLIP_CONFIG.slipAvg.tp[baseSymbol]} ticks, Stop=${SLIP_CONFIG.slipAvg.stop[baseSymbol]} ticks`);
  console.log(`Fees: $${SLIP_CONFIG.feesPerSideUSD[baseSymbol]} per side per contract`);

  // Fetch 1-minute bars
  console.log('\nFetching 1-minute bars...');
  const bars = await fetchTopstepXFuturesBars({
    contractId: metadata.id,
    startTime: START_DATE.toISOString(),
    endTime: END_DATE.toISOString(),
    unit: 2, // Minutes
    unitNumber: 1, // 1-minute bars
    limit: 50000,
  });

  if (!bars.length) {
    throw new Error('No bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-minute bars`);

  const trades: TradeRecord[] = [];
  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stopLoss: number;
    target: number;
    pattern: string;
  } | null = null;

  // Process each bar
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];

    // Check if position needs to be exited
    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // Check stop loss
      const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                      (direction === -1 && bar.high >= position.stopLoss);

      if (hitStop) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss), TICK_SIZE);
        const rawPnl = (exitPrice - position.entryPrice) * direction * TICK_VALUE * NUM_CONTRACTS / TICK_SIZE;
        const fees = addFees(baseSymbol, NUM_CONTRACTS) * 2; // Entry + exit
        const netPnl = rawPnl - fees;

        trades.push({
          entryTime: position.entryTime,
          exitTime: bar.timestamp,
          side: position.side,
          entryPrice: position.entryPrice,
          exitPrice,
          pnl: netPnl,
          grossPnl: rawPnl,
          fees,
          exitReason: 'stop',
          pattern: position.pattern,
        });

        console.log(
          `[${bar.timestamp}] STOP LOSS: ${position.side.toUpperCase()} @ ${exitPrice.toFixed(2)} | ` +
          `PnL: $${netPnl.toFixed(2)}`
        );

        position = null;
        continue;
      }

      // Check take profit
      const hitTarget = (direction === 1 && bar.high >= position.target) ||
                        (direction === -1 && bar.low <= position.target);

      if (hitTarget) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillTP(baseSymbol, closeSide, position.target), TICK_SIZE);
        const rawPnl = (exitPrice - position.entryPrice) * direction * TICK_VALUE * NUM_CONTRACTS / TICK_SIZE;
        const fees = addFees(baseSymbol, NUM_CONTRACTS) * 2; // Entry + exit
        const netPnl = rawPnl - fees;

        trades.push({
          entryTime: position.entryTime,
          exitTime: bar.timestamp,
          side: position.side,
          entryPrice: position.entryPrice,
          exitPrice,
          pnl: netPnl,
          grossPnl: rawPnl,
          fees,
          exitReason: 'target',
          pattern: position.pattern,
        });

        console.log(
          `[${bar.timestamp}] TAKE PROFIT: ${position.side.toUpperCase()} @ ${exitPrice.toFixed(2)} | ` +
          `PnL: $${netPnl.toFixed(2)}`
        );

        position = null;
        continue;
      }

      continue;
    }

    // Look for entry signals (no position active)
    const bos = detectBOS(bars, i);
    const wicked = detectWickedCandle(bar);

    // Entry condition: Wicked Candle + BOS on SAME bar with SAME direction
    if (wicked && bos && wicked.type === bos.type) {
      const side = wicked.type === 'bullish' ? 'long' : 'short';
      const entrySide = side === 'long' ? 'buy' : 'sell';
      const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close), TICK_SIZE);

      position = {
        side,
        entryPrice,
        entryTime: bar.timestamp,
        stopLoss: roundToTick(
          side === 'long'
            ? entryPrice - (STOP_LOSS_TICKS * TICK_SIZE)
            : entryPrice + (STOP_LOSS_TICKS * TICK_SIZE),
          TICK_SIZE
        ),
        target: roundToTick(
          side === 'long'
            ? entryPrice + (TAKE_PROFIT_TICKS * TICK_SIZE)
            : entryPrice - (TAKE_PROFIT_TICKS * TICK_SIZE),
          TICK_SIZE
        ),
        pattern: `${wicked.type} wicked + ${bos.type} BOS`,
      };

      console.log(
        `[${bar.timestamp}] ENTRY: ${side.toUpperCase()} @ ${entryPrice.toFixed(2)} | ` +
        `SL: ${position.stopLoss.toFixed(2)} (${STOP_LOSS_TICKS}t) | ` +
        `TP: ${position.target.toFixed(2)} (${TAKE_PROFIT_TICKS}t) | ` +
        `Pattern: ${position.pattern}`
      );
    }
  }

  // Close any remaining position at end of data
  if (position) {
    const lastBar = bars[bars.length - 1];
    const closeSide = position.side === 'long' ? 'sell' : 'buy';
    const exitPrice = roundToTick(fillStop(baseSymbol, closeSide, lastBar.close), TICK_SIZE);
    const direction = position.side === 'long' ? 1 : -1;
    const rawPnl = (exitPrice - position.entryPrice) * direction * TICK_VALUE * NUM_CONTRACTS / TICK_SIZE;
    const fees = addFees(baseSymbol, NUM_CONTRACTS) * 2;
    const netPnl = rawPnl - fees;

    trades.push({
      entryTime: position.entryTime,
      exitTime: lastBar.timestamp,
      side: position.side,
      entryPrice: position.entryPrice,
      exitPrice,
      pnl: netPnl,
      grossPnl: rawPnl,
      fees,
      exitReason: 'stop',
      pattern: position.pattern,
    });

    position = null;
  }

  // Calculate Statistics
  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

  const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
  const totalGrossPnL = trades.reduce((sum, t) => sum + t.grossPnl, 0);
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);

  const avgWin = winningTrades.length > 0
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length > 0
    ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length)
    : 0;

  const grossProfit = winningTrades.reduce((sum, t) => sum + t.grossPnl, 0);
  const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.grossPnl, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

  // Calculate max drawdown
  let runningPnL = 0;
  let peakPnL = 0;
  let maxDrawdown = 0;

  trades.forEach(trade => {
    runningPnL += trade.pnl;
    if (runningPnL > peakPnL) {
      peakPnL = runningPnL;
    }
    const drawdown = peakPnL - runningPnL;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  });

  // Calculate trades per day
  const totalDays = (END_DATE.getTime() - START_DATE.getTime()) / (1000 * 60 * 60 * 24);
  const tradesPerDay = totalDays > 0 ? trades.length / totalDays : 0;

  // Print Results
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST RESULTS');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${trades.length}`);
  console.log(`Winning Trades: ${winningTrades.length}`);
  console.log(`Losing Trades: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`\nNet PnL: $${totalPnL.toFixed(2)}`);
  console.log(`Gross PnL: $${totalGrossPnL.toFixed(2)}`);
  console.log(`Total Fees: $${totalFees.toFixed(2)}`);
  console.log(`\nProfit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: $${maxDrawdown.toFixed(2)}`);
  console.log(`\nAverage Win: $${avgWin.toFixed(2)}`);
  console.log(`Average Loss: $${avgLoss.toFixed(2)}`);
  console.log(`\nTrades Per Day: ${tradesPerDay.toFixed(2)}`);

  // Exit reason breakdown
  const exitReasons = trades.reduce((acc, t) => {
    acc[t.exitReason] = (acc[t.exitReason] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`\nExit Reasons: ${Object.entries(exitReasons).map(([r, c]) => `${r}=${c}`).join(', ')}`);

  // Print recent trades
  if (trades.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RECENT TRADES (Last 10)');
    console.log('='.repeat(80));
    trades.slice(-10).forEach(trade => {
      const pnlStr = trade.pnl >= 0 ? `+$${trade.pnl.toFixed(2)}` : `-$${Math.abs(trade.pnl).toFixed(2)}`;
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${pnlStr.padStart(12)} (${trade.exitReason})`
      );
    });
  }

  console.log('='.repeat(80));
}

// Run the backtest
runBacktest().catch(err => {
  console.error('Backtest failed:', err);
  process.exit(1);
});
