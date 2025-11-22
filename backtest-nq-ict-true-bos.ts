#!/usr/bin/env tsx
/**
 * NQ ICT Strategy Backtest - TRUE Break of Structure (Internal + External)
 *
 * Strategy Logic (Proper ICT/SMC BOS):
 * - Break of Structure (BOS) Detection:
 *   - Identify swing highs/lows using pivot detection (3-5 bar pivots)
 *   - Internal BOS: Break of most recent swing high/low
 *   - External BOS: Break of a significant previous swing level
 *   - Must have pullback structure (not just higher highs/lower lows)
 *
 * - Wicked Candle Detection (rejection confirmation):
 *   - Bullish: Bottom wick >60% of range, close in top 40%, top wick <20%
 *   - Bearish: Top wick >60% of range, close in bottom 40%, bottom wick <20%
 *
 * - Entry Signal: TRUE BOS + Wicked Candle (same direction)
 * - Entry: Market entry on signal bar close
 *
 * Risk Parameters:
 * - Symbol: NQZ5 (full-size Nasdaq)
 * - Stop Loss: 4 ticks (current live config)
 * - Take Profit: 16 ticks (TP1), 32 ticks (TP2) - scaled exits
 * - Contracts: 3
 * - Tick Size: 0.25
 * - Tick Value: $5 per tick
 *
 * Backtest Settings:
 * - Date Range: Last 14 days (Nov 1-14, 2025)
 * - Pivot Length: 5 bars (left + right validation)
 * - No session filters
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
const STOP_LOSS_TICKS = 4;
const TAKE_PROFIT_1_TICKS = 16;
const TAKE_PROFIT_2_TICKS = 32;
const NUM_CONTRACTS = 3;
const TICK_SIZE = 0.25;
const TICK_VALUE = 5; // $5 per tick for NQ
const PIVOT_LENGTH = 5; // Bars on each side for pivot validation
const SCALE_OUT_PERCENT = 0.5; // 50% exit at TP1

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

// Swing Point Interface
interface SwingPoint {
  index: number;
  price: number;
  type: 'high' | 'low';
  timestamp: string;
}

// Detect Swing Highs and Lows using pivot detection
function detectSwingPoints(bars: TopstepXFuturesBar[], pivotLength: number): SwingPoint[] {
  const swings: SwingPoint[] = [];

  for (let i = pivotLength; i < bars.length - pivotLength; i++) {
    const bar = bars[i];

    // Check for swing high (current high is highest in surrounding bars)
    let isSwingHigh = true;
    for (let j = i - pivotLength; j <= i + pivotLength; j++) {
      if (j !== i && bars[j].high >= bar.high) {
        isSwingHigh = false;
        break;
      }
    }
    if (isSwingHigh) {
      swings.push({
        index: i,
        price: bar.high,
        type: 'high',
        timestamp: bar.timestamp,
      });
    }

    // Check for swing low (current low is lowest in surrounding bars)
    let isSwingLow = true;
    for (let j = i - pivotLength; j <= i + pivotLength; j++) {
      if (j !== i && bars[j].low <= bar.low) {
        isSwingLow = false;
        break;
      }
    }
    if (isSwingLow) {
      swings.push({
        index: i,
        price: bar.low,
        type: 'low',
        timestamp: bar.timestamp,
      });
    }
  }

  return swings.sort((a, b) => a.index - b.index);
}

// Detect TRUE Break of Structure
function detectTrueBOS(
  bars: TopstepXFuturesBar[],
  currentIndex: number,
  swingPoints: SwingPoint[]
): { type: 'bullish' | 'bearish'; level: number; internal: boolean } | null {
  const currentBar = bars[currentIndex];

  // Get swing points before current bar
  const previousSwings = swingPoints.filter(s => s.index < currentIndex);
  if (previousSwings.length < 2) return null;

  // Bullish BOS: Close above a previous swing high
  const swingHighs = previousSwings.filter(s => s.type === 'high');
  if (swingHighs.length > 0) {
    // Check for break of most recent swing high (internal structure)
    const mostRecentHigh = swingHighs[swingHighs.length - 1];
    if (currentBar.close > mostRecentHigh.price) {
      return {
        type: 'bullish',
        level: mostRecentHigh.price,
        internal: true,
      };
    }

    // Check for break of significant previous swing high (external structure)
    for (let i = swingHighs.length - 2; i >= Math.max(0, swingHighs.length - 5); i--) {
      const swingHigh = swingHighs[i];
      if (currentBar.close > swingHigh.price && swingHigh.price > mostRecentHigh.price) {
        return {
          type: 'bullish',
          level: swingHigh.price,
          internal: false,
        };
      }
    }
  }

  // Bearish BOS: Close below a previous swing low
  const swingLows = previousSwings.filter(s => s.type === 'low');
  if (swingLows.length > 0) {
    // Check for break of most recent swing low (internal structure)
    const mostRecentLow = swingLows[swingLows.length - 1];
    if (currentBar.close < mostRecentLow.price) {
      return {
        type: 'bearish',
        level: mostRecentLow.price,
        internal: true,
      };
    }

    // Check for break of significant previous swing low (external structure)
    for (let i = swingLows.length - 2; i >= Math.max(0, swingLows.length - 5); i--) {
      const swingLow = swingLows[i];
      if (currentBar.close < swingLow.price && swingLow.price < mostRecentLow.price) {
        return {
          type: 'bearish',
          level: swingLow.price,
          internal: false,
        };
      }
    }
  }

  return null;
}

// Wicked Candle Detection
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
  exitReason: 'stop' | 'tp1' | 'tp2';
  pattern: string;
  bosType: string;
}

// Active Position Interface
interface ActivePosition {
  entryTime: string;
  entryPrice: number;
  side: 'long' | 'short';
  stopLoss: number;
  takeProfit1: number;
  takeProfit2: number;
  remainingContracts: number;
  pattern: string;
  bosType: string;
}

// Main Backtest Function
async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('NQ ICT STRATEGY BACKTEST - TRUE BREAK OF STRUCTURE (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Period: ${START_DATE.toISOString()} -> ${END_DATE.toISOString()}`);
  console.log(`Strategy: TRUE BOS (Internal/External) + Wicked Candle`);
  console.log(`Pivot Length: ${PIVOT_LENGTH} bars | BOS Type: Swing-based structure breaks`);
  console.log(`Stop Loss: ${STOP_LOSS_TICKS} ticks | TP1: ${TAKE_PROFIT_1_TICKS} ticks (${SCALE_OUT_PERCENT * 100}%) | TP2: ${TAKE_PROFIT_2_TICKS} ticks`);
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

  // Detect swing points
  console.log(`Detecting swing points (pivot length: ${PIVOT_LENGTH})...`);
  const swingPoints = detectSwingPoints(bars, PIVOT_LENGTH);
  console.log(`Identified ${swingPoints.length} swing points (${swingPoints.filter(s => s.type === 'high').length} highs, ${swingPoints.filter(s => s.type === 'low').length} lows)`);

  // Backtest variables
  const trades: TradeRecord[] = [];
  let activePosition: ActivePosition | null = null;
  let internalBOSCount = 0;
  let externalBOSCount = 0;

  // Main loop
  console.log('\nRunning backtest...\n');
  for (let i = PIVOT_LENGTH + 1; i < bars.length; i++) {
    const bar = bars[i];

    // Check for exit if we have an active position
    if (activePosition) {
      let exitPrice: number | null = null;
      let exitReason: 'stop' | 'tp1' | 'tp2' | null = null;
      let contractsToClose = 0;

      if (activePosition.side === 'long') {
        // Check stop loss
        if (bar.low <= activePosition.stopLoss) {
          exitPrice = fillStop(baseSymbol, 'sell', activePosition.stopLoss);
          exitReason = 'stop';
          contractsToClose = activePosition.remainingContracts;
        }
        // Check TP2 first (full position if remaining)
        else if (bar.high >= activePosition.takeProfit2 && activePosition.remainingContracts > 0) {
          exitPrice = fillTP(baseSymbol, 'sell', activePosition.takeProfit2);
          exitReason = 'tp2';
          contractsToClose = activePosition.remainingContracts;
        }
        // Check TP1 (partial exit)
        else if (bar.high >= activePosition.takeProfit1 && activePosition.remainingContracts === NUM_CONTRACTS) {
          exitPrice = fillTP(baseSymbol, 'sell', activePosition.takeProfit1);
          exitReason = 'tp1';
          contractsToClose = NUM_CONTRACTS * SCALE_OUT_PERCENT;
        }
      } else { // short
        // Check stop loss
        if (bar.high >= activePosition.stopLoss) {
          exitPrice = fillStop(baseSymbol, 'buy', activePosition.stopLoss);
          exitReason = 'stop';
          contractsToClose = activePosition.remainingContracts;
        }
        // Check TP2 first
        else if (bar.low <= activePosition.takeProfit2 && activePosition.remainingContracts > 0) {
          exitPrice = fillTP(baseSymbol, 'buy', activePosition.takeProfit2);
          exitReason = 'tp2';
          contractsToClose = activePosition.remainingContracts;
        }
        // Check TP1 (partial exit)
        else if (bar.low <= activePosition.takeProfit1 && activePosition.remainingContracts === NUM_CONTRACTS) {
          exitPrice = fillTP(baseSymbol, 'buy', activePosition.takeProfit1);
          exitReason = 'tp1';
          contractsToClose = NUM_CONTRACTS * SCALE_OUT_PERCENT;
        }
      }

      if (exitPrice && exitReason && contractsToClose > 0) {
        // Close position (full or partial)
        const pointsGained = activePosition.side === 'long'
          ? (exitPrice - activePosition.entryPrice)
          : (activePosition.entryPrice - exitPrice);
        const ticksGained = pointsGained / TICK_SIZE;
        const grossPnl = ticksGained * TICK_VALUE * contractsToClose;
        const fees = addFees(baseSymbol, contractsToClose) * 2; // Entry + exit
        const netPnl = grossPnl - fees;

        trades.push({
          entryTime: activePosition.entryTime,
          exitTime: bar.timestamp,
          side: activePosition.side,
          entryPrice: activePosition.entryPrice,
          exitPrice,
          pnl: netPnl,
          grossPnl,
          fees,
          exitReason,
          pattern: activePosition.pattern,
          bosType: activePosition.bosType,
        });

        // Update remaining contracts or close position
        if (exitReason === 'tp1') {
          activePosition.remainingContracts -= contractsToClose;
        } else {
          activePosition = null;
        }
      }
    }

    // Check for new entry if no active position
    if (!activePosition) {
      const bos = detectTrueBOS(bars, i, swingPoints);
      const wicked = detectWickedCandle(bar);

      if (bos && wicked && bos.type === wicked.type) {
        const side = bos.type === 'bullish' ? 'long' : 'short';
        const entryPrice = fillEntry(baseSymbol, side === 'long' ? 'buy' : 'sell', bar.close);
        const stopLoss = side === 'long'
          ? roundToTick(entryPrice - (STOP_LOSS_TICKS * TICK_SIZE), TICK_SIZE)
          : roundToTick(entryPrice + (STOP_LOSS_TICKS * TICK_SIZE), TICK_SIZE);
        const takeProfit1 = side === 'long'
          ? roundToTick(entryPrice + (TAKE_PROFIT_1_TICKS * TICK_SIZE), TICK_SIZE)
          : roundToTick(entryPrice - (TAKE_PROFIT_1_TICKS * TICK_SIZE), TICK_SIZE);
        const takeProfit2 = side === 'long'
          ? roundToTick(entryPrice + (TAKE_PROFIT_2_TICKS * TICK_SIZE), TICK_SIZE)
          : roundToTick(entryPrice - (TAKE_PROFIT_2_TICKS * TICK_SIZE), TICK_SIZE);

        const bosType = bos.internal ? 'Internal BOS' : 'External BOS';
        if (bos.internal) internalBOSCount++;
        else externalBOSCount++;

        activePosition = {
          entryTime: bar.timestamp,
          entryPrice,
          side,
          stopLoss,
          takeProfit1,
          takeProfit2,
          remainingContracts: NUM_CONTRACTS,
          pattern: `${wicked.type} wick + ${bosType}`,
          bosType,
        };
      }
    }
  }

  // Close any open position at the end
  if (activePosition) {
    const lastBar = bars[bars.length - 1];
    const exitPrice = activePosition.side === 'long' ? lastBar.close : lastBar.close;
    const pointsGained = activePosition.side === 'long'
      ? (exitPrice - activePosition.entryPrice)
      : (activePosition.entryPrice - exitPrice);
    const ticksGained = pointsGained / TICK_SIZE;
    const grossPnl = ticksGained * TICK_VALUE * activePosition.remainingContracts;
    const fees = addFees(baseSymbol, activePosition.remainingContracts) * 2;
    const netPnl = grossPnl - fees;

    trades.push({
      entryTime: activePosition.entryTime,
      exitTime: lastBar.timestamp,
      side: activePosition.side,
      entryPrice: activePosition.entryPrice,
      exitPrice,
      pnl: netPnl,
      grossPnl,
      fees,
      exitReason: 'stop',
      pattern: activePosition.pattern,
      bosType: activePosition.bosType,
    });
  }

  // Calculate statistics
  const totalTrades = trades.length;
  const wins = trades.filter(t => t.pnl > 0);
  const losses = trades.filter(t => t.pnl <= 0);
  const winRate = totalTrades > 0 ? (wins.length / totalTrades) * 100 : 0;
  const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
  const totalGrossPnL = trades.reduce((sum, t) => sum + t.grossPnl, 0);
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);
  const avgWin = wins.length > 0 ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0;
  const profitFactor = losses.reduce((sum, t) => sum + Math.abs(t.pnl), 0) > 0
    ? wins.reduce((sum, t) => sum + t.pnl, 0) / losses.reduce((sum, t) => sum + Math.abs(t.pnl), 0)
    : Infinity;

  // Calculate max drawdown
  let equity = 0;
  let peak = 0;
  let maxDrawdown = 0;
  for (const trade of trades) {
    equity += trade.pnl;
    if (equity > peak) peak = equity;
    const drawdown = peak - equity;
    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
  }

  const stopExits = trades.filter(t => t.exitReason === 'stop').length;
  const tp1Exits = trades.filter(t => t.exitReason === 'tp1').length;
  const tp2Exits = trades.filter(t => t.exitReason === 'tp2').length;

  const dayCount = (END_DATE.getTime() - START_DATE.getTime()) / (1000 * 60 * 60 * 24);
  const tradesPerDay = totalTrades / dayCount;

  // BOS breakdown
  const internalBOSTrades = trades.filter(t => t.bosType === 'Internal BOS');
  const externalBOSTrades = trades.filter(t => t.bosType === 'External BOS');
  const internalWins = internalBOSTrades.filter(t => t.pnl > 0).length;
  const externalWins = externalBOSTrades.filter(t => t.pnl > 0).length;
  const internalWinRate = internalBOSTrades.length > 0 ? (internalWins / internalBOSTrades.length) * 100 : 0;
  const externalWinRate = externalBOSTrades.length > 0 ? (externalWins / externalBOSTrades.length) * 100 : 0;

  // Print results
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST RESULTS - TRUE BOS (SWING-BASED STRUCTURE)');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${totalTrades} | Wins: ${wins.length} | Losses: ${losses.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net PnL: $${totalPnL.toFixed(2)} | Gross PnL: $${totalGrossPnL.toFixed(2)} | Fees: $${totalFees.toFixed(2)}`);
  console.log(`Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: $${maxDrawdown.toFixed(2)}`);
  console.log(`Average Win: $${avgWin.toFixed(2)} | Average Loss: $${avgLoss.toFixed(2)}`);
  console.log(`Trades Per Day: ${tradesPerDay.toFixed(2)}`);
  console.log(`Exit Reasons: Stop=${stopExits}, TP1=${tp1Exits}, TP2=${tp2Exits}`);
  console.log('\n--- BOS Type Breakdown ---');
  console.log(`Internal BOS: ${internalBOSTrades.length} trades (${internalWins} wins, ${internalWinRate.toFixed(1)}% WR)`);
  console.log(`External BOS: ${externalBOSTrades.length} trades (${externalWins} wins, ${externalWinRate.toFixed(1)}% WR)`);
  console.log('='.repeat(80));

  // Comparison
  console.log('\n' + '='.repeat(80));
  console.log('COMPARISON TO SIMPLE 3-BAR BOS (Current Strategy)');
  console.log('='.repeat(80));
  console.log('Current Strategy (3-bar BOS + Wicked, 4/16/32):');
  console.log('  - Win Rate: 70.9%');
  console.log('  - Net PnL: +$53,609 (14 days)');
  console.log('  - Profit Factor: 5.71');
  console.log('  - Trades: 536 (~19/day)');
  console.log('\nNew Strategy (TRUE BOS + Wicked, 4/16/32):');
  console.log(`  - Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`  - Net PnL: $${totalPnL.toFixed(2)} (14 days)`);
  console.log(`  - Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`  - Trades: ${totalTrades} (~${tradesPerDay.toFixed(1)}/day)`);
  console.log('='.repeat(80));
}

runBacktest().catch(console.error);
