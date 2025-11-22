#!/usr/bin/env tsx
/**
 * NQ ICT Strategy Backtest - Bollinger Band + Wicked Candle (Mean Reversion)
 *
 * Strategy Logic:
 * - Entry Signal: Wicked Candle that breaches Bollinger Bands (NO BOS required)
 * - Bollinger Bands: 20-period, 2 standard deviations
 * - Wicked Candle Detection:
 *   - Bullish: Bottom wick >60% of range, close in top 40%, top wick <20%
 *   - Bearish: Top wick >60% of range, close in bottom 40%, bottom wick <20%
 * - BB Breach:
 *   - Long: Wicked bullish candle where LOW wicks BELOW lower BB
 *   - Short: Wicked bearish candle where HIGH wicks ABOVE upper BB
 * - Entry: Market entry on signal bar close
 *
 * Risk Parameters:
 * - Symbol: NQZ5 (full-size Nasdaq)
 * - Stop Loss: 8 ticks
 * - Take Profit: 33 ticks (single exit, no scaling)
 * - Contracts: 3
 * - Tick Size: 0.25
 * - Tick Value: $5 per tick
 *
 * Backtest Settings:
 * - Date Range: Last 14 days (Nov 1-14, 2025)
 * - No session filters
 * - Includes slippage from slip-config.json
 * - Includes commissions
 */

import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import { BollingerBands } from 'technicalindicators';
import * as fs from 'fs';
import * as path from 'path';

// Configuration
const SYMBOL = 'NQZ5';
const STOP_LOSS_TICKS = 8;
const TAKE_PROFIT_TICKS = 33;
const NUM_CONTRACTS = 3;
const TICK_SIZE = 0.25;
const TICK_VALUE = 5; // $5 per tick for NQ
const BB_PERIOD = 20;
const BB_STDDEV = 2.0;

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

// Calculate Bollinger Bands
function calculateBollingerBands(bars: TopstepXFuturesBar[]): Array<{ upper: number; middle: number; lower: number } | null> {
  if (bars.length < BB_PERIOD) {
    return bars.map(() => null);
  }

  const closes = bars.map(b => b.close);
  const bbResult = BollingerBands.calculate({
    period: BB_PERIOD,
    values: closes,
    stdDev: BB_STDDEV,
  });

  // Pad the beginning with nulls (not enough data yet)
  const paddedResult: Array<{ upper: number; middle: number; lower: number } | null> = [];
  for (let i = 0; i < BB_PERIOD - 1; i++) {
    paddedResult.push(null);
  }
  bbResult.forEach(bb => paddedResult.push(bb));

  return paddedResult;
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
  console.log('NQ ICT STRATEGY BACKTEST - BOLLINGER BAND + WICKED CANDLE (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Period: ${START_DATE.toISOString()} -> ${END_DATE.toISOString()}`);
  console.log(`Strategy: Wicked Candle + BB Breach (NO BOS)`);
  console.log(`Bollinger Bands: ${BB_PERIOD} period, ${BB_STDDEV} std dev`);
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

  // Calculate Bollinger Bands
  console.log('Calculating Bollinger Bands...');
  const bb = calculateBollingerBands(bars);

  // Backtest variables
  const trades: TradeRecord[] = [];
  let activePosition: {
    entryTime: string;
    entryPrice: number;
    side: 'long' | 'short';
    stopLoss: number;
    takeProfit: number;
    pattern: string;
  } | null = null;

  let longSignals = 0;
  let shortSignals = 0;

  // Main loop
  console.log('\nRunning backtest...\n');
  for (let i = BB_PERIOD; i < bars.length; i++) {
    const bar = bars[i];
    const currentBB = bb[i];

    if (!currentBB) continue;

    // Check for exit if we have an active position
    if (activePosition) {
      let exitPrice: number | null = null;
      let exitReason: 'stop' | 'target' | null = null;

      if (activePosition.side === 'long') {
        // Check stop loss
        if (bar.low <= activePosition.stopLoss) {
          exitPrice = fillStop(baseSymbol, 'sell', activePosition.stopLoss);
          exitReason = 'stop';
        }
        // Check take profit
        else if (bar.high >= activePosition.takeProfit) {
          exitPrice = fillTP(baseSymbol, 'sell', activePosition.takeProfit);
          exitReason = 'target';
        }
      } else { // short
        // Check stop loss
        if (bar.high >= activePosition.stopLoss) {
          exitPrice = fillStop(baseSymbol, 'buy', activePosition.stopLoss);
          exitReason = 'stop';
        }
        // Check take profit
        else if (bar.low <= activePosition.takeProfit) {
          exitPrice = fillTP(baseSymbol, 'buy', activePosition.takeProfit);
          exitReason = 'target';
        }
      }

      if (exitPrice && exitReason) {
        // Close position
        const pointsGained = activePosition.side === 'long'
          ? (exitPrice - activePosition.entryPrice)
          : (activePosition.entryPrice - exitPrice);
        const ticksGained = pointsGained / TICK_SIZE;
        const grossPnl = ticksGained * TICK_VALUE * NUM_CONTRACTS;
        const fees = addFees(baseSymbol, NUM_CONTRACTS) * 2; // Entry + exit
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
        });

        activePosition = null;
      }
    }

    // Check for new entry if no active position
    if (!activePosition) {
      const wicked = detectWickedCandle(bar);

      if (wicked) {
        let entrySignal: 'long' | 'short' | null = null;

        // Long: Bullish wicked candle with LOW below lower BB
        if (wicked.type === 'bullish' && bar.low < currentBB.lower) {
          entrySignal = 'long';
          longSignals++;
        }
        // Short: Bearish wicked candle with HIGH above upper BB
        else if (wicked.type === 'bearish' && bar.high > currentBB.upper) {
          entrySignal = 'short';
          shortSignals++;
        }

        if (entrySignal) {
          const side = entrySignal;
          const entryPrice = fillEntry(baseSymbol, side === 'long' ? 'buy' : 'sell', bar.close);
          const stopLoss = side === 'long'
            ? roundToTick(entryPrice - (STOP_LOSS_TICKS * TICK_SIZE), TICK_SIZE)
            : roundToTick(entryPrice + (STOP_LOSS_TICKS * TICK_SIZE), TICK_SIZE);
          const takeProfit = side === 'long'
            ? roundToTick(entryPrice + (TAKE_PROFIT_TICKS * TICK_SIZE), TICK_SIZE)
            : roundToTick(entryPrice - (TAKE_PROFIT_TICKS * TICK_SIZE), TICK_SIZE);

          activePosition = {
            entryTime: bar.timestamp,
            entryPrice,
            side,
            stopLoss,
            takeProfit,
            pattern: `${wicked.type} wick + BB breach (${wicked.type === 'bullish' ? 'below lower' : 'above upper'})`,
          };
        }
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
    const grossPnl = ticksGained * TICK_VALUE * NUM_CONTRACTS;
    const fees = addFees(baseSymbol, NUM_CONTRACTS) * 2;
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
      exitReason: 'stop', // Treat end-of-data as stop
      pattern: activePosition.pattern,
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
  const targetExits = trades.filter(t => t.exitReason === 'target').length;

  const dayCount = (END_DATE.getTime() - START_DATE.getTime()) / (1000 * 60 * 60 * 24);
  const tradesPerDay = totalTrades / dayCount;

  // Long/Short breakdown
  const longTrades = trades.filter(t => t.side === 'long');
  const shortTrades = trades.filter(t => t.side === 'short');
  const longWins = longTrades.filter(t => t.pnl > 0).length;
  const shortWins = shortTrades.filter(t => t.pnl > 0).length;
  const longWinRate = longTrades.length > 0 ? (longWins / longTrades.length) * 100 : 0;
  const shortWinRate = shortTrades.length > 0 ? (shortWins / shortTrades.length) * 100 : 0;

  // Print results
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST RESULTS');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${totalTrades} | Wins: ${wins.length} | Losses: ${losses.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net PnL: $${totalPnL.toFixed(2)} | Gross PnL: $${totalGrossPnL.toFixed(2)} | Fees: $${totalFees.toFixed(2)}`);
  console.log(`Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: $${maxDrawdown.toFixed(2)}`);
  console.log(`Average Win: $${avgWin.toFixed(2)} | Average Loss: $${avgLoss.toFixed(2)}`);
  console.log(`Trades Per Day: ${tradesPerDay.toFixed(2)}`);
  console.log(`Exit Reasons: Stop=${stopExits} (${((stopExits / totalTrades) * 100).toFixed(1)}%), Target=${targetExits} (${((targetExits / totalTrades) * 100).toFixed(1)}%)`);
  console.log('\n--- Long/Short Breakdown ---');
  console.log(`Long Trades: ${longTrades.length} (${longWins} wins, ${longWinRate.toFixed(1)}% WR) | Signals: ${longSignals}`);
  console.log(`Short Trades: ${shortTrades.length} (${shortWins} wins, ${shortWinRate.toFixed(1)}% WR) | Signals: ${shortSignals}`);
  console.log('='.repeat(80));

  // Comparison to current strategy
  console.log('\n' + '='.repeat(80));
  console.log('COMPARISON TO CURRENT STRATEGY (4/16/32)');
  console.log('='.repeat(80));
  console.log('Current Strategy (Wicked + BOS, 4 SL/16 TP1/32 TP2):');
  console.log('  - Win Rate: 70.9%');
  console.log('  - Net PnL: +$53,609 (14 days)');
  console.log('  - Profit Factor: 5.71');
  console.log('  - Trades: 536 (~19/day)');
  console.log('\nNew Strategy (Wicked + BB, 8 SL/33 TP):');
  console.log(`  - Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`  - Net PnL: $${totalPnL.toFixed(2)} (14 days)`);
  console.log(`  - Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`  - Trades: ${totalTrades} (~${tradesPerDay.toFixed(1)}/day)`);
  console.log('='.repeat(80));
}

runBacktest().catch(console.error);
