#!/usr/bin/env tsx
/**
 * ICT/SMC/PO3 Strategy - High Frequency
 * 1-Minute Timeframe
 *
 * Entry Logic (ICT/SMC/PO3):
 * - LONG: BOS (Break of Structure) above swing high + FVG filled + PO3 candle confirmation
 * - SHORT: BOS below swing low + FVG filled + PO3 candle confirmation
 *
 * PO3 (Point of Intrusion): Entry on first candle that intrudes into FVG
 * BOS (Break of Structure): Break above last swing high/low
 * FVG (Fair Value Gap): Unfilled gap between candles
 *
 * Exit Strategy:
 * - Stop Loss: 8 ticks below/above entry
 * - Take Profit: 16 ticks above/below entry
 */

import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'stop' | 'target' | 'end_of_data';
  structureQuality: string;
  candlePattern: string;
}

const DEFAULT_SYMBOL = process.env.TOPSTEPX_MR_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.TOPSTEPX_MR_CONTRACT_ID;
const STOP_LOSS_TICKS = parseInt(process.env.TOPSTEPX_MR_STOP_LOSS_TICKS || '8');
const TAKE_PROFIT_TICKS = parseInt(process.env.TOPSTEPX_MR_TAKE_PROFIT_TICKS || '36');
const TAKE_PROFIT_1_TICKS = parseInt(process.env.TOPSTEPX_MR_TAKE_PROFIT_1_TICKS || '0'); // First target for scaling (0 = disabled)
const TAKE_PROFIT_2_TICKS = parseInt(process.env.TOPSTEPX_MR_TAKE_PROFIT_2_TICKS || '0'); // Second target for scaling (0 = use TAKE_PROFIT_TICKS)
const SCALED_EXITS = TAKE_PROFIT_1_TICKS > 0; // Enable scaled exits if TP1 is set
const HIGH_ACCURACY_MODE = process.env.TOPSTEPX_MR_HIGH_ACCURACY === 'true'; // Wicked candles (regular)
const STRICT_WICKED_MODE = process.env.TOPSTEPX_MR_STRICT_WICKED === 'true'; // Stricter wicked candles
const ULTRA_ACCURACY_MODE = process.env.TOPSTEPX_MR_ULTRA_ACCURACY === 'true'; // Strict wicked + BOS confluence
const HTF_CONFIRMATION = process.env.TOPSTEPX_MR_HTF_CONFIRM === 'true'; // 5-min BOS confirmation

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

const SLIP_CONFIG: SlippageConfig = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'slip-config.json'), 'utf-8')
);

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;

function getBaseSymbol(symbol: string): string {
  return symbol.replace(/Z\d+$/, '').replace(/Z5$/, '');
}

function isTradingAllowed(timestamp: string): boolean {
  const date = new Date(timestamp);
  const ctDate = new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 0 || day === 6) return false;
  return minutes >= 510 && minutes < 900; // RTH: 8:30 AM - 3:00 PM CT
}

async function runBacktest() {
  const metadata = await fetchTopstepXFuturesMetadata(
    DEFAULT_CONTRACT_ID || DEFAULT_SYMBOL
  );

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${DEFAULT_SYMBOL}`);
  }

  const contractId = metadata.id;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 5;
  const tickSize = metadata.tickSize || 0.25;

  const baseSymbol = getBaseSymbol(DEFAULT_SYMBOL);

  console.log(`\n${'='.repeat(80)}`);
  console.log('MARKET STRUCTURE + CANDLESTICK PATTERN STRATEGY');
  console.log(`${'='.repeat(80)}`);
  console.log(`Symbol: ${DEFAULT_SYMBOL} (${metadata.name})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize}`);
  console.log(`Stop Loss: 8 ticks | Take Profit: 16 ticks`);
  console.log(`${'='.repeat(80)}\n`);

  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: '2025-11-01',
    endTime: '2025-11-14',
    unit: 2,
    unitNumber: 1,
    limit: 100000,
  });

  if (!bars.length) {
    throw new Error('No 1-minute bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-minute bars\n`);

  // Load 5-minute bars for HTF confirmation
  let bars5min: TopstepXFuturesBar[] = [];
  if (HTF_CONFIRMATION) {
    bars5min = await fetchTopstepXFuturesBars({
      contractId,
      startTime: '2025-11-01',
      endTime: '2025-11-14',
      unit: 2,
      unitNumber: 5,
      limit: 100000,
    });
    bars5min.reverse();
    console.log(`Loaded ${bars5min.length} five-minute bars for HTF confirmation\n`);
  }

  const trades: TradeRecord[] = [];
  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stopLoss: number;
    target: number;
    target2?: number; // Second target for scaled exits
    structureQuality: string;
    candlePattern: string;
    scaledHitTP1?: boolean; // Track if TP1 was already hit
  } | null = null;

  let realizedPnL = 0;
  let peakBalance = 0;
  let maxDrawdown = 0;
  const commissionPerSide = inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 0.35);

  // Helper to round to tick
  const roundToTick = (price: number): number => {
    return Math.round(price / tickSize) * tickSize;
  };

  // Helper to apply entry slippage
  const fillEntry = (side: 'buy' | 'sell', price: number): number => {
    const slipTicks = SLIP_CONFIG.slipAvg.entry[baseSymbol] || 0.5;
    const slipPoints = slipTicks * tickSize;
    return side === 'buy' ? price + slipPoints : price - slipPoints;
  };

  // Helper to apply TP slippage
  const fillTP = (side: 'buy' | 'sell', price: number): number => {
    const slipTicks = SLIP_CONFIG.slipAvg.tp[baseSymbol] || 0.3;
    const slipPoints = slipTicks * tickSize;
    return side === 'buy' ? price - slipPoints : price + slipPoints;
  };

  // Helper to apply stop slippage
  const fillStop = (side: 'buy' | 'sell', price: number): number => {
    const slipTicks = SLIP_CONFIG.slipAvg.stop[baseSymbol] || 1;
    const slipPoints = slipTicks * tickSize;
    return side === 'buy' ? price + slipPoints : price - slipPoints;
  };

  // ICT/SMC: Detect Fair Value Gap (FVG) - gap between candles
  const detectFVG = (bars: TopstepXFuturesBar[], index: number): { type: 'bullish' | 'bearish'; high: number; low: number } | null => {
    if (index < 1) return null;

    const prev = bars[index - 1];
    const curr = bars[index];

    // Bullish FVG: previous low > current high (gap up)
    if (prev.low > curr.high) {
      return {
        type: 'bullish',
        low: curr.high,
        high: prev.low,
      };
    }

    // Bearish FVG: previous high < current low (gap down)
    if (prev.high < curr.low) {
      return {
        type: 'bearish',
        low: prev.high,
        high: curr.low,
      };
    }

    return null;
  };

  // ICT/SMC: Detect Break of Structure (BOS)
  const detectBOS = (bars: TopstepXFuturesBar[], index: number, lookback: number = 3): { type: 'bullish' | 'bearish'; level: number } | null => {
    if (index < lookback) return null;

    // Find swing high and low in lookback window
    let swingHigh = bars[index - lookback].high;
    let swingLow = bars[index - lookback].low;

    for (let i = index - lookback; i < index; i++) {
      swingHigh = Math.max(swingHigh, bars[i].high);
      swingLow = Math.min(swingLow, bars[i].low);
    }

    const curr = bars[index];

    // Bullish BOS: Close above swing high
    if (curr.close > swingHigh && curr.close > curr.open) {
      return { type: 'bullish', level: swingHigh };
    }

    // Bearish BOS: Close below swing low
    if (curr.close < swingLow && curr.close < curr.open) {
      return { type: 'bearish', level: swingLow };
    }

    return null;
  };

  // HTF (Higher TimeFrame) Confirmation: Check if 5-min is in agreement
  // IMPORTANT: Only uses CLOSED 5-min bars to avoid intra-bar look-ahead bias
  const validateHTFBOS = (entryTime: string, direction: 'long' | 'short'): boolean => {
    if (!HTF_CONFIRMATION || bars5min.length === 0) return true; // No HTF validation if disabled

    // Find the 5-min bar STRICTLY BEFORE entry time (not equal - must be closed/completed)
    // This ensures we only use finalized 5-min candle data
    let htfBarIndex = -1;
    for (let i = bars5min.length - 1; i >= 0; i--) {
      if (bars5min[i].timestamp < entryTime) {
        htfBarIndex = i;
        break;
      }
    }

    if (htfBarIndex < 3) return false; // Not enough data for HTF validation

    // Use the PREVIOUS closed 5-min bar for BOS analysis (ensures no intra-bar bias)
    const htfBar = bars5min[htfBarIndex];
    const htfBOS = detectBOS(bars5min, htfBarIndex, 5); // Longer lookback for 5-min

    // For LONG: Need 5-min bullish BOS
    if (direction === 'long') {
      return htfBOS && htfBOS.type === 'bullish';
    }

    // For SHORT: Need 5-min bearish BOS
    if (direction === 'short') {
      return htfBOS && htfBOS.type === 'bearish';
    }

    return false;
  };

  // ICT Wicked Candle: Strong rejection (high probability reversal)
  // Definition: Candle with a long wick in one direction, closing in opposite direction
  // Bullish Wick: High wick >70% of range, closes in top 30% = institutional rejection of highs
  // Bearish Wick: Low wick >70% of range, closes in bottom 30% = institutional rejection of lows
  const detectWickedCandle = (bars: TopstepXFuturesBar[], index: number, strictMode: boolean = false): { type: 'bullish' | 'bearish'; strength: number } | null => {
    const curr = bars[index];
    const range = curr.high - curr.low;

    if (range < 0.01) return null; // Ignore very small candles

    const closePercent = (curr.close - curr.low) / range;
    const topWickPercent = (curr.high - Math.max(curr.open, curr.close)) / range;
    const bottomWickPercent = (Math.min(curr.open, curr.close) - curr.low) / range;

    // STRICT mode: Moderately stricter rejection (>70% wick, tighter close)
    if (strictMode) {
      // Bullish: Bottom wick >70%, close in top 50%
      if (topWickPercent < 0.2 && bottomWickPercent > 0.7 && closePercent > 0.65) {
        const strength = bottomWickPercent;
        return { type: 'bullish', strength };
      }
      // Bearish: Top wick >70%, close in bottom 50%
      if (bottomWickPercent < 0.2 && topWickPercent > 0.7 && closePercent < 0.35) {
        const strength = topWickPercent;
        return { type: 'bearish', strength };
      }
    } else {
      // Regular mode: Balanced requirements
      // Bullish rejection wick: Price tried to go low, rejected, closed high
      // Top wick <20%, bottom wick >60%, close in top 40%
      if (topWickPercent < 0.2 && bottomWickPercent > 0.6 && closePercent > 0.6) {
        return { type: 'bullish', strength: bottomWickPercent };
      }

      // Bearish rejection wick: Price tried to go high, rejected, closed low
      // Bottom wick <20%, top wick >60%, close in bottom 40%
      if (bottomWickPercent < 0.2 && topWickPercent > 0.6 && closePercent < 0.4) {
        return { type: 'bearish', strength: topWickPercent };
      }
    }

    return null;
  };

  // PO3 (Point of Intrusion): Entry candle intrudes into FVG
  interface TradeSetup {
    type: 'long' | 'short';
    entryBar: TopstepXFuturesBar;
    fvg: { type: 'bullish' | 'bearish'; high: number; low: number };
    bos: { type: 'bullish' | 'bearish'; level: number };
    liquiditySweep?: boolean;
  }

  const findTradeSetup = (bars: TopstepXFuturesBar[], index: number): TradeSetup | null => {
    if (index < 3) return null;

    const fvg = detectFVG(bars, index);
    const bos = detectBOS(bars, index, 3);
    const wickedCandle = detectWickedCandle(bars, index, false); // Normal mode
    const strictWickedCandle = detectWickedCandle(bars, index, true); // Strict mode
    const curr = bars[index];
    const prev = bars[index - 1];

    // ULTRA ACCURACY MODE: Strict Wicked Candle + BOS confluence (strongest signals only)
    if (ULTRA_ACCURACY_MODE) {
      if (strictWickedCandle && bos && strictWickedCandle.type === bos.type) {
        if (strictWickedCandle.type === 'bullish') {
          return {
            type: 'long',
            entryBar: curr,
            fvg: { type: 'bullish', high: curr.high, low: curr.low },
            bos,
            liquiditySweep: true
          };
        }
        if (strictWickedCandle.type === 'bearish') {
          return {
            type: 'short',
            entryBar: curr,
            fvg: { type: 'bearish', high: curr.high, low: curr.low },
            bos,
            liquiditySweep: true
          };
        }
      }
      return null;
    }

    // STRICT WICKED MODE: Stricter wicked candles only (no BOS requirement)
    if (STRICT_WICKED_MODE) {
      if (strictWickedCandle) {
        if (strictWickedCandle.type === 'bullish') {
          return {
            type: 'long',
            entryBar: curr,
            fvg: { type: 'bullish', high: curr.high, low: curr.low },
            bos: bos || { type: 'bullish', level: curr.low },
            liquiditySweep: true
          };
        }
        if (strictWickedCandle.type === 'bearish') {
          return {
            type: 'short',
            entryBar: curr,
            fvg: { type: 'bearish', high: curr.high, low: curr.low },
            bos: bos || { type: 'bearish', level: curr.high },
            liquiditySweep: true
          };
        }
      }
      return null;
    }

    // HIGH ACCURACY MODE: Regular Wicked Candles (institutional rejection candles)
    if (HIGH_ACCURACY_MODE) {
      if (wickedCandle) {
        if (wickedCandle.type === 'bullish') {
          return {
            type: 'long',
            entryBar: curr,
            fvg: { type: 'bullish', high: curr.high, low: curr.low },
            bos: bos || { type: 'bullish', level: curr.low },
            liquiditySweep: true  // Marks as high accuracy signal
          };
        }
        if (wickedCandle.type === 'bearish') {
          return {
            type: 'short',
            entryBar: curr,
            fvg: { type: 'bearish', high: curr.high, low: curr.low },
            bos: bos || { type: 'bearish', level: curr.high },
            liquiditySweep: true
          };
        }
      }
      return null; // Skip all other signals in high accuracy mode
    }

    // NORMAL MODE: All 3 accuracy tiers
    // HIGHEST ACCURACY: BOS + Wicked Candle detected
    if (bos && wickedCandle && bos.type === wickedCandle.type) {
      if (wickedCandle.type === 'bullish' && curr.close > curr.open) {
        return {
          type: 'long',
          entryBar: curr,
          fvg: fvg || { type: 'bullish', high: curr.high, low: curr.low },
          bos,
          liquiditySweep: true
        };
      }
      if (wickedCandle.type === 'bearish' && curr.close < curr.open) {
        return {
          type: 'short',
          entryBar: curr,
          fvg: fvg || { type: 'bearish', high: curr.high, low: curr.low },
          bos,
          liquiditySweep: true
        };
      }
    }

    // MEDIUM ACCURACY: BOS + FVG (no sweep required, but confirmed by FVG)
    if (bos && fvg && bos.type === fvg.type) {
      if (bos.type === 'bullish' && curr.close > curr.open && curr.close > prev.close) {
        return {
          type: 'long',
          entryBar: curr,
          fvg,
          bos,
          liquiditySweep: false
        };
      }
      if (bos.type === 'bearish' && curr.close < curr.open && curr.close < prev.close) {
        return {
          type: 'short',
          entryBar: curr,
          fvg,
          bos,
          liquiditySweep: false
        };
      }
    }

    // LOWER ACCURACY: Just BOS (high frequency, lower win rate)
    if (bos && !fvg && !wickedCandle) {
      if (bos.type === 'bullish' && curr.close > curr.open && curr.close > prev.close) {
        return {
          type: 'long',
          entryBar: curr,
          fvg: { type: 'bullish', high: curr.high, low: curr.low },
          bos,
          liquiditySweep: false
        };
      }
      if (bos.type === 'bearish' && curr.close < curr.open && curr.close < prev.close) {
        return {
          type: 'short',
          entryBar: curr,
          fvg: { type: 'bearish', high: curr.high, low: curr.low },
          bos,
          liquiditySweep: false
        };
      }
    }

    return null;
  };

  // Detect Market Structure: Higher High + Higher Low = Bullish, Lower High + Lower Low = Bearish
  const detectStructure = (bars: TopstepXFuturesBar[], index: number): string | null => {
    if (index < 2) return null;

    const prev2 = bars[index - 2];
    const prev1 = bars[index - 1];
    const current = bars[index];

    const bullish =
      current.high > prev1.high &&
      current.low > prev1.low &&
      prev1.high > prev2.high &&
      prev1.low > prev2.low;

    const bearish =
      current.high < prev1.high &&
      current.low < prev1.low &&
      prev1.high < prev2.high &&
      prev1.low < prev2.low;

    if (bullish) return 'HH/HL';
    if (bearish) return 'LH/LL';
    return null;
  };

  // Detect Candlestick Pattern Quality
  const detectCandlePattern = (bar: TopstepXFuturesBar, side: 'long' | 'short'): string => {
    const range = bar.high - bar.low;
    if (range === 0) return 'doji';

    const closePercent = (bar.close - bar.low) / range;

    if (side === 'long') {
      // Bullish: close in top 50%, body in top 50%
      if (closePercent > 0.5) return 'bullish';
      return 'weak-bullish';
    } else {
      // Bearish: close in bottom 50%, body in bottom 50%
      if (closePercent < 0.5) return 'bearish';
      return 'weak-bearish';
    }
  };

  // Exit position function
  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: 'stop' | 'target' | 'end_of_data'
  ) => {
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * 3; // 3 contracts
    const exitFees = commissionPerSide * 2; // entry + exit
    const netPnl = rawPnL - exitFees;

    const trade: TradeRecord = {
      entryTime: position.entryTime,
      exitTime,
      side: position.side,
      entryPrice: position.entryPrice,
      exitPrice,
      pnl: netPnl,
      grossPnl: rawPnL,
      fees: exitFees,
      exitReason: reason,
      structureQuality: position.structureQuality,
      candlePattern: position.candlePattern,
    };

    trades.push(trade);
    realizedPnL += netPnl;

    // Track drawdown
    if (realizedPnL > peakBalance) {
      peakBalance = realizedPnL;
    }
    const currentDrawdown = peakBalance - realizedPnL;
    if (currentDrawdown > maxDrawdown) {
      maxDrawdown = currentDrawdown;
    }

    position = null;
  };

  // Main backtest loop
  for (let i = 2; i < bars.length; i++) {
    const bar = bars[i];

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(closeSide, bar.close));
        exitPosition(sessionExitPrice, bar.timestamp, 'end_of_data');
      }
      continue;
    }

    // Check if position should exit
    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // For scaled exits: Check TP1 first, then TP2
      if (SCALED_EXITS && position.target2) {
        // Check if TP1 hit (first scale)
        if (!position.scaledHitTP1) {
          if ((direction === 1 && bar.high >= position.target) ||
              (direction === -1 && bar.low <= position.target)) {
            const closeSide = position.side === 'long' ? 'sell' : 'buy';
            const basePrice = direction === 1
              ? Math.min(bar.high, position.target)
              : Math.max(bar.low, position.target);
            const exitPrice = roundToTick(fillTP(closeSide, basePrice));
            console.log(`[${bar.timestamp}] TP1 hit ${DEFAULT_SYMBOL} ${position.side.toUpperCase()}: Exit 50% (1.5 contracts) @ ${exitPrice.toFixed(2)}`);
            // Record partial exit (50% of position)
            const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * 1.5; // 1.5 contracts
            const exitFees = commissionPerSide * 2; // entry + exit
            const netPnl = rawPnL - exitFees;
            const trade: TradeRecord = {
              entryTime: position.entryTime,
              exitTime: bar.timestamp,
              side: position.side,
              entryPrice: position.entryPrice,
              exitPrice,
              pnl: netPnl,
              grossPnl: rawPnL,
              fees: exitFees,
              exitReason: 'target',
              structureQuality: position.structureQuality,
              candlePattern: position.candlePattern,
            };
            trades.push(trade);
            realizedPnL += netPnl;
            if (realizedPnL > peakBalance) peakBalance = realizedPnL;
            const currentDrawdown = peakBalance - realizedPnL;
            if (currentDrawdown > maxDrawdown) maxDrawdown = currentDrawdown;

            // Mark TP1 as hit, continue waiting for TP2
            position.scaledHitTP1 = true;
            continue;
          }
        }

        // Check if TP2 hit (second scale) - if we got here, we've already hit TP1
        if (position.scaledHitTP1) {
          if ((direction === 1 && bar.high >= position.target2) ||
              (direction === -1 && bar.low <= position.target2)) {
            const closeSide = position.side === 'long' ? 'sell' : 'buy';
            const basePrice = direction === 1
              ? Math.min(bar.high, position.target2)
              : Math.max(bar.low, position.target2);
            const exitPrice = roundToTick(fillTP(closeSide, basePrice));
            console.log(`[${bar.timestamp}] TP2 hit ${DEFAULT_SYMBOL} ${position.side.toUpperCase()}: Exit remaining 50% (1.5 contracts) @ ${exitPrice.toFixed(2)}`);
            // Record final exit (remaining 50%)
            const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * 1.5; // 1.5 contracts
            const exitFees = commissionPerSide * 2; // entry + exit
            const netPnl = rawPnL - exitFees;
            const trade: TradeRecord = {
              entryTime: position.entryTime,
              exitTime: bar.timestamp,
              side: position.side,
              entryPrice: position.entryPrice,
              exitPrice,
              pnl: netPnl,
              grossPnl: rawPnL,
              fees: exitFees,
              exitReason: 'target',
              structureQuality: position.structureQuality,
              candlePattern: position.candlePattern,
            };
            trades.push(trade);
            realizedPnL += netPnl;
            if (realizedPnL > peakBalance) peakBalance = realizedPnL;
            const currentDrawdown = peakBalance - realizedPnL;
            if (currentDrawdown > maxDrawdown) maxDrawdown = currentDrawdown;

            position = null;
            continue;
          }
        }
      } else {
        // Standard single TP exit
        if ((direction === 1 && bar.high >= position.target) ||
            (direction === -1 && bar.low <= position.target)) {
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const basePrice = direction === 1
            ? Math.min(bar.high, position.target)
            : Math.max(bar.low, position.target);
          const exitPrice = roundToTick(fillTP(closeSide, basePrice));
          console.log(`[${bar.timestamp}] TARGET hit ${DEFAULT_SYMBOL} ${position.side.toUpperCase()}: Exit all 3 @ ${exitPrice.toFixed(2)}`);
          exitPosition(exitPrice, bar.timestamp, 'target');
          continue;
        }
      }

      // Stop hit
      if ((direction === 1 && bar.low <= position.stopLoss) ||
          (direction === -1 && bar.high >= position.stopLoss)) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const exitPrice = roundToTick(fillStop(closeSide, position.stopLoss));
        console.log(`[${bar.timestamp}] STOP hit ${DEFAULT_SYMBOL} ${position.side.toUpperCase()}: Exit all 3 @ ${exitPrice.toFixed(2)}`);
        exitPosition(exitPrice, bar.timestamp, 'stop');
        continue;
      }
    }

    // Check for new entries using ICT/SMC/PO3
    if (!position) {
      const setup = findTradeSetup(bars, i);

      if (setup && setup.type === 'long') {
        // HTF validation: Only take long if 5-min confirms
        const htfValid = validateHTFBOS(bar.timestamp, 'long');
        if (HTF_CONFIRMATION && !htfValid) {
          // Skip this signal - 5-min doesn't confirm
          continue;
        }

        const entryPrice = roundToTick(fillEntry('buy', bar.close));
        const stopLoss = roundToTick(entryPrice - (STOP_LOSS_TICKS * tickSize));
        const target = roundToTick(entryPrice + (TAKE_PROFIT_1_TICKS > 0 ? TAKE_PROFIT_1_TICKS : TAKE_PROFIT_TICKS) * tickSize);
        const target2 = TAKE_PROFIT_1_TICKS > 0
          ? roundToTick(entryPrice + ((TAKE_PROFIT_2_TICKS > 0 ? TAKE_PROFIT_2_TICKS : TAKE_PROFIT_TICKS) * tickSize))
          : undefined;

        position = {
          side: 'long',
          entryPrice,
          entryTime: bar.timestamp,
          stopLoss,
          target,
          target2,
          structureQuality: 'BOS/FVG/PO3',
          candlePattern: 'bullish-intrusion',
        };

        if (SCALED_EXITS && target2) {
          console.log(`[${bar.timestamp}] LONG entry ${DEFAULT_SYMBOL} @ ${entryPrice.toFixed(2)} (SL=${STOP_LOSS_TICKS}t TP1=${TAKE_PROFIT_1_TICKS}t TP2=${TAKE_PROFIT_2_TICKS}t) | Stop: ${stopLoss.toFixed(2)}, TP1: ${target.toFixed(2)}, TP2: ${target2.toFixed(2)}`);
        } else {
          console.log(`[${bar.timestamp}] LONG entry ${DEFAULT_SYMBOL} @ ${entryPrice.toFixed(2)} (BOS/FVG/PO3, SL=${STOP_LOSS_TICKS}t TP=${TAKE_PROFIT_TICKS}t) | Stop: ${stopLoss.toFixed(2)}, Target: ${target.toFixed(2)}`);
        }
      } else if (setup && setup.type === 'short') {
        // HTF validation: Only take short if 5-min confirms
        const htfValid = validateHTFBOS(bar.timestamp, 'short');
        if (HTF_CONFIRMATION && !htfValid) {
          // Skip this signal - 5-min doesn't confirm
          continue;
        }

        const entryPrice = roundToTick(fillEntry('sell', bar.close));
        const stopLoss = roundToTick(entryPrice + (STOP_LOSS_TICKS * tickSize));
        const target = roundToTick(entryPrice - (TAKE_PROFIT_1_TICKS > 0 ? TAKE_PROFIT_1_TICKS : TAKE_PROFIT_TICKS) * tickSize);
        const target2 = TAKE_PROFIT_1_TICKS > 0
          ? roundToTick(entryPrice - ((TAKE_PROFIT_2_TICKS > 0 ? TAKE_PROFIT_2_TICKS : TAKE_PROFIT_TICKS) * tickSize))
          : undefined;

        position = {
          side: 'short',
          entryPrice,
          entryTime: bar.timestamp,
          stopLoss,
          target,
          target2,
          structureQuality: 'BOS/FVG/PO3',
          candlePattern: 'bearish-intrusion',
        };

        if (SCALED_EXITS && target2) {
          console.log(`[${bar.timestamp}] SHORT entry ${DEFAULT_SYMBOL} @ ${entryPrice.toFixed(2)} (SL=${STOP_LOSS_TICKS}t TP1=${TAKE_PROFIT_1_TICKS}t TP2=${TAKE_PROFIT_2_TICKS}t) | Stop: ${stopLoss.toFixed(2)}, TP1: ${target.toFixed(2)}, TP2: ${target2.toFixed(2)}`);
        } else {
          console.log(`[${bar.timestamp}] SHORT entry ${DEFAULT_SYMBOL} @ ${entryPrice.toFixed(2)} (BOS/FVG/PO3) | Stop: ${stopLoss.toFixed(2)}, Target: ${target.toFixed(2)}`);
        }
      }
    }
  }

  // Close any remaining position
  if (position) {
    const lastBar = bars[bars.length - 1];
    const closeSide = position.side === 'long' ? 'sell' : 'buy';
    const endExitPrice = roundToTick(fillStop(closeSide, lastBar.close));
    exitPosition(endExitPrice, lastBar.timestamp, 'end_of_data');
  }

  // Print results
  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST SUMMARY');
  console.log('='.repeat(80));

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);

  const avgWin = winningTrades.length
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length
    ? losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length
    : 0;

  const totalGross = trades.reduce((sum, t) => sum + t.grossPnl, 0);
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);
  const winRate = trades.length ? (winningTrades.length / trades.length) * 100 : 0;
  const profitFactor = losingTrades.length && totalGross > 0
    ? Math.abs(winningTrades.reduce((sum, t) => sum + t.grossPnl, 0) / losingTrades.reduce((sum, t) => sum + t.grossPnl, 0))
    : 0;

  console.log(`Total Trades: ${trades.length} | Wins: ${winningTrades.length} | Losses: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net Realized PnL: ${realizedPnL > 0 ? '+' : ''}${realizedPnL.toFixed(2)} USD | Fees Paid: $${totalFees.toFixed(2)}`);
  console.log(`Avg Win: ${avgWin > 0 ? '+' : ''}${avgWin.toFixed(2)} | Avg Loss: ${avgLoss.toFixed(2)}`);
  console.log(`Profit Factor: ${profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: -$${maxDrawdown.toFixed(2)}`);
  const drawdownPercent = peakBalance > 0 ? (maxDrawdown / peakBalance) * 100 : 0;
  console.log(`Max Drawdown %: ${drawdownPercent.toFixed(2)}%`);

  if (trades.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RECENT TRADES');
    console.log('='.repeat(80));
    trades.slice(-10).forEach(trade => {
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${trade.pnl > 0 ? '+' : ''}${trade.pnl.toFixed(2)} ` +
        `(${trade.exitReason}) [${trade.structureQuality}/${trade.candlePattern}]`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('Backtest failed:', err);
  process.exit(1);
});
