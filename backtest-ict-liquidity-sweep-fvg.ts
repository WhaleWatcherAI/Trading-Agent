#!/usr/bin/env tsx
/**
 * ICT Strategy #1: Liquidity-Sweep + FVG Return Backtest
 *
 * CORE CONCEPT (ICT 2022):
 * Wait for price to raid obvious resting liquidity (prior day's H/L),
 * then if it immediately prints an imbalance (FVG), fade back into the
 * range at the FVG 50% ("mean threshold").
 *
 * Session filter (NY): trade only 09:30–11:30 and 13:30–15:30 ET
 * HTF anchors: previous day high/low
 *
 * Entry Logic:
 * 1. Detect liquidity sweep:
 *    - Sell-side sweep (for long): today's low breaks yesterday's low,
 *      then we close back above yesterday's low within N bars (≤5 bars)
 *    - Buy-side sweep (for short): today's high breaks yesterday's high,
 *      then we close back below it within N bars
 *
 * 2. Detect FVG (3-bar):
 *    - Bullish FVG on bar t if low[t] > high[t-2]
 *    - Midpoint (mean threshold) = (low[t] + high[t-2]) / 2
 *    - Bearish FVG on bar t if high[t] < low[t-2]
 *    - Midpoint = (high[t] + low[t-2]) / 2
 *
 * 3. Entry:
 *    - Long: After sell-side sweep, wait for first bullish FVG.
 *      Limit entry at its 50% midpoint during NY session.
 *    - Short: Mirror with buy-side sweep → first bearish FVG → 50% midpoint
 *
 * Risk/Targets:
 * - Stop = just beyond the sweep extreme (1-2 ticks outside swept low/high)
 * - TP1 = 1R; TP2 = 2R (or previous day's open/mid)
 * - Move stop to breakeven at TP1
 */

import { RSI } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  reclaimBars: number;           // Max bars to reclaim after sweep
  stopLossBuffer: number;         // Ticks beyond sweep extreme for stop
  tp1RMultiple: number;           // First target = 1R
  tp2RMultiple: number;           // Second target = 2R
  numberOfContracts: number;
  scaleOutPercent: number;        // % to exit at TP1 (e.g., 0.5 = 50%)
  commissionPerSide: number;
  contractMultiplier?: number;
  useTrendFilter: boolean;        // Use 20-EMA trend filter
  useTimeFilter: boolean;         // Use tighter time filter (09:30-11:00)
  minFVGSizeATR: number;          // Minimum FVG size as % of ATR (e.g., 0.25)
}

interface TradeRecord {
  entryTime: string;
  exitTime: string;
  side: 'long' | 'short';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  grossPnl: number;
  fees: number;
  exitReason: 'tp1' | 'tp2' | 'stop' | 'session' | 'end_of_data';
  sweepPrice: number;
  fvgMidpoint: number;
  slippageCost?: number;
}

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

const getBaseSymbol = (fullSymbol: string): string => {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
};

const DEFAULT_DAYS = 90;
const DEFAULT_SYMBOL = process.env.ICT_SWEEP_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.ICT_SWEEP_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.ICT_SWEEP_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.ICT_SWEEP_END || new Date().toISOString(),
  reclaimBars: Number(process.env.ICT_SWEEP_RECLAIM_BARS || '5'),
  stopLossBuffer: Number(process.env.ICT_SWEEP_SL_BUFFER || '2'),
  tp1RMultiple: Number(process.env.ICT_SWEEP_TP1 || '1'),
  tp2RMultiple: Number(process.env.ICT_SWEEP_TP2 || '2'),
  numberOfContracts: Number(process.env.ICT_SWEEP_CONTRACTS || '2'),
  scaleOutPercent: Number(process.env.ICT_SWEEP_SCALE_PERCENT || '0.5'),
  commissionPerSide: process.env.ICT_SWEEP_COMMISSION
    ? Number(process.env.ICT_SWEEP_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 1.40),
  useTrendFilter: process.env.ICT_SWEEP_USE_TREND !== '0', // Default true, set to 0 to disable
  useTimeFilter: process.env.ICT_SWEEP_USE_TIME !== '0',   // Default true, set to 0 to disable
  minFVGSizeATR: Number(process.env.ICT_SWEEP_MIN_FVG_ATR || '0.25'), // 25% of ATR minimum
};

// Calculate ATR (Average True Range)
function calculateATR(bars: TopstepXFuturesBar[], period: number = 14): number[] {
  const atr: number[] = [];
  if (bars.length < period) return atr;

  const trueRanges: number[] = [];

  for (let i = 0; i < bars.length; i++) {
    if (i === 0) {
      trueRanges.push(bars[i].high - bars[i].low);
    } else {
      const tr = Math.max(
        bars[i].high - bars[i].low,
        Math.abs(bars[i].high - bars[i - 1].close),
        Math.abs(bars[i].low - bars[i - 1].close)
      );
      trueRanges.push(tr);
    }
  }

  // Initial ATR (simple average)
  let currentATR = trueRanges.slice(0, period).reduce((a, b) => a + b, 0) / period;
  atr.push(currentATR);

  // Smoothed ATR
  for (let i = period; i < trueRanges.length; i++) {
    currentATR = ((currentATR * (period - 1)) + trueRanges[i]) / period;
    atr.push(currentATR);
  }

  return atr;
}

// Session detection
type SessionType = 'asia' | 'london' | 'ny' | 'other';

function getSessionType(timestamp: string): SessionType {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();
  const minutes = etDate.getUTCMinutes();
  const totalMinutes = hours * 60 + minutes;

  // Asia: 20:00-00:00 ET (1200-1440 minutes, wraps midnight)
  if (totalMinutes >= 1200 || totalMinutes < 0) {
    return 'asia';
  }

  // London: 02:00-05:00 ET (120-300 minutes)
  if (totalMinutes >= 120 && totalMinutes <= 300) {
    return 'london';
  }

  // NY: 09:30-11:30 and 13:30-15:30 ET
  const nySession1 = totalMinutes >= 570 && totalMinutes <= 690;
  const nySession2 = totalMinutes >= 810 && totalMinutes <= 930;
  if (nySession1 || nySession2) {
    return 'ny';
  }

  return 'other';
}

// Calculate EMA (Exponential Moving Average)
function calculateEMA(bars: TopstepXFuturesBar[], period: number = 20): number[] {
  const ema: number[] = [];
  if (bars.length < period) return ema;

  // Initial SMA
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += bars[i].close;
  }
  ema.push(sum / period);

  // Calculate multiplier
  const multiplier = 2 / (period + 1);

  // Calculate EMA for remaining bars
  for (let i = period; i < bars.length; i++) {
    const currentEMA = (bars[i].close - ema[ema.length - 1]) * multiplier + ema[ema.length - 1];
    ema.push(currentEMA);
  }

  return ema;
}

// NY session times in ET (09:30-11:30, 13:30-15:30)
// Or tighter window (09:30-11:00) if useTimeFilter is enabled
function isNYSession(timestamp: string, useTightWindow: boolean = false): boolean {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();
  const minutes = etDate.getUTCMinutes();
  const totalMinutes = hours * 60 + minutes;

  if (useTightWindow) {
    // Tighter window: 09:30-11:00 ET only (570-660 minutes)
    return totalMinutes >= 570 && totalMinutes <= 660;
  } else {
    // Full NY session: 09:30-11:30 and 13:30-15:30 ET
    const nySession1 = totalMinutes >= 570 && totalMinutes <= 690;
    const nySession2 = totalMinutes >= 810 && totalMinutes <= 930;
    return nySession1 || nySession2;
  }
}

interface FVG {
  type: 'bullish' | 'bearish';
  midpoint: number;
  upper: number;
  lower: number;
  barIndex: number;
}

// Detect Fair Value Gap (3-bar pattern) with optional ATR filter
function detectFVG(
  bars: TopstepXFuturesBar[],
  currentIndex: number,
  minFVGSizeATR: number = 0,
  atrValue: number = 0
): FVG | null {
  if (currentIndex < 2) return null;

  const current = bars[currentIndex];
  const twoAgo = bars[currentIndex - 2];

  // Bullish FVG: low[t] > high[t-2] (gap down, unfilled space above)
  if (current.low > twoAgo.high) {
    const fvgSize = current.low - twoAgo.high;

    // Check minimum size if ATR filter enabled
    if (minFVGSizeATR > 0 && atrValue > 0) {
      const minSize = minFVGSizeATR * atrValue;
      if (fvgSize < minSize) {
        return null; // FVG too small
      }
    }

    return {
      type: 'bullish',
      midpoint: (current.low + twoAgo.high) / 2,
      upper: current.low,
      lower: twoAgo.high,
      barIndex: currentIndex,
    };
  }

  // Bearish FVG: high[t] < low[t-2] (gap up, unfilled space below)
  if (current.high < twoAgo.low) {
    const fvgSize = twoAgo.low - current.high;

    // Check minimum size if ATR filter enabled
    if (minFVGSizeATR > 0 && atrValue > 0) {
      const minSize = minFVGSizeATR * atrValue;
      if (fvgSize < minSize) {
        return null; // FVG too small
      }
    }

    return {
      type: 'bearish',
      midpoint: (current.high + twoAgo.low) / 2,
      upper: twoAgo.low,
      lower: current.high,
      barIndex: currentIndex,
    };
  }

  return null;
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

// Get previous day's high/low
function getPreviousDayHighLow(
  bars: TopstepXFuturesBar[],
  currentIndex: number
): { prevHigh: number; prevLow: number } | null {
  if (currentIndex < 1) return null;

  const currentBar = bars[currentIndex];
  const currentDate = new Date(currentBar.timestamp);
  const currentDay = currentDate.toISOString().split('T')[0];

  // Find the previous day
  let prevHigh = -Infinity;
  let prevLow = Infinity;
  let foundPrevDay = false;

  for (let i = currentIndex - 1; i >= 0; i--) {
    const bar = bars[i];
    const barDate = new Date(bar.timestamp);
    const barDay = barDate.toISOString().split('T')[0];

    if (barDay === currentDay) {
      continue; // Same day
    }

    // Different day - collect all bars from that day
    if (!foundPrevDay) {
      foundPrevDay = true;
      prevHigh = bar.high;
      prevLow = bar.low;
    } else if (barDay === barDate.toISOString().split('T')[0]) {
      prevHigh = Math.max(prevHigh, bar.high);
      prevLow = Math.min(prevLow, bar.low);
    } else {
      break; // Moved to day before previous
    }
  }

  return foundPrevDay ? { prevHigh, prevLow } : null;
}

// Get session ranges (Asia, London) from current day
interface SessionRange {
  high: number;
  low: number;
}

function getSessionRanges(
  bars: TopstepXFuturesBar[],
  currentIndex: number
): { asia: SessionRange | null; london: SessionRange | null } {
  const currentBar = bars[currentIndex];
  const currentDate = new Date(currentBar.timestamp);
  const currentDay = currentDate.toISOString().split('T')[0];

  let asiaHigh = -Infinity;
  let asiaLow = Infinity;
  let londonHigh = -Infinity;
  let londonLow = Infinity;
  let hasAsia = false;
  let hasLondon = false;

  // Look back through current day for session bars
  for (let i = currentIndex - 1; i >= 0; i--) {
    const bar = bars[i];
    const barDate = new Date(bar.timestamp);
    const barDay = barDate.toISOString().split('T')[0];

    if (barDay !== currentDay) break; // Different day

    const session = getSessionType(bar.timestamp);

    if (session === 'asia') {
      asiaHigh = Math.max(asiaHigh, bar.high);
      asiaLow = Math.min(asiaLow, bar.low);
      hasAsia = true;
    } else if (session === 'london') {
      londonHigh = Math.max(londonHigh, bar.high);
      londonLow = Math.min(londonLow, bar.low);
      hasLondon = true;
    }
  }

  return {
    asia: hasAsia ? { high: asiaHigh, low: asiaLow } : null,
    london: hasLondon ? { high: londonHigh, low: londonLow } : null,
  };
}

// Get NY session highs/lows so far today
function getNYSessionRange(
  bars: TopstepXFuturesBar[],
  currentIndex: number
): SessionRange | null {
  const currentBar = bars[currentIndex];
  const currentDate = new Date(currentBar.timestamp);
  const currentDay = currentDate.toISOString().split('T')[0];

  let nyHigh = -Infinity;
  let nyLow = Infinity;
  let hasNY = false;

  for (let i = currentIndex - 1; i >= 0; i--) {
    const bar = bars[i];
    const barDate = new Date(bar.timestamp);
    const barDay = barDate.toISOString().split('T')[0];

    if (barDay !== currentDay) break;

    const session = getSessionType(bar.timestamp);

    if (session === 'ny') {
      nyHigh = Math.max(nyHigh, bar.high);
      nyLow = Math.min(nyLow, bar.low);
      hasNY = true;
    }
  }

  return hasNY ? { high: nyHigh, low: nyLow } : null;
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('ICT LIQUIDITY-SWEEP + FVG RETURN BACKTEST (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`Reclaim Bars: ${CONFIG.reclaimBars} | Stop Buffer: ${CONFIG.stopLossBuffer} ticks`);
  console.log(`TP1: ${CONFIG.tp1RMultiple}R (scale ${CONFIG.scaleOutPercent * 100}%) | TP2: ${CONFIG.tp2RMultiple}R`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Session: NY (09:30-11:30, 13:30-15:30 ET)`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[ICT-Sweep] Unable to fetch metadata:', err.message);
    return null;
  });

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 20;
  const tickSize = metadata.tickSize || 0.25;
  const baseSymbol = getBaseSymbol(CONFIG.symbol);

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize}`);

  console.log('\nFetching 1-minute bars...');
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: CONFIG.start,
    endTime: CONFIG.end,
    unit: 2,
    unitNumber: 1,
    limit: 50000,
  });

  if (!bars.length) {
    throw new Error('No 1-minute bars returned.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-minute bars`);

  // Calculate indicators
  const atrValues = calculateATR(bars, 14);
  const emaValues = calculateEMA(bars, 20);

  console.log(`Trend Filter: ${CONFIG.useTrendFilter ? 'ENABLED (20-EMA)' : 'DISABLED'}`);
  console.log(`Time Filter: ${CONFIG.useTimeFilter ? 'ENABLED (09:30-11:00 only)' : 'DISABLED'}`);
  console.log(`FVG Size Filter: ${CONFIG.minFVGSizeATR > 0 ? `ENABLED (min ${CONFIG.minFVGSizeATR * 100}% of ATR)` : 'DISABLED'}`);

  const roundToTick = (price: number): number => {
    return Math.round(price / tickSize) * tickSize;
  };

  const trades: TradeRecord[] = [];
  let realizedPnL = 0;
  let totalPositions = 0; // Track number of position entries

  interface Position {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    stopLoss: number;
    tp1: number;
    tp2: number;
    remainingQty: number;
    scaledQty: number;
    feesPaid: number;
    sweepPrice: number;
    fvgMidpoint: number;
  }

  let position: Position | null = null;

  // Track sweep state
  interface SweepState {
    type: 'sell_side' | 'buy_side';
    sweepPrice: number;
    sweepTime: string;
    sweepBarIndex: number;
    prevDayLevel: number;
  }

  let sweepState: SweepState | null = null;

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
    qty: number,
  ) => {
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * qty;
    const exitFees = addFees(baseSymbol, qty);
    const totalFees = qty === position.remainingQty + position.scaledQty
      ? position.feesPaid + exitFees
      : exitFees;
    const grossPnl = rawPnL;
    const netPnl = grossPnl - totalFees;

    const trade: TradeRecord = {
      entryTime: position.entryTime,
      exitTime,
      side: position.side,
      entryPrice: position.entryPrice,
      exitPrice,
      pnl: netPnl,
      grossPnl,
      fees: totalFees,
      exitReason: reason,
      sweepPrice: position.sweepPrice,
      fvgMidpoint: position.fvgMidpoint,
    };

    trades.push(trade);
    realizedPnL += netPnl;

    // Update position qty
    if (qty >= position.remainingQty) {
      position = null;
    } else {
      position.remainingQty -= qty;
    }
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const inSession = isNYSession(bar.timestamp, CONFIG.useTimeFilter);

    // Get current indicators (need at least 20 bars for EMA)
    const atrIndex = i >= 14 ? i - 14 : -1;
    const emaIndex = i >= 20 ? i - 20 : -1;
    const currentATR = atrIndex >= 0 ? atrValues[atrIndex] : 0;
    const currentEMA = emaIndex >= 0 ? emaValues[emaIndex] : 0;

    // Get previous day high/low
    const prevDayHL = getPreviousDayHighLow(bars, i);
    if (!prevDayHL) continue;

    const { prevHigh, prevLow } = prevDayHL;

    // Manage existing position
    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // Check TP1 (scale out)
      if (position.remainingQty === CONFIG.numberOfContracts) {
        const hitTP1 = (direction === 1 && bar.high >= position.tp1) ||
                       (direction === -1 && bar.low <= position.tp1);

        if (hitTP1) {
          const basePrice = direction === 1
            ? Math.min(bar.high, position.tp1)
            : Math.max(bar.low, position.tp1);
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const exitPrice = roundToTick(fillTP(baseSymbol, closeSide, basePrice));

          const scaleQty = Math.floor(CONFIG.numberOfContracts * CONFIG.scaleOutPercent);
          console.log(
            `[${bar.timestamp}] TP1 HIT ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
            `Scale ${scaleQty} @ ${exitPrice.toFixed(2)}`
          );

          exitPosition(exitPrice, bar.timestamp, 'tp1', scaleQty);

          if (position) {
            // Move stop to breakeven
            position.stopLoss = position.entryPrice;
            position.scaledQty = scaleQty;
            console.log(`[${bar.timestamp}] Stop moved to breakeven @ ${position.stopLoss.toFixed(2)}`);
          }
          continue;
        }
      }

      // Check TP2 (full exit)
      if (position) {
        const hitTP2 = (direction === 1 && bar.high >= position.tp2) ||
                       (direction === -1 && bar.low <= position.tp2);

        if (hitTP2) {
          const basePrice = direction === 1
            ? Math.min(bar.high, position.tp2)
            : Math.max(bar.low, position.tp2);
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const exitPrice = roundToTick(fillTP(baseSymbol, closeSide, basePrice));

          console.log(
            `[${bar.timestamp}] TP2 HIT ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
            `Exit remaining @ ${exitPrice.toFixed(2)}`
          );
          exitPosition(exitPrice, bar.timestamp, 'tp2', position.remainingQty);
          continue;
        }
      }

      // Check stop loss
      if (position) {
        const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                       (direction === -1 && bar.high >= position.stopLoss);

        if (hitStop) {
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const stopExitPrice = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss));

          console.log(
            `[${bar.timestamp}] STOP HIT ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
            `Exit @ ${stopExitPrice.toFixed(2)}`
          );
          exitPosition(stopExitPrice, bar.timestamp, 'stop', position.remainingQty + position.scaledQty);
          continue;
        }
      }

      // Exit at session end
      if (position && !inSession) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(baseSymbol, closeSide, bar.close));
        exitPosition(sessionExitPrice, bar.timestamp, 'session', position.remainingQty + position.scaledQty);
        continue;
      }

      continue;
    }

    // Only trade during NY session
    if (!inSession) {
      sweepState = null;
      continue;
    }

    // Step 1: Detect liquidity sweep
    if (!sweepState) {
      // Sell-side sweep (for long): low breaks prevLow
      if (bar.low < prevLow) {
        sweepState = {
          type: 'sell_side',
          sweepPrice: bar.low,
          sweepTime: bar.timestamp,
          sweepBarIndex: i,
          prevDayLevel: prevLow,
        };
        console.log(
          `[${bar.timestamp}] SELL-SIDE SWEEP detected @ ${bar.low.toFixed(2)} ` +
          `(prev low: ${prevLow.toFixed(2)})`
        );
      }
      // Buy-side sweep (for short): high breaks prevHigh
      else if (bar.high > prevHigh) {
        sweepState = {
          type: 'buy_side',
          sweepPrice: bar.high,
          sweepTime: bar.timestamp,
          sweepBarIndex: i,
          prevDayLevel: prevHigh,
        };
        console.log(
          `[${bar.timestamp}] BUY-SIDE SWEEP detected @ ${bar.high.toFixed(2)} ` +
          `(prev high: ${prevHigh.toFixed(2)})`
        );
      }
    }

    // Step 2: Check for reclaim after sweep
    if (sweepState) {
      const barsAfterSweep = i - sweepState.sweepBarIndex;

      // Timeout if too many bars passed
      if (barsAfterSweep > CONFIG.reclaimBars) {
        sweepState = null;
        continue;
      }

      // Check for reclaim
      let reclaimed = false;
      if (sweepState.type === 'sell_side' && bar.close > sweepState.prevDayLevel) {
        reclaimed = true;
        console.log(
          `[${bar.timestamp}] RECLAIM UP @ ${bar.close.toFixed(2)} ` +
          `(${barsAfterSweep} bars after sweep)`
        );
      } else if (sweepState.type === 'buy_side' && bar.close < sweepState.prevDayLevel) {
        reclaimed = true;
        console.log(
          `[${bar.timestamp}] RECLAIM DOWN @ ${bar.close.toFixed(2)} ` +
          `(${barsAfterSweep} bars after sweep)`
        );
      }

      if (!reclaimed) continue;

      // Step 3: Look for FVG after reclaim (with ATR filter if enabled)
      const fvg = detectFVG(bars, i, CONFIG.minFVGSizeATR, currentATR);

      if (fvg) {
        // Long setup: sell-side sweep + bullish FVG
        // Check trend filter: only long if price above EMA (if enabled)
        const trendAllowsLong = !CONFIG.useTrendFilter || currentEMA === 0 || bar.close > currentEMA;

        if (sweepState.type === 'sell_side' && fvg.type === 'bullish' && trendAllowsLong) {
          const entrySide = 'buy';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, fvg.midpoint));
          const stopLoss = roundToTick(sweepState.sweepPrice - CONFIG.stopLossBuffer * tickSize);
          const riskPerContract = entryPrice - stopLoss;
          const tp1 = roundToTick(entryPrice + riskPerContract * CONFIG.tp1RMultiple);
          const tp2 = roundToTick(entryPrice + riskPerContract * CONFIG.tp2RMultiple);

          position = {
            side: 'long',
            entryPrice,
            entryTime: bar.timestamp,
            stopLoss,
            tp1,
            tp2,
            remainingQty: CONFIG.numberOfContracts,
            scaledQty: 0,
            feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
            sweepPrice: sweepState.sweepPrice,
            fvgMidpoint: fvg.midpoint,
          };

          totalPositions++; // Increment position counter

          console.log(
            `[${bar.timestamp}] LONG ENTRY @ ${entryPrice.toFixed(2)} ` +
            `(FVG mid: ${fvg.midpoint.toFixed(2)}, Stop: ${stopLoss.toFixed(2)}, ` +
            `TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)})`
          );

          sweepState = null;
        }
        // Short setup: buy-side sweep + bearish FVG
        // Check trend filter: only short if price below EMA (if enabled)
        else if (sweepState.type === 'buy_side' && fvg.type === 'bearish') {
          const trendAllowsShort = !CONFIG.useTrendFilter || currentEMA === 0 || bar.close < currentEMA;

          if (trendAllowsShort) {
          const entrySide = 'sell';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, fvg.midpoint));
          const stopLoss = roundToTick(sweepState.sweepPrice + CONFIG.stopLossBuffer * tickSize);
          const riskPerContract = stopLoss - entryPrice;
          const tp1 = roundToTick(entryPrice - riskPerContract * CONFIG.tp1RMultiple);
          const tp2 = roundToTick(entryPrice - riskPerContract * CONFIG.tp2RMultiple);

          position = {
            side: 'short',
            entryPrice,
            entryTime: bar.timestamp,
            stopLoss,
            tp1,
            tp2,
            remainingQty: CONFIG.numberOfContracts,
            scaledQty: 0,
            feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
            sweepPrice: sweepState.sweepPrice,
            fvgMidpoint: fvg.midpoint,
          };

          totalPositions++; // Increment position counter

          console.log(
            `[${bar.timestamp}] SHORT ENTRY @ ${entryPrice.toFixed(2)} ` +
            `(FVG mid: ${fvg.midpoint.toFixed(2)}, Stop: ${stopLoss.toFixed(2)}, ` +
            `TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)})`
          );

          sweepState = null;
          }
        }
      }
    }
  }

  // Close any open position at end
  if (position) {
    const lastBar = bars[bars.length - 1];
    const closeSide = position.side === 'long' ? 'sell' : 'buy';
    const endExitPrice = roundToTick(fillStop(baseSymbol, closeSide, lastBar.close));
    exitPosition(endExitPrice, lastBar.timestamp, 'end_of_data', position.remainingQty + position.scaledQty);
  }

  // Performance metrics
  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

  const avgWin = winningTrades.length
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length
    ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length)
    : 0;

  const grossProfit = trades.filter(t => t.grossPnl > 0).reduce((sum, t) => sum + t.grossPnl, 0);
  const grossLoss = Math.abs(trades.filter(t => t.grossPnl <= 0).reduce((sum, t) => sum + t.grossPnl, 0));
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

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
  console.log('BACKTEST SUMMARY');
  console.log('='.repeat(80));
  console.log(`Positions: ${totalPositions} | Trade Legs: ${trades.length} (Wins: ${winningTrades.length}, Losses: ${losingTrades.length})`);
  console.log(`Win Rate: ${winRate.toFixed(1)}% (by leg)`);
  console.log(`Net Realized PnL: ${formatCurrency(realizedPnL)} USD | Fees Paid: $${totalFees.toFixed(2)}`);
  console.log(`Gross Profit: ${formatCurrency(grossProfit)} | Gross Loss: ${formatCurrency(grossLoss)}`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  const exitReasons = trades.reduce((acc, t) => {
    acc[t.exitReason] = (acc[t.exitReason] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`\nExit Reasons: ${Object.entries(exitReasons).map(([r, c]) => `${r}=${c}`).join(', ')}`);

  if (trades.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RECENT TRADES');
    console.log('='.repeat(80));
    trades.slice(-10).forEach(trade => {
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl).padStart(10)} ` +
        `(${trade.exitReason}, Sweep: ${trade.sweepPrice.toFixed(2)}, FVG: ${trade.fvgMidpoint.toFixed(2)})`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('ICT Liquidity-Sweep + FVG backtest failed:', err);
  process.exit(1);
});
