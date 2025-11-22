#!/usr/bin/env tsx
/**
 * ICT Strategy #2: BOS/CHOCH + FVG Backtest (Trend-Following)
 *
 * CORE CONCEPT:
 * Simpler variant - confirm change of character (CHOCH) or break of structure (BOS)
 * using swing high/low breaks, then take the next FVG in that direction at 50% fill.
 *
 * This is more momentum-friendly and easy to code.
 *
 * Entry Logic:
 * 1. Detect BOS/CHOCH:
 *    - BOS (Break of Structure): Price breaks a swing high (bullish) or swing low (bearish)
 *    - Use pivot detection (e.g., pivotLen=3) to identify swing points
 *    - Bullish BOS: current high > recent swing high
 *    - Bearish BOS: current low < recent swing low
 *
 * 2. Detect FVG after BOS:
 *    - After bullish BOS, wait for bullish FVG (3-bar pattern)
 *    - After bearish BOS, wait for bearish FVG
 *
 * 3. Entry:
 *    - Enter at 50% FVG midpoint with limit order
 *    - Direction: same as BOS direction
 *
 * Risk/Targets:
 * - Stop = just beyond the swing low/high that triggered BOS
 * - TP1 = 1R; TP2 = 2R
 * - Move stop to breakeven at TP1
 *
 * Session filter: NY session only (09:30-11:30, 13:30-15:30 ET)
 */

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
  pivotLength: number;           // Lookback for swing high/low detection
  stopLossBuffer: number;         // Ticks beyond swing extreme for stop
  tp1RMultiple: number;           // First target = 1R
  tp2RMultiple: number;           // Second target = 2R
  numberOfContracts: number;
  scaleOutPercent: number;        // % to exit at TP1
  commissionPerSide: number;
  contractMultiplier?: number;
  fvgLookbackBars: number;        // Max bars to look for FVG after BOS
  minBOSDisplacement: number;     // Min bar range as multiple of ATR (e.g., 1.0)
  minBOSBodyPercent: number;      // Min body % of range (e.g., 0.6 = 60%)
  minFVGSizeATR: number;          // Min FVG size as % of ATR (e.g., 0.35)
  minFVGSizeTicks: number;        // Min FVG size in ticks (e.g., 6)
  cooldownBars: number;           // Bars to wait after any trade
  maxLongsPerSession: number;     // Max long positions per session
  maxShortsPerSession: number;    // Max short positions per session
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
  bosPrice: number;
  fvgMidpoint: number;
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
const DEFAULT_SYMBOL = process.env.ICT_BOS_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.ICT_BOS_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.ICT_BOS_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.ICT_BOS_END || new Date().toISOString(),
  pivotLength: Number(process.env.ICT_BOS_PIVOT_LEN || '6'),  // Increased from 3 to 6
  stopLossBuffer: Number(process.env.ICT_BOS_SL_BUFFER || '2'),
  tp1RMultiple: Number(process.env.ICT_BOS_TP1 || '1'),
  tp2RMultiple: Number(process.env.ICT_BOS_TP2 || '2'),
  numberOfContracts: Number(process.env.ICT_BOS_CONTRACTS || '2'),
  scaleOutPercent: Number(process.env.ICT_BOS_SCALE_PERCENT || '0.5'),
  fvgLookbackBars: Number(process.env.ICT_BOS_FVG_LOOKBACK || '10'),
  minBOSDisplacement: Number(process.env.ICT_BOS_MIN_DISPLACEMENT || '1.0'),  // Min 1.0×ATR bar range
  minBOSBodyPercent: Number(process.env.ICT_BOS_MIN_BODY_PCT || '0.6'),      // Min 60% body
  minFVGSizeATR: Number(process.env.ICT_BOS_MIN_FVG_ATR || '0.35'),          // Min 35% of ATR
  minFVGSizeTicks: Number(process.env.ICT_BOS_MIN_FVG_TICKS || '6'),          // Min 6 ticks
  cooldownBars: Number(process.env.ICT_BOS_COOLDOWN || '20'),                 // 20 bars cooldown
  maxLongsPerSession: Number(process.env.ICT_BOS_MAX_LONGS || '1'),           // Max 1 long/session
  maxShortsPerSession: Number(process.env.ICT_BOS_MAX_SHORTS || '1'),         // Max 1 short/session
  commissionPerSide: process.env.ICT_BOS_COMMISSION
    ? Number(process.env.ICT_BOS_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 1.40),
};

// NY session times in ET (09:30-11:30, 13:30-15:30)
function isNYSession(timestamp: string): boolean {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();
  const minutes = etDate.getUTCMinutes();
  const totalMinutes = hours * 60 + minutes;
  const session1 = totalMinutes >= 570 && totalMinutes <= 690;
  const session2 = totalMinutes >= 810 && totalMinutes <= 930;
  return session1 || session2;
}

// Calculate ATR (Average True Range) using 14-period default
function calculateATR(bars: TopstepXFuturesBar[], currentIndex: number, period: number = 14): number {
  if (currentIndex < period) return 0;

  let sumTR = 0;
  for (let i = 0; i < period; i++) {
    const idx = currentIndex - i;
    const bar = bars[idx];
    const prevBar = idx > 0 ? bars[idx - 1] : bar;

    const high_low = bar.high - bar.low;
    const high_prevClose = Math.abs(bar.high - prevBar.close);
    const low_prevClose = Math.abs(bar.low - prevBar.close);

    const tr = Math.max(high_low, high_prevClose, low_prevClose);
    sumTR += tr;
  }

  return sumTR / period;
}

// Get trading session key for tracking daily limits
function getSessionKey(timestamp: string): string {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  return etDate.toISOString().split('T')[0]; // YYYY-MM-DD in ET
}

// Get previous day's high/low for proximity guard
function getPrevDayHighLow(bars: TopstepXFuturesBar[], currentIndex: number): { high: number; low: number } | null {
  if (currentIndex < 2) return null;

  const currentDate = new Date(bars[currentIndex].timestamp);
  const etCurrentDate = new Date(currentDate.getTime() - 5 * 60 * 60 * 1000);
  const currentDay = etCurrentDate.toISOString().split('T')[0];

  let prevDayHigh = -Infinity;
  let prevDayLow = Infinity;
  let foundPrevDay = false;

  // Look back to find previous day's range
  for (let i = currentIndex - 1; i >= 0; i--) {
    const date = new Date(bars[i].timestamp);
    const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
    const day = etDate.toISOString().split('T')[0];

    if (day !== currentDay) {
      if (!foundPrevDay) {
        foundPrevDay = true;
      }
      prevDayHigh = Math.max(prevDayHigh, bars[i].high);
      prevDayLow = Math.min(prevDayLow, bars[i].low);
    } else if (foundPrevDay) {
      break; // We've collected all prev day bars
    }
  }

  if (!foundPrevDay || prevDayHigh === -Infinity) return null;
  return { high: prevDayHigh, low: prevDayLow };
}

interface FVG {
  type: 'bullish' | 'bearish';
  midpoint: number;
  upper: number;
  lower: number;
  barIndex: number;
}

function detectFVG(
  bars: TopstepXFuturesBar[],
  currentIndex: number,
  tickSize: number,
  atr: number,
  config: BacktestConfig
): FVG | null {
  if (currentIndex < 2) return null;

  const current = bars[currentIndex];
  const twoAgo = bars[currentIndex - 2];

  // Bullish FVG: low[t] > high[t-2]
  if (current.low > twoAgo.high) {
    const fvgSize = current.low - twoAgo.high;
    const minSize = Math.max(
      config.minFVGSizeTicks * tickSize,
      config.minFVGSizeATR * atr
    );

    if (fvgSize < minSize) return null;

    return {
      type: 'bullish',
      midpoint: (current.low + twoAgo.high) / 2,
      upper: current.low,
      lower: twoAgo.high,
      barIndex: currentIndex,
    };
  }

  // Bearish FVG: high[t] < low[t-2]
  if (current.high < twoAgo.low) {
    const fvgSize = twoAgo.low - current.high;
    const minSize = Math.max(
      config.minFVGSizeTicks * tickSize,
      config.minFVGSizeATR * atr
    );

    if (fvgSize < minSize) return null;

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

// Detect swing high/low using pivot method
interface Pivot {
  type: 'high' | 'low';
  price: number;
  barIndex: number;
}

function detectSwingPivots(
  bars: TopstepXFuturesBar[],
  currentIndex: number,
  pivotLength: number
): Pivot[] {
  const pivots: Pivot[] = [];

  if (currentIndex < pivotLength * 2) return pivots;

  // Check for swing high
  const midBar = bars[currentIndex - pivotLength];
  let isSwingHigh = true;
  let isSwingLow = true;

  for (let j = 1; j <= pivotLength; j++) {
    const leftBar = bars[currentIndex - pivotLength - j];
    const rightBar = bars[currentIndex - pivotLength + j];

    if (leftBar.high >= midBar.high || rightBar.high >= midBar.high) {
      isSwingHigh = false;
    }
    if (leftBar.low <= midBar.low || rightBar.low <= midBar.low) {
      isSwingLow = false;
    }
  }

  if (isSwingHigh) {
    pivots.push({
      type: 'high',
      price: midBar.high,
      barIndex: currentIndex - pivotLength,
    });
  }

  if (isSwingLow) {
    pivots.push({
      type: 'low',
      price: midBar.low,
      barIndex: currentIndex - pivotLength,
    });
  }

  return pivots;
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('ICT BOS/CHOCH + FVG BACKTEST (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`Pivot Length: ${CONFIG.pivotLength} | FVG Lookback: ${CONFIG.fvgLookbackBars} bars`);
  console.log(`Stop Buffer: ${CONFIG.stopLossBuffer} ticks`);
  console.log(`TP1: ${CONFIG.tp1RMultiple}R (scale ${CONFIG.scaleOutPercent * 100}%) | TP2: ${CONFIG.tp2RMultiple}R`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Session: NY (09:30-11:30, 13:30-15:30 ET)`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[ICT-BOS] Unable to fetch metadata:', err.message);
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

  const roundToTick = (price: number): number => {
    return Math.round(price / tickSize) * tickSize;
  };

  const trades: TradeRecord[] = [];
  let realizedPnL = 0;

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
    bosPrice: number;
    fvgMidpoint: number;
  }

  let position: Position | null = null;

  // Track BOS state
  interface BOSState {
    type: 'bullish' | 'bearish';
    bosPrice: number;
    bosTime: string;
    bosBarIndex: number;
    swingExtreme: number; // The swing low/high that was broken
    consumed: boolean; // Single pullback rule: only first FVG after BOS
  }

  let bosState: BOSState | null = null;

  // Track most recent swing pivots
  let lastSwingHigh: Pivot | null = null;
  let lastSwingLow: Pivot | null = null;

  // Track cooldown: bar index of last entry
  let lastEntryBarIndex: number = -9999;

  // Track session counters
  const sessionCounters: Record<string, { longs: number; shorts: number }> = {};

  // Track today's open for proximity guard
  let todayOpen: number | null = null;
  let currentSessionKey: string | null = null;

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
      bosPrice: position.bosPrice,
      fvgMidpoint: position.fvgMidpoint,
    };

    trades.push(trade);
    realizedPnL += netPnl;

    if (qty >= position.remainingQty) {
      position = null;
    } else {
      position.remainingQty -= qty;
    }
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const inSession = isNYSession(bar.timestamp);

    // Track session changes and reset counters
    const sessionKey = getSessionKey(bar.timestamp);
    if (sessionKey !== currentSessionKey) {
      currentSessionKey = sessionKey;
      if (!sessionCounters[sessionKey]) {
        sessionCounters[sessionKey] = { longs: 0, shorts: 0 };
      }
      todayOpen = bar.open; // Set today's open at first bar of new session
    }

    // Calculate ATR for current bar
    const atr = calculateATR(bars, i, 14);

    // Get prev day H/L for proximity guard
    const prevDayHL = getPrevDayHighLow(bars, i);

    // Detect swing pivots
    const pivots = detectSwingPivots(bars, i, CONFIG.pivotLength);
    for (const pivot of pivots) {
      if (pivot.type === 'high') {
        lastSwingHigh = pivot;
      } else {
        lastSwingLow = pivot;
      }
    }

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
            position.stopLoss = position.entryPrice;
            position.scaledQty = scaleQty;
            console.log(`[${bar.timestamp}] Stop moved to breakeven @ ${position.stopLoss.toFixed(2)}`);
          }
          continue;
        }
      }

      // Check TP2
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
      bosState = null;
      continue;
    }

    // Step 1: Detect BOS (Break of Structure) with filters
    if (!bosState || bosState.consumed) {
      // Reset BOS if consumed
      if (bosState && bosState.consumed) {
        bosState = null;
      }

      // Check cooldown
      const barsAfterLastEntry = i - lastEntryBarIndex;
      if (barsAfterLastEntry < CONFIG.cooldownBars) {
        continue; // Still in cooldown
      }

      // Bullish BOS: current high breaks last swing high
      if (lastSwingHigh && bar.high > lastSwingHigh.price && atr > 0) {
        // Check displacement: bar range >= minBOSDisplacement * ATR
        const barRange = bar.high - bar.low;
        const minDisplacement = CONFIG.minBOSDisplacement * atr;
        if (barRange < minDisplacement) continue;

        // Check body percentage: (close - low) / range >= minBOSBodyPercent (for bullish)
        const bodySize = Math.abs(bar.close - bar.open);
        const bodyPercent = bodySize / barRange;
        if (bodyPercent < CONFIG.minBOSBodyPercent) continue;

        // Check if close is near top of bar (bullish conviction)
        const closeFromTop = (bar.high - bar.close) / barRange;
        if (closeFromTop > 0.4) continue; // Close should be in top 60% of bar

        // Check session quota
        if (sessionCounters[sessionKey].longs >= CONFIG.maxLongsPerSession) {
          continue; // Already hit max longs for this session
        }

        // Proximity guard: skip if near prev day H/L or today's open
        if (prevDayHL && todayOpen) {
          const proximityTicks = 10;
          const proximityBuffer = proximityTicks * tickSize;
          const nearPrevDayLow = Math.abs(bar.low - prevDayHL.low) < proximityBuffer;
          const nearPrevDayHigh = Math.abs(bar.high - prevDayHL.high) < proximityBuffer;
          const nearTodayOpen = Math.abs(bar.close - todayOpen) < proximityBuffer;
          if (nearPrevDayLow || nearPrevDayHigh || nearTodayOpen) continue;
        }

        bosState = {
          type: 'bullish',
          bosPrice: bar.high,
          bosTime: bar.timestamp,
          bosBarIndex: i,
          swingExtreme: lastSwingHigh.price,
          consumed: false,
        };
        console.log(
          `[${bar.timestamp}] BULLISH BOS @ ${bar.high.toFixed(2)} ` +
          `(broke swing high: ${lastSwingHigh.price.toFixed(2)}, range: ${barRange.toFixed(2)}, ATR: ${atr.toFixed(2)})`
        );
      }
      // Bearish BOS: current low breaks last swing low
      else if (lastSwingLow && bar.low < lastSwingLow.price && atr > 0) {
        // Check displacement: bar range >= minBOSDisplacement * ATR
        const barRange = bar.high - bar.low;
        const minDisplacement = CONFIG.minBOSDisplacement * atr;
        if (barRange < minDisplacement) continue;

        // Check body percentage
        const bodySize = Math.abs(bar.close - bar.open);
        const bodyPercent = bodySize / barRange;
        if (bodyPercent < CONFIG.minBOSBodyPercent) continue;

        // Check if close is near bottom of bar (bearish conviction)
        const closeFromBottom = (bar.close - bar.low) / barRange;
        if (closeFromBottom > 0.4) continue; // Close should be in bottom 60% of bar

        // Check session quota
        if (sessionCounters[sessionKey].shorts >= CONFIG.maxShortsPerSession) {
          continue; // Already hit max shorts for this session
        }

        // Proximity guard
        if (prevDayHL && todayOpen) {
          const proximityTicks = 10;
          const proximityBuffer = proximityTicks * tickSize;
          const nearPrevDayLow = Math.abs(bar.low - prevDayHL.low) < proximityBuffer;
          const nearPrevDayHigh = Math.abs(bar.high - prevDayHL.high) < proximityBuffer;
          const nearTodayOpen = Math.abs(bar.close - todayOpen) < proximityBuffer;
          if (nearPrevDayLow || nearPrevDayHigh || nearTodayOpen) continue;
        }

        bosState = {
          type: 'bearish',
          bosPrice: bar.low,
          bosTime: bar.timestamp,
          bosBarIndex: i,
          swingExtreme: lastSwingLow.price,
          consumed: false,
        };
        console.log(
          `[${bar.timestamp}] BEARISH BOS @ ${bar.low.toFixed(2)} ` +
          `(broke swing low: ${lastSwingLow.price.toFixed(2)}, range: ${barRange.toFixed(2)}, ATR: ${atr.toFixed(2)})`
        );
      }
    }

    // Step 2: Look for FVG after BOS
    if (bosState && !bosState.consumed) {
      const barsAfterBOS = i - bosState.bosBarIndex;

      // Timeout if too many bars passed
      if (barsAfterBOS > CONFIG.fvgLookbackBars) {
        bosState = null;
        continue;
      }

      const fvg = detectFVG(bars, i, tickSize, atr, CONFIG);

      if (fvg) {
        // Long setup: bullish BOS + bullish FVG
        if (bosState.type === 'bullish' && fvg.type === 'bullish') {
          const entrySide = 'buy';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, fvg.midpoint));
          const stopLoss = roundToTick(bosState.swingExtreme - CONFIG.stopLossBuffer * tickSize);
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
            bosPrice: bosState.bosPrice,
            fvgMidpoint: fvg.midpoint,
          };

          console.log(
            `[${bar.timestamp}] LONG ENTRY @ ${entryPrice.toFixed(2)} ` +
            `(FVG mid: ${fvg.midpoint.toFixed(2)}, Stop: ${stopLoss.toFixed(2)}, ` +
            `TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)})`
          );

          // Mark BOS as consumed (single pullback rule)
          bosState.consumed = true;

          // Update session counter
          sessionCounters[sessionKey].longs++;

          // Update cooldown
          lastEntryBarIndex = i;
        }
        // Short setup: bearish BOS + bearish FVG
        else if (bosState.type === 'bearish' && fvg.type === 'bearish') {
          const entrySide = 'sell';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, fvg.midpoint));
          const stopLoss = roundToTick(bosState.swingExtreme + CONFIG.stopLossBuffer * tickSize);
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
            bosPrice: bosState.bosPrice,
            fvgMidpoint: fvg.midpoint,
          };

          console.log(
            `[${bar.timestamp}] SHORT ENTRY @ ${entryPrice.toFixed(2)} ` +
            `(FVG mid: ${fvg.midpoint.toFixed(2)}, Stop: ${stopLoss.toFixed(2)}, ` +
            `TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)})`
          );

          // Mark BOS as consumed (single pullback rule)
          bosState.consumed = true;

          // Update session counter
          sessionCounters[sessionKey].shorts++;

          // Update cooldown
          lastEntryBarIndex = i;
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
  console.log(`Total Trades: ${trades.length} | Wins: ${winningTrades.length} | Losses: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
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
        `(${trade.exitReason}, BOS: ${trade.bosPrice.toFixed(2)}, FVG: ${trade.fvgMidpoint.toFixed(2)})`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('ICT BOS/CHOCH + FVG backtest failed:', err);
  process.exit(1);
});
