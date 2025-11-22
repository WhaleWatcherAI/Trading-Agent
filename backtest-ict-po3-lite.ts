#!/usr/bin/env tsx
/**
 * ICT Strategy #3: Power of Three (PO3) Lite Backtest
 *
 * CORE CONCEPT (Session-based):
 * The "Power of Three" is ICT's session-based model:
 * 1. Accumulation (Asia range)
 * 2. Manipulation (London sweeps one side of Asia range)
 * 3. Distribution (NY trades back into/through the range)
 *
 * Implementation:
 * - Define Asia range: 20:00-00:00 ET (evening session)
 * - London session: 02:00-05:00 ET
 * - NY session: 09:30-11:30 and 13:30-15:30 ET
 *
 * Entry Logic:
 * 1. Track Asia range (high/low during 20:00-00:00 ET)
 * 2. Detect London manipulation:
 *    - If London breaks above Asia high → bearish manipulation
 *    - If London breaks below Asia low → bullish manipulation
 * 3. During NY session, trade back into the range:
 *    - After bullish manipulation (London swept low), look for bullish FVG
 *    - After bearish manipulation (London swept high), look for bearish FVG
 *    - Enter at FVG 50% midpoint, targeting opposite side of Asia range
 *
 * Risk/Targets:
 * - Stop = just beyond London sweep extreme
 * - TP1 = Asia range midpoint
 * - TP2 = Opposite side of Asia range
 * - Move stop to breakeven at TP1
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
  stopLossBuffer: number;         // Ticks beyond sweep extreme for stop
  numberOfContracts: number;
  scaleOutPercent: number;        // % to exit at TP1 (Asia midpoint)
  commissionPerSide: number;
  contractMultiplier?: number;
  minAsiaRangeATR: number;        // Min Asia range as multiple of ATR (e.g., 0.8)
  minFVGSizeTicks: number;        // Min FVG size in ticks (e.g., 6)
  minFVGSizeATR: number;          // Min FVG size as % of ATR (e.g., 0.35)
  minSweepTicks: number;          // Min ticks London must break Asia H/L by (e.g., 5)
  minBarsAfterSweep: number;      // Min bars to wait after London sweep (e.g., 30)
  tp2RangePercent: number;        // TP2 as % of Asia range from mid (e.g., 0.75 = 75%)
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
  asiaHigh: number;
  asiaLow: number;
  londonSweep: number;
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
const DEFAULT_SYMBOL = process.env.ICT_PO3_SYMBOL || 'NQZ5';
const DEFAULT_CONTRACT_ID = process.env.ICT_PO3_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_SYMBOL,
  contractId: DEFAULT_CONTRACT_ID,
  start: process.env.ICT_PO3_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.ICT_PO3_END || new Date().toISOString(),
  stopLossBuffer: Number(process.env.ICT_PO3_SL_BUFFER || '2'),
  numberOfContracts: Number(process.env.ICT_PO3_CONTRACTS || '2'),
  scaleOutPercent: Number(process.env.ICT_PO3_SCALE_PERCENT || '0.5'),
  minAsiaRangeATR: Number(process.env.ICT_PO3_MIN_ASIA_ATR || '0.8'),        // Min 0.8×ATR Asia range
  minFVGSizeTicks: Number(process.env.ICT_PO3_MIN_FVG_TICKS || '6'),         // Min 6 ticks FVG
  minFVGSizeATR: Number(process.env.ICT_PO3_MIN_FVG_ATR || '0.35'),          // Min 35% of ATR
  minSweepTicks: Number(process.env.ICT_PO3_MIN_SWEEP_TICKS || '5'),         // Min 5 ticks sweep
  minBarsAfterSweep: Number(process.env.ICT_PO3_MIN_BARS_AFTER_SWEEP || '30'), // Wait 30 bars
  tp2RangePercent: Number(process.env.ICT_PO3_TP2_RANGE_PCT || '0.75'),      // TP2 = 75% of range
  commissionPerSide: process.env.ICT_PO3_COMMISSION
    ? Number(process.env.ICT_PO3_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_CONTRACT_ID, DEFAULT_SYMBOL], 1.40),
};

// Session detection (ET timezone)
function getSessionType(timestamp: string): 'asia' | 'london' | 'ny' | 'other' {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();
  const minutes = etDate.getUTCMinutes();
  const totalMinutes = hours * 60 + minutes;

  // Asia: 20:00-00:00 ET = 1200-1440 minutes (evening) or 0-0 (midnight)
  if (totalMinutes >= 1200 || totalMinutes < 0) {
    return 'asia';
  }

  // London: 02:00-05:00 ET = 120-300 minutes
  if (totalMinutes >= 120 && totalMinutes <= 300) {
    return 'london';
  }

  // NY: 09:30-11:30 and 13:30-15:30 ET = 570-690 and 810-930 minutes
  const nySession1 = totalMinutes >= 570 && totalMinutes <= 690;
  const nySession2 = totalMinutes >= 810 && totalMinutes <= 930;
  if (nySession1 || nySession2) {
    return 'ny';
  }

  return 'other';
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

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

// Get current trading day (resets at 18:00 ET for futures)
function getTradingDay(timestamp: string): string {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();

  // If before 18:00 ET, use current day; otherwise next day
  if (hours < 18) {
    return etDate.toISOString().split('T')[0];
  } else {
    const nextDay = new Date(etDate);
    nextDay.setUTCDate(nextDay.getUTCDate() + 1);
    return nextDay.toISOString().split('T')[0];
  }
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('ICT POWER OF THREE (PO3) LITE BACKTEST (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`Stop Buffer: ${CONFIG.stopLossBuffer} ticks`);
  console.log(`Scale at TP1 (Asia mid): ${CONFIG.scaleOutPercent * 100}%`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Sessions: Asia (20:00-00:00), London (02:00-05:00), NY (09:30-11:30, 13:30-15:30 ET)`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[ICT-PO3] Unable to fetch metadata:', err.message);
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
    asiaHigh: number;
    asiaLow: number;
    londonSweep: number;
    fvgMidpoint: number;
  }

  let position: Position | null = null;

  // Track daily session data
  interface DayData {
    tradingDay: string;
    asiaHigh: number | null;
    asiaLow: number | null;
    londonManipulation: 'bullish' | 'bearish' | null; // Sweep direction
    londonSweepPrice: number | null;
    londonSweepBarIndex: number | null; // Track when sweep occurred
    enteredToday: boolean; // One trade per day max
  }

  let currentDay: DayData = {
    tradingDay: '',
    asiaHigh: null,
    asiaLow: null,
    londonManipulation: null,
    londonSweepPrice: null,
    londonSweepBarIndex: null,
    enteredToday: false,
  };

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
      asiaHigh: position.asiaHigh,
      asiaLow: position.asiaLow,
      londonSweep: position.londonSweep,
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
    const session = getSessionType(bar.timestamp);
    const tradingDay = getTradingDay(bar.timestamp);

    // Calculate ATR for current bar
    const atr = calculateATR(bars, i, 14);

    // Reset for new trading day
    if (tradingDay !== currentDay.tradingDay) {
      currentDay = {
        tradingDay,
        asiaHigh: null,
        asiaLow: null,
        londonManipulation: null,
        londonSweepPrice: null,
        londonSweepBarIndex: null,
        enteredToday: false,
      };
    }

    // Manage existing position
    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // Check TP1 (Asia midpoint - scale out)
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
            `[${bar.timestamp}] TP1 (Asia Mid) HIT ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
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

      // Check TP2 (opposite Asia range)
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
            `[${bar.timestamp}] TP2 (Asia Range) HIT ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
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

      // Exit at session end (after NY closes)
      if (position && session === 'other') {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(baseSymbol, closeSide, bar.close));
        exitPosition(sessionExitPrice, bar.timestamp, 'session', position.remainingQty + position.scaledQty);
        continue;
      }

      continue;
    }

    // Step 1: Track Asia range (Accumulation)
    if (session === 'asia') {
      if (currentDay.asiaHigh === null || bar.high > currentDay.asiaHigh) {
        currentDay.asiaHigh = bar.high;
      }
      if (currentDay.asiaLow === null || bar.low < currentDay.asiaLow) {
        currentDay.asiaLow = bar.low;
      }
    }

    // Step 2: Detect London manipulation with filters
    if (session === 'london' && currentDay.asiaHigh !== null && currentDay.asiaLow !== null && atr > 0) {
      // Check if Asia range is wide enough (filter choppy days)
      const asiaRange = currentDay.asiaHigh - currentDay.asiaLow;
      const minAsiaRange = CONFIG.minAsiaRangeATR * atr;

      if (asiaRange < minAsiaRange) {
        // Asia range too narrow - skip this day
        continue;
      }

      // Bearish manipulation: London sweeps Asia high
      if (!currentDay.londonManipulation && bar.high > currentDay.asiaHigh) {
        const sweepSize = bar.high - currentDay.asiaHigh;
        const minSweep = CONFIG.minSweepTicks * tickSize;

        if (sweepSize >= minSweep) {
          currentDay.londonManipulation = 'bearish';
          currentDay.londonSweepPrice = bar.high;
          currentDay.londonSweepBarIndex = i;
          console.log(
            `[${bar.timestamp}] LONDON BEARISH MANIPULATION @ ${bar.high.toFixed(2)} ` +
            `(swept Asia high: ${currentDay.asiaHigh.toFixed(2)} by ${sweepSize.toFixed(2)}, Asia range: ${asiaRange.toFixed(2)})`
          );
        }
      }
      // Bullish manipulation: London sweeps Asia low
      else if (!currentDay.londonManipulation && bar.low < currentDay.asiaLow) {
        const sweepSize = currentDay.asiaLow - bar.low;
        const minSweep = CONFIG.minSweepTicks * tickSize;

        if (sweepSize >= minSweep) {
          currentDay.londonManipulation = 'bullish';
          currentDay.londonSweepPrice = bar.low;
          currentDay.londonSweepBarIndex = i;
          console.log(
            `[${bar.timestamp}] LONDON BULLISH MANIPULATION @ ${bar.low.toFixed(2)} ` +
            `(swept Asia low: ${currentDay.asiaLow.toFixed(2)} by ${sweepSize.toFixed(2)}, Asia range: ${asiaRange.toFixed(2)})`
          );
        }
      }
    }

    // Step 3: Trade NY distribution (back into Asia range) with filters
    if (session === 'ny' &&
        currentDay.asiaHigh !== null &&
        currentDay.asiaLow !== null &&
        currentDay.londonManipulation !== null &&
        currentDay.londonSweepPrice !== null &&
        currentDay.londonSweepBarIndex !== null &&
        !currentDay.enteredToday && // Max 1 trade per day
        atr > 0) {

      // Wait minimum bars after London sweep
      const barsAfterSweep = i - currentDay.londonSweepBarIndex;
      if (barsAfterSweep < CONFIG.minBarsAfterSweep) {
        continue; // Too soon after sweep
      }

      const fvg = detectFVG(bars, i, tickSize, atr, CONFIG);

      if (fvg) {
        // Long setup: After bullish London manipulation, look for bullish FVG in NY
        if (currentDay.londonManipulation === 'bullish' && fvg.type === 'bullish') {
          const entrySide = 'buy';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, fvg.midpoint));

          // Entry must be inside Asia range (confirming reversal)
          if (entryPrice >= currentDay.asiaHigh) {
            continue; // Entry too high, not inside range
          }

          // LONG stop must be BELOW entry to cut losses (below FVG lower edge)
          const stopLoss = roundToTick(fvg.lower - CONFIG.stopLossBuffer * tickSize);
          const asiaMid = (currentDay.asiaHigh + currentDay.asiaLow) / 2;
          const tp1 = roundToTick(asiaMid);

          // Better TP2: 75% of Asia range from midpoint (less aggressive)
          const rangeFromMid = (currentDay.asiaHigh - asiaMid) * CONFIG.tp2RangePercent;
          const tp2 = roundToTick(asiaMid + rangeFromMid);

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
            asiaHigh: currentDay.asiaHigh,
            asiaLow: currentDay.asiaLow,
            londonSweep: currentDay.londonSweepPrice,
            fvgMidpoint: fvg.midpoint,
          };

          console.log(
            `[${bar.timestamp}] LONG ENTRY (PO3) @ ${entryPrice.toFixed(2)} ` +
            `(FVG: ${fvg.midpoint.toFixed(2)}, Stop: ${stopLoss.toFixed(2)}, ` +
            `TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)})`
          );

          // Mark day as entered (one trade per day)
          currentDay.enteredToday = true;
          // Reset manipulation state after entry
          currentDay.londonManipulation = null;
        }
        // Short setup: After bearish London manipulation, look for bearish FVG in NY
        else if (currentDay.londonManipulation === 'bearish' && fvg.type === 'bearish') {
          const entrySide = 'sell';
          const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, fvg.midpoint));

          // Entry must be inside Asia range (confirming reversal)
          if (entryPrice <= currentDay.asiaLow) {
            continue; // Entry too low, not inside range
          }

          // SHORT stop must be ABOVE entry to cut losses (above FVG upper edge)
          const stopLoss = roundToTick(fvg.upper + CONFIG.stopLossBuffer * tickSize);
          const asiaMid = (currentDay.asiaHigh + currentDay.asiaLow) / 2;
          const tp1 = roundToTick(asiaMid);

          // Better TP2: 75% of Asia range from midpoint (less aggressive)
          const rangeFromMid = (asiaMid - currentDay.asiaLow) * CONFIG.tp2RangePercent;
          const tp2 = roundToTick(asiaMid - rangeFromMid);

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
            asiaHigh: currentDay.asiaHigh,
            asiaLow: currentDay.asiaLow,
            londonSweep: currentDay.londonSweepPrice,
            fvgMidpoint: fvg.midpoint,
          };

          console.log(
            `[${bar.timestamp}] SHORT ENTRY (PO3) @ ${entryPrice.toFixed(2)} ` +
            `(FVG: ${fvg.midpoint.toFixed(2)}, Stop: ${stopLoss.toFixed(2)}, ` +
            `TP1: ${tp1.toFixed(2)}, TP2: ${tp2.toFixed(2)})`
          );

          // Mark day as entered (one trade per day)
          currentDay.enteredToday = true;
          // Reset manipulation state after entry
          currentDay.londonManipulation = null;
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
        `(${trade.exitReason}, Asia: ${trade.asiaLow.toFixed(2)}-${trade.asiaHigh.toFixed(2)}, ` +
        `London: ${trade.londonSweep.toFixed(2)}, FVG: ${trade.fvgMidpoint.toFixed(2)})`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('ICT Power of Three (PO3) backtest failed:', err);
  process.exit(1);
});
