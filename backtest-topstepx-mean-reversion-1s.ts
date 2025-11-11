#!/usr/bin/env tsx
/**
 * TopstepX Mean Reversion Backtest on 1-Second Bars
 *
 * OPTIMIZED FOR: MES (Micro E-mini S&P 500)
 * 3 contracts - EXIT ALL at middle band (no scaling)
 *
 * Strategy:
 * - Bollinger Bands: 20-period SMA with 3 standard deviations
 * - RSI confirmation: RSI(24) with 30/70 levels (oversold/overbought)
 * - Two-stage entry:
 *   Stage 1: Price touches BB outer band + RSI extreme creates setup
 *   Stage 2: TTM Squeeze ON triggers entry
 * - FVG Filter: DISABLED
 * - ADX Filter: DISABLED (more entries, higher trade volume)
 * - Contracts: 3 total
 * - Exit Strategy: ALL 3 contracts exit at middle band (20-period SMA)
 * - Stop Loss: 0.04% from entry, quick exits if wrong
 * - Target Time: ~1-10 seconds typical hold
 */

import { RSI, ADX, ATR, EMA, MACD } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import * as fs from 'fs';
import * as path from 'path';
import { calculateTtmSqueeze } from './lib/ttmSqueeze';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import {
  calculateSessionProfile,
  calculateBias,
  SessionProfile,
  Bias,
} from './lib/svpFramework';

interface BacktestConfig {
  symbol: string;
  contractId?: string;
  start: string;
  end: string;
  bbPeriod: number;
  bbStdDev: number;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  adxPeriod: number;
  adxThreshold: number;
  bypassAdx: boolean;
  adxBuffer: number;
  fastSmaPeriod: number;
  atrPeriod: number;
  atrMinRangePercent: number;
  htfEmaPeriod: number;
  htfEmaEnabled: boolean;
  fvgEnabled: boolean;
  contractMultiplier?: number;
  numberOfContracts: number;
  commissionPerSide: number;
  slippageTicks: number;
  svpBiasFilterEnabled: boolean;
  trailingStopPercent: number;
  useTrailingStop: boolean;
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
  exitReason: 'stop' | 'target' | 'scale' | 'session' | 'end_of_data';
  entryRSI: number;
  entryADX?: number;
  scaled: boolean;
  scalePnL?: number;
  finalPnL?: number;
  entrySlippageTicks?: number;
  exitSlippageTicks?: number;
  slippageCost?: number;
  bias?: Bias;
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DEFAULT_DAYS = 7;

const DEFAULT_MR_SYMBOL = process.env.TOPSTEPX_MR_SYMBOL || 'MESZ5';
const DEFAULT_MR_CONTRACT_ID = process.env.TOPSTEPX_MR_CONTRACT_ID;

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

// Helper functions for realistic fill simulation
// Matches the clean deterministic model:
//   Entry (aggressive): mid ± (0.5*spread + σ_entry)
//   TP (passive/agg mix): mid ∓ E_tp_ticks where E_tp_ticks = (1-p)*(spread + σ_tp)
//   Stop (adverse): trigger ∓ σ_stop
//   Fees: applied after price calculations

function tpCostTicks(sym: string): number {
  const S = SLIP_CONFIG.avgSpreadTicks[sym];        // usually 1
  const sig = SLIP_CONFIG.slipAvg.tp[sym];          // avg TP slippage (ticks)
  const p = SLIP_CONFIG.p_tp_passive[sym];          // passive probability
  return (1 - p) * (S + sig);                       // expected ticks
}
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

  // Expected TP cost in ticks: E[cost_tp_ticks] = (1 - p_passive) * (spread_ticks + σ_tp_ticks)
  const E_tp_ticks = (1 - p_passive) * (S_ticks + sigma_tp_ticks);

  // Close long (sell): fill = mid - E_tp_ticks * t
  // Close short (buy): fill = mid + E_tp_ticks * t
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

// Extract base symbol (MESZ5 -> MES, ESZ5 -> ES, GCZ5 -> GC)
const getBaseSymbol = (fullSymbol: string): string => {
  // Remove futures month code (Z5, H5, etc.) and digits
  return fullSymbol.replace(/[A-Z]\d+$/, '');
};

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_MR_SYMBOL,
  contractId: DEFAULT_MR_CONTRACT_ID,
  start: process.env.TOPSTEPX_MR_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.TOPSTEPX_MR_END || new Date().toISOString(),
  bbPeriod: Number(process.env.TOPSTEPX_MR_BB_PERIOD || '20'),
  bbStdDev: Number(process.env.TOPSTEPX_MR_BB_STDDEV || '3'),
  rsiPeriod: Number(process.env.TOPSTEPX_MR_RSI_PERIOD || '24'),
  rsiOversold: Number(process.env.TOPSTEPX_MR_RSI_OVERSOLD || '30'),
  rsiOverbought: Number(process.env.TOPSTEPX_MR_RSI_OVERBOUGHT || '70'),
  stopLossPercent: Number(process.env.TOPSTEPX_MR_STOP_LOSS_PERCENT || '0.0004'),
  takeProfitPercent: Number(process.env.TOPSTEPX_MR_TAKE_PROFIT_PERCENT || '0.0024'),
  adxPeriod: Number(process.env.TOPSTEPX_MR_ADX_PERIOD || '900'),
  adxThreshold: Number(process.env.TOPSTEPX_MR_ADX_THRESHOLD || '22'),
  bypassAdx: process.env.TOPSTEPX_MR_BYPASS_ADX === 'true' || true,
  adxBuffer: Number(process.env.TOPSTEPX_MR_ADX_BUFFER || '4'),
  fastSmaPeriod: Number(process.env.TOPSTEPX_MR_FAST_SMA_PERIOD || '50'),
  atrPeriod: Number(process.env.TOPSTEPX_MR_ATR_PERIOD || '900'),
  atrMinRangePercent: Number(process.env.TOPSTEPX_MR_ATR_MIN_RANGE_PERCENT || '0'),
  htfEmaPeriod: Number(process.env.TOPSTEPX_MR_HTF_EMA_PERIOD || '200'),
  htfEmaEnabled: process.env.TOPSTEPX_MR_HTF_EMA_ENABLED === 'true',
  fvgEnabled: process.env.TOPSTEPX_MR_FVG_ENABLED === 'true',
  numberOfContracts: Number(process.env.TOPSTEPX_MR_CONTRACTS || '3'),
  commissionPerSide: process.env.TOPSTEPX_MR_COMMISSION
    ? Number(process.env.TOPSTEPX_MR_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_MR_CONTRACT_ID, DEFAULT_MR_SYMBOL], 0.35),
  slippageTicks: Number(process.env.TOPSTEPX_MR_SLIPPAGE_TICKS || '1'),
  svpBiasFilterEnabled: process.env.TOPSTEPX_MR_SVP_BIAS_FILTER === 'true',
  trailingStopPercent: Number(process.env.TOPSTEPX_MR_TRAILING_STOP_PERCENT || '0.0004'),
  useTrailingStop: process.env.TOPSTEPX_MR_USE_TRAILING_STOP === 'true',
};

function toCentralTime(date: Date): Date {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isTradingAllowed(timestamp: string | Date): boolean {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 6) return false;
  if (day === 0 && minutes < WEEKEND_REOPEN_MINUTES) return false;
  if (day === 5 && minutes >= CUT_OFF_MINUTES) return false;

  return minutes < CUT_OFF_MINUTES || minutes >= REOPEN_MINUTES;
}

function calculateBollingerBands(
  values: number[],
  period: number,
  stdDev: number,
): { upper: number; middle: number; lower: number } | null {
  if (values.length < period) return null;

  const slice = values.slice(-period);
  const sum = slice.reduce((acc, val) => acc + val, 0);
  const mean = sum / period;

  const squaredDiffs = slice.map(val => Math.pow(val - mean, 2));
  const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / period;
  const standardDeviation = Math.sqrt(variance);

  return {
    upper: mean + standardDeviation * stdDev,
    middle: mean,
    lower: mean - standardDeviation * stdDev,
  };
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

// CANDLESTICK PATTERNS & MARKET STRUCTURE
interface CandleStickPattern {
  isPinBar: boolean;
  isHammer: boolean;
  isEngulfing: boolean;
  isInsideBar: boolean;
}

interface MarketStructure {
  hasHigherLow: boolean;
  hasLowerHigh: boolean;
  isOrderBlock: boolean;
}

function isPinBar(current: TopstepXFuturesBar, previous: TopstepXFuturesBar | null): boolean {
  if (!previous) return false;
  const bodySize = Math.abs(current.close - current.open);
  const fullRange = current.high - current.low;
  const tailSize = Math.min(current.high - Math.max(current.close, current.open), Math.min(current.close, current.open) - current.low);

  // Pin bar: body is small, at least one tail is 1.5x the body (relaxed for 1s bars)
  return bodySize > 0 && tailSize > bodySize * 1.5 && fullRange > bodySize * 2;
}

function isHammer(current: TopstepXFuturesBar, previous: TopstepXFuturesBar | null): boolean {
  if (!previous) return false;
  const bodySize = Math.abs(current.close - current.open);
  const lowerTail = Math.min(current.close, current.open) - current.low;
  const upperTail = current.high - Math.max(current.close, current.open);

  // Hammer: lower tail 1.2x body, upper tail small (relaxed for 1s bars)
  return lowerTail > bodySize * 1.2 && upperTail < bodySize * 0.5 && bodySize > 0;
}

function isInvertedHammer(current: TopstepXFuturesBar, previous: TopstepXFuturesBar | null): boolean {
  if (!previous) return false;
  const bodySize = Math.abs(current.close - current.open);
  const lowerTail = Math.min(current.close, current.open) - current.low;
  const upperTail = current.high - Math.max(current.close, current.open);

  // Inverted hammer: upper tail 1.2x body, lower tail small (relaxed for 1s bars)
  return upperTail > bodySize * 1.2 && lowerTail < bodySize * 0.5 && bodySize > 0;
}

function isBullishEngulfing(current: TopstepXFuturesBar, previous: TopstepXFuturesBar): boolean {
  // Current bar is bullish, completely engulfs previous bearish bar
  const currentOpen = Math.min(current.open, current.close);
  const currentClose = Math.max(current.open, current.close);
  const prevOpen = Math.min(previous.open, previous.close);
  const prevClose = Math.max(previous.open, previous.close);

  return (
    current.close > current.open && // Current is bullish
    previous.close < previous.open && // Previous is bearish
    currentOpen < prevOpen &&
    currentClose > prevClose
  );
}

function isBearishEngulfing(current: TopstepXFuturesBar, previous: TopstepXFuturesBar): boolean {
  // Current bar is bearish, completely engulfs previous bullish bar
  const currentOpen = Math.min(current.open, current.close);
  const currentClose = Math.max(current.open, current.close);
  const prevOpen = Math.min(previous.open, previous.close);
  const prevClose = Math.max(previous.open, previous.close);

  return (
    current.close < current.open && // Current is bearish
    previous.close > previous.open && // Previous is bullish
    currentOpen < prevOpen &&
    currentClose > prevClose
  );
}

function isInsideBar(current: TopstepXFuturesBar, previous: TopstepXFuturesBar): boolean {
  // Current bar completely inside previous bar (lower high, higher low)
  return current.high < previous.high && current.low > previous.low;
}

function detectCandlestickPatterns(current: TopstepXFuturesBar, previous: TopstepXFuturesBar | null): CandleStickPattern {
  if (!previous) {
    return { isPinBar: false, isHammer: false, isEngulfing: false, isInsideBar: false };
  }

  return {
    isPinBar: isPinBar(current, previous),
    isHammer: isHammer(current, previous) || isInvertedHammer(current, previous),
    isEngulfing: isBullishEngulfing(current, previous) || isBearishEngulfing(current, previous),
    isInsideBar: isInsideBar(current, previous),
  };
}

// Market Structure Detection
function hasHigherLow(current: TopstepXFuturesBar, recentBars: TopstepXFuturesBar[]): boolean {
  if (recentBars.length < 3) return false;
  // Current low > previous significant low (looking back 5 bars)
  const recentLows = recentBars.slice(-5).map(b => b.low);
  const prevLow = Math.min(...recentLows.slice(0, -1));
  return current.low > prevLow;
}

function hasLowerHigh(current: TopstepXFuturesBar, recentBars: TopstepXFuturesBar[]): boolean {
  if (recentBars.length < 3) return false;
  // Current high < previous significant high (looking back 5 bars)
  const recentHighs = recentBars.slice(-5).map(b => b.high);
  const prevHigh = Math.max(...recentHighs.slice(0, -1));
  return current.high < prevHigh;
}

function isOrderBlock(current: TopstepXFuturesBar, recentBars: TopstepXFuturesBar[]): boolean {
  if (recentBars.length < 5) return false;
  // Order block: recent price rejection zone (strong candle where price reversed)
  const lastThree = recentBars.slice(-3);
  const volatilityLast = lastThree.reduce((sum, bar) => sum + (bar.high - bar.low), 0) / 3;
  const currentBodySize = Math.abs(current.close - current.open);
  const currentRange = current.high - current.low;

  // Strong candle relative to recent bars = potential order block (relaxed for 1s bars)
  // Body size > 50% of avg volatility OR full range > avg volatility
  return currentBodySize > volatilityLast * 0.5 || currentRange > volatilityLast;
}

function detectMarketStructure(
  current: TopstepXFuturesBar,
  recentBars: TopstepXFuturesBar[],
): MarketStructure {
  return {
    hasHigherLow: hasHigherLow(current, recentBars),
    hasLowerHigh: hasLowerHigh(current, recentBars),
    isOrderBlock: isOrderBlock(current, recentBars),
  };
}

function calculateSimpleMovingAverage(values: number[], period: number): number | null {
  if (period <= 0 || values.length < period) return null;
  const slice = values.slice(-period);
  const sum = slice.reduce((acc, val) => acc + val, 0);
  return sum / period;
}

interface FairValueGap {
  bullishFVG: boolean;  // At least 2 gaps down (high[i-2] < low[i]) - supports LONG
  bearishFVG: boolean;  // At least 2 gaps up (low[i-2] > high[i]) - supports SHORT
  bullishCount: number;
  bearishCount: number;
}

function detectFairValueGap(
  bars: TopstepXFuturesBar[],
  currentIndex: number,
  lookback: number = 10,
  minCount: number = 1
): FairValueGap | null {
  // Need at least 3 bars for FVG detection
  if (currentIndex < 2) {
    return null;
  }

  let bullishCount = 0;
  let bearishCount = 0;

  // Look back over the last N bars to count FVGs
  const startIndex = Math.max(2, currentIndex - lookback + 1);
  for (let i = startIndex; i <= currentIndex; i++) {
    const bar = bars[i];
    const barTwoAgo = bars[i - 2];

    // Bullish FVG: high of bar 2 candles ago is below low of current bar
    // This means price gapped down, leaving unfilled space above
    if (barTwoAgo.high < bar.low) {
      bullishCount++;
    }

    // Bearish FVG: low of bar 2 candles ago is above high of current bar
    // This means price gapped up, leaving unfilled space below
    if (barTwoAgo.low > bar.high) {
      bearishCount++;
    }
  }

  return {
    bullishFVG: bullishCount >= minCount,
    bearishFVG: bearishCount >= minCount,
    bullishCount,
    bearishCount,
  };
}

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX MEAN REVERSION BACKTEST (1-SECOND BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`BB Period: ${CONFIG.bbPeriod} bars (${CONFIG.bbPeriod}s) | Std Dev: ${CONFIG.bbStdDev}`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | Oversold: ${CONFIG.rsiOversold} | Overbought: ${CONFIG.rsiOverbought}`);
  console.log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Fast SMA Period: ${CONFIG.fastSmaPeriod}`);
  const adxEffectiveThreshold = CONFIG.adxThreshold - CONFIG.adxBuffer;
  const effectiveAdxThreshold = Math.max(0, CONFIG.adxThreshold - CONFIG.adxBuffer);
  console.log(
    `ADX Filter: period ${CONFIG.adxPeriod} | threshold < ${CONFIG.adxThreshold}` +
    ` | buffer ${CONFIG.adxBuffer} -> effective ${(effectiveAdxThreshold).toFixed(1)} ` +
    `(${CONFIG.bypassAdx ? 'bypassed' : 'active'})`
  );
  console.log(
    `ATR Filter: period ${CONFIG.atrPeriod} | min bar range >= ${(CONFIG.atrMinRangePercent * 100).toFixed(0)}% of ATR`
  );
  console.log(`FVG Filter: ${CONFIG.fvgEnabled ? 'ENABLED (10-bar lookback, min 1 FVG, directional)' : 'DISABLED'}`);
  console.log(`SVP Bias Filter: ${CONFIG.svpBiasFilterEnabled ? 'ENABLED (POC migration bias filtering)' : 'DISABLED'}`);
  console.log(`Commission/side: ${CONFIG.commissionPerSide.toFixed(2)} USD | Slippage: ${CONFIG.slippageTicks} tick(s)`);
  console.log(`Order Types: Entry=MARKET (slip), Stop=MARKET (slip), Scale=LIMIT, Target=LIMIT`);
  console.log('='.repeat(80));

  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey).catch(err => {
    console.warn('[topstepx-mr] Unable to fetch metadata:', err.message);
    return null;
  });

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  const multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || CONFIG.contractMultiplier || 5;
  const tickSize = metadata.tickSize || 0.25;

  // Get base symbol for slippage lookup
  const baseSymbol = getBaseSymbol(CONFIG.symbol);

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize}`);
  console.log(`Spread Model (${baseSymbol}): Avg spread ${SLIP_CONFIG.avgSpreadTicks[baseSymbol]} tick, Fees $${SLIP_CONFIG.feesPerSideUSD[baseSymbol]} per side`);
  console.log(`Slippage (${baseSymbol}): Entry=${SLIP_CONFIG.slipAvg.entry[baseSymbol]} ticks, TP=${SLIP_CONFIG.slipAvg.tp[baseSymbol]} ticks (${(SLIP_CONFIG.p_tp_passive[baseSymbol] * 100).toFixed(0)}% passive), Stop=${SLIP_CONFIG.slipAvg.stop[baseSymbol]} ticks`);

  console.log('\nFetching 1-second bars...');
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: CONFIG.start,
    endTime: CONFIG.end,
    unit: 1,
    unitNumber: 1,
    limit: 50000,
  });

  if (!bars.length) {
    throw new Error('No 1-second bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-second bars`);

  // Calculate SVP session profiles if bias filter enabled
  const sessionProfiles: SessionProfile[] = [];
  const sessionBiasMap = new Map<string, Bias>();

  if (CONFIG.svpBiasFilterEnabled) {
    // RTH detection function
    function isRTH(timestamp: string | Date): boolean {
      const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
      const ctDate = new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
      const day = ctDate.getUTCDay();
      const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();
      if (day === 0 || day === 6) return false;
      return minutes >= 510 && minutes < 900; // 8:30 AM - 3:00 PM CT
    }

    // Separate bars by RTH session date
    const sessionBarsMap = new Map<string, TopstepXFuturesBar[]>();
    for (const bar of bars) {
      if (!isRTH(bar.timestamp)) continue;
      const sessionDate = new Date(bar.timestamp).toISOString().split('T')[0];
      if (!sessionBarsMap.has(sessionDate)) {
        sessionBarsMap.set(sessionDate, []);
      }
      sessionBarsMap.get(sessionDate)!.push(bar);
    }

    const sessionDates = Array.from(sessionBarsMap.keys()).sort();
    console.log(`Found ${sessionDates.length} RTH sessions for SVP calculation`);

    // Calculate session profiles
    for (const date of sessionDates) {
      const sessionBars = sessionBarsMap.get(date)!;
      const profile = calculateSessionProfile(sessionBars, tickSize);
      sessionProfiles.push(profile);
    }

    // Calculate bias for each session
    for (let i = 0; i < sessionProfiles.length; i++) {
      const todayProfile = sessionProfiles[i];
      const yesterdayProfile = i > 0 ? sessionProfiles[i - 1] : null;
      const twoDaysAgoProfile = i > 1 ? sessionProfiles[i - 2] : null;
      const historicalProfiles = sessionProfiles.slice(0, i + 1);

      const biasCalc = calculateBias(todayProfile, yesterdayProfile, twoDaysAgoProfile, historicalProfiles);
      sessionBiasMap.set(todayProfile.date, biasCalc.bias);

      console.log(
        `[${todayProfile.date}] Bias: ${biasCalc.bias.toUpperCase()} ` +
        `(POC drift: ${biasCalc.pocDrift.toFixed(2)}, gate: ${biasCalc.driftGate.toFixed(2)})`
      );
    }
  }

  // Helper function to get bias for a bar's date
  const getBiasForBar = (timestamp: string): Bias => {
    const sessionDate = new Date(timestamp).toISOString().split('T')[0];
    return sessionBiasMap.get(sessionDate) || 'neutral';
  };

  // Helper function to round price to valid tick increments
  const roundToTick = (price: number): number => {
    return Math.round(price / tickSize) * tickSize;
  };

  const closes: number[] = [];
  const adxIndicator = new ADX({
    period: CONFIG.adxPeriod,
    high: [],
    low: [],
    close: [],
  });
  const atrIndicator = new ATR({
    period: CONFIG.atrPeriod,
    high: [],
    low: [],
    close: [],
  });
  const trades: TradeRecord[] = [];
  let filteredByAtr = 0;
  let filteredByRsiDivergence = 0;
  let filteredByFvg = 0;
  let filteredByVolume = 0;
  let filteredBySvpBias = 0;

  let pendingSetup: {
    side: 'long' | 'short';
    setupTime: string;
    setupPrice: number;
    rsi: number;
    adx: number | null;
    bb: { upper: number; middle: number; lower: number };
  } | null = null;

  let position: {
    side: 'long' | 'short';
    entryPrice: number;
    entryTime: string;
    entryRSI: number;
    entryADX: number | null;
    stopLoss: number | null;
    target: number | null;
    scaled: boolean;
    remainingQty: number;
    scalePnL: number;
    feesPaid: number;
    fastScaleFilled: boolean;
    fastScaleTarget: number | null;
    bias?: Bias;
    highestPrice: number;
    lowestPrice: number;
  } | null = null;

  let realizedPnL = 0;

  // Track ADX values for analysis
  const adxValues: number[] = [];
  const entryAdxValues: number[] = [];

  // Track volume for filtering
  const volumes: number[] = [];

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
    exitSlippagePoints: number = 0,
  ) => {
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * position.remainingQty;
    const exitFees = addFees(baseSymbol, position.remainingQty);
    const totalFees = position.feesPaid + exitFees;
    const grossPnl = rawPnL + position.scalePnL;
    const netPnl = grossPnl - totalFees;

    // Calculate slippage cost (entry + exit slippage in dollars)
    const entrySlippagePoints = (SLIP_CONFIG.slipAvg.entry[baseSymbol] + 0.5 * SLIP_CONFIG.avgSpreadTicks[baseSymbol]) * tickSize;
    const totalSlippagePoints = entrySlippagePoints + exitSlippagePoints;
    const slippageCost = totalSlippagePoints * multiplier * position.remainingQty;
    const entrySlippageTicks = entrySlippagePoints / tickSize;
    const exitSlippageTicks = exitSlippagePoints / tickSize;

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
      entryRSI: position.entryRSI,
      entryADX: position.entryADX ?? undefined,
      scaled: position.scaled,
      scalePnL: position.scalePnL,
      finalPnL: rawPnL,
      entrySlippageTicks: entrySlippageTicks,
      exitSlippageTicks: exitSlippageTicks,
      slippageCost: slippageCost,
      bias: position.bias,
    };

    trades.push(trade);
    realizedPnL += netPnl;
    position = null;
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    closes.push(bar.close);
    volumes.push(bar.volume ?? 0);
    const adxResult = adxIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    } as any);
    const currentADX = typeof adxResult === 'number'
      ? adxResult
      : (adxResult?.adx ?? null);

    // Track ADX values for statistics
    if (currentADX !== null && closes.length >= CONFIG.bbPeriod) {
      adxValues.push(currentADX);
    }

    const atrResult = atrIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });
    const currentATR = typeof atrResult === 'number' ? atrResult : null;

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        // Session exit uses stop slippage (emergency exit like stop)
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(baseSymbol, closeSide, bar.close));
        const stopSlippagePoints = (SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize);
        exitPosition(sessionExitPrice, bar.timestamp, 'session', stopSlippagePoints);
      }
      continue;
    }

    if (closes.length < CONFIG.bbPeriod) {
      continue;
    }

    const bb = calculateBollingerBands(closes, CONFIG.bbPeriod, CONFIG.bbStdDev);
    if (!bb) continue;

    const bbWidth = bb.upper - bb.lower;

    const rsiValues = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });
    const currentRSI = rsiValues[rsiValues.length - 1];
    if (currentRSI === undefined) continue;

    // Calculate MACD (28, 104, 9) for crossover confirmation
    const macdValues = MACD.calculate({
      values: closes,
      fastPeriod: 28,
      slowPeriod: 104,
      signalPeriod: 9,
      SimpleMAOscillator: false,
      SimpleMASignal: false,
    });
    const currentMACD = macdValues.length >= 2 ? macdValues[macdValues.length - 1] : null;
    const prevMACD = macdValues.length >= 2 ? macdValues[macdValues.length - 2] : null;

    const fastSma = calculateSimpleMovingAverage(closes, CONFIG.fastSmaPeriod);

    const ttmBars = bars.slice(Math.max(0, i - 20), i + 1);
    const ttmSqueeze = calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });
    if (!ttmSqueeze) continue;

    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // Update highest/lowest price for trailing stop
      if (CONFIG.useTrailingStop) {
        position.highestPrice = Math.max(position.highestPrice, bar.high);
        position.lowestPrice = Math.min(position.lowestPrice, bar.low);

        // Update trailing stop
        if (position.side === 'long') {
          const trailingStopPrice = position.highestPrice * (1 - CONFIG.trailingStopPercent);
          position.stopLoss = Math.max(position.stopLoss || 0, roundToTick(trailingStopPrice));
        } else {
          const trailingStopPrice = position.lowestPrice * (1 + CONFIG.trailingStopPercent);
          position.stopLoss = Math.min(position.stopLoss || Infinity, roundToTick(trailingStopPrice));
        }
      }

      // Middle Band Exit - EXIT ALL CONTRACTS
      // Best strategy for 1-second bars: take profit quickly at middle band
      if (!position.scaled && position.target !== null) {
        const hitTarget = (direction === 1 && bar.high >= position.target) ||
                         (direction === -1 && bar.low <= position.target);

        if (hitTarget) {
          // EXIT ALL CONTRACTS at middle band
          const basePrice = direction === 1
            ? Math.min(bar.high, position.target)
            : Math.max(bar.low, position.target);
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const exitPrice = roundToTick(fillTP(baseSymbol, closeSide, basePrice));

          console.log(
            `[${bar.timestamp}] Target hit at middle band ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
            `Exit all ${CONFIG.numberOfContracts} @ ${exitPrice.toFixed(2)}`
          );

          // TP slippage cost
          const S_ticks = SLIP_CONFIG.avgSpreadTicks[baseSymbol];
          const sigma_tp_ticks = SLIP_CONFIG.slipAvg.tp[baseSymbol];
          const p_passive = SLIP_CONFIG.p_tp_passive[baseSymbol];
          const E_tp_ticks = (1 - p_passive) * (S_ticks + sigma_tp_ticks);
          const tpSlippagePoints = E_tp_ticks * tickSize;
          exitPosition(exitPrice, bar.timestamp, 'target', tpSlippagePoints);

          continue;
        }
      }

      if (position.stopLoss !== null) {
        const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                       (direction === -1 && bar.high >= position.stopLoss);
        if (hitStop) {
          // Stop loss: use fillStop() for realistic adverse fill
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const stopExitPrice = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss));
          const stopSlippagePoints = SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize;
          exitPosition(stopExitPrice, bar.timestamp, 'stop', stopSlippagePoints);
          continue;
        }
      }


      continue;
    }

    if (!CONFIG.bypassAdx) {
      if (currentADX === null) {
        pendingSetup = null;
        continue;
      }
      if (currentADX >= effectiveAdxThreshold) {
        if (pendingSetup) {
          pendingSetup = null;
        }
        continue;
      }
    }

    const price = bar.close;
    const distanceToUpper = bb.upper - price;
    const distanceToLower = price - bb.lower;

    // ATR noise filter: check if bar range is >= 25% of ATR
    const barRange = bar.high - bar.low;
    const atrFilterPassed = currentATR === null || barRange >= (currentATR * CONFIG.atrMinRangePercent);

    // Fair Value Gap filter
    const fvg = detectFairValueGap(bars, i);
    const fvgFilterPassed = !CONFIG.fvgEnabled || fvg === null ||
      (fvg.bullishFVG || fvg.bearishFVG);  // Allow if any FVG exists when enabled

    // Apply FVG directional filter for long/short
    const longFvgOk = !CONFIG.fvgEnabled || fvg === null || fvg.bullishFVG;
    const shortFvgOk = !CONFIG.fvgEnabled || fvg === null || fvg.bearishFVG;

    // Volume filter: Current volume must be > 1.5x average
    const volumePeriod = Math.min(200, volumes.length);
    const avgVolume = volumePeriod > 0
      ? volumes.slice(-volumePeriod).reduce((sum, v) => sum + v, 0) / volumePeriod
      : 0;
    const currentVolume = bar.volume ?? 0;
    const volumeFilterPassed = avgVolume === 0 || currentVolume > avgVolume * 1.5;

    // Candlestick Pattern & Market Structure Confirmations
    const previousBar = i > 0 ? bars[i - 1] : null;
    const recentBars = bars.slice(Math.max(0, i - 10), i + 1);
    const patterns = detectCandlestickPatterns(bar, previousBar);
    const structure = detectMarketStructure(bar, recentBars);

    // Pin bar, hammer, or engulfing for reversal confirmation
    const hasReversalPattern = patterns.isPinBar || patterns.isHammer || patterns.isEngulfing;

    // Market structure confirmation: higher low for long, lower high for short
    const longStructureOk = structure.hasHigherLow && structure.isOrderBlock;
    const shortStructureOk = structure.hasLowerHigh && structure.isOrderBlock;

    // Core setup: BB touch + RSI extreme (band touch is required filter) + ATR + FVG + Volume
    // Candlestick patterns and market structure are optional confirmations (logged but not required)
    const longSetupDetected = bar.low <= bb.lower && currentRSI < CONFIG.rsiOversold && atrFilterPassed && longFvgOk && volumeFilterPassed;
    const shortSetupDetected = bar.high >= bb.upper && currentRSI > CONFIG.rsiOverbought && atrFilterPassed && shortFvgOk && volumeFilterPassed;

    // Track quality confirmations for logging
    const longQualityConfirms = (hasReversalPattern ? 1 : 0) + (longStructureOk ? 1 : 0);
    const shortQualityConfirms = (hasReversalPattern ? 1 : 0) + (shortStructureOk ? 1 : 0);

    // Track filtered setups
    if (!atrFilterPassed && (
      (bar.low <= bb.lower && currentRSI < CONFIG.rsiOversold) ||
      (bar.high >= bb.upper && currentRSI > CONFIG.rsiOverbought)
    )) {
      filteredByAtr++;
    }

    // Track FVG filtered setups
    if (CONFIG.fvgEnabled && fvg !== null && atrFilterPassed) {
      if (bar.low <= bb.lower && currentRSI < CONFIG.rsiOversold && !fvg.bullishFVG) {
        filteredByFvg++;
      }
      if (bar.high >= bb.upper && currentRSI > CONFIG.rsiOverbought && !fvg.bearishFVG) {
        filteredByFvg++;
      }
    }

    if (!pendingSetup && longSetupDetected) {
      pendingSetup = {
        side: 'long',
        setupTime: bar.timestamp,
        setupPrice: bar.close,
        rsi: currentRSI,
        adx: currentADX,
        bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
      };
      const patternNames = [
        patterns.isPinBar ? 'PinBar' : '',
        patterns.isHammer ? 'Hammer' : '',
        patterns.isEngulfing ? 'Engulfing' : '',
      ].filter(p => p).join('+');

      console.log(
        `[${bar.timestamp}] LONG setup detected ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
        `(RSI ${currentRSI.toFixed(1)}, ADX ${currentADX?.toFixed(1) ?? 'N/A'}, Quality Confirms: ${longQualityConfirms}/2 [${patternNames}${patternNames && longStructureOk ? ', ' : ''}${longStructureOk ? 'HigherLow+OB' : ''}], awaiting TTM Squeeze)`
      );
    } else if (!pendingSetup && shortSetupDetected) {
      const patternNames = [
        patterns.isPinBar ? 'PinBar' : '',
        patterns.isHammer ? 'Hammer' : '',
        patterns.isEngulfing ? 'Engulfing' : '',
      ].filter(p => p).join('+');

      pendingSetup = {
        side: 'short',
        setupTime: bar.timestamp,
        setupPrice: bar.close,
        rsi: currentRSI,
        adx: currentADX,
        bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
      };
      console.log(
        `[${bar.timestamp}] SHORT setup detected ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
        `(RSI ${currentRSI.toFixed(1)}, ADX ${currentADX?.toFixed(1) ?? 'N/A'}, Quality Confirms: ${shortQualityConfirms}/2 [${patternNames}${patternNames && shortStructureOk ? ', ' : ''}${shortStructureOk ? 'LowerHigh+OB' : ''}], awaiting TTM Squeeze)`
      );
    }

    // RSI divergence tracking DISABLED - not used for entry filter

    // Check for TTM Squeeze trigger
    if (pendingSetup && ttmSqueeze.squeezeOn) {

      // SVP Bias Filter: Check if setup aligns with daily bias
      if (CONFIG.svpBiasFilterEnabled) {
        const currentBias = getBiasForBar(bar.timestamp);

        // Filter misaligned setups:
        // - Bull bias: reject shorts
        // - Bear bias: reject longs
        // - Neutral: allow both
        const setupMisaligned =
          (currentBias === 'bull' && pendingSetup.side === 'short') ||
          (currentBias === 'bear' && pendingSetup.side === 'long');

        if (setupMisaligned) {
          filteredBySvpBias++;
          pendingSetup = null;
          continue;
        }
      }

      // Entry allowed on TTM Squeeze
      // Use fillEntry() for realistic entry fill (crossing the spread + slippage)
      const entrySide = pendingSetup.side === 'long' ? 'buy' : 'sell';
      const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, bar.close));

      const entryAdx = pendingSetup.adx ?? currentADX ?? null;
      const fastSmaForEntry = fastSma;

      // Track ADX at entry
      if (entryAdx !== null) {
        entryAdxValues.push(entryAdx);
      }

      position = {
        side: pendingSetup.side,
        entryPrice,
        entryTime: bar.timestamp,
        entryRSI: pendingSetup.rsi,
        entryADX: entryAdx,
        stopLoss: roundToTick(pendingSetup.side === 'long'
          ? pendingSetup.bb.lower * (1 - CONFIG.stopLossPercent)
          : pendingSetup.bb.upper * (1 + CONFIG.stopLossPercent)),
        target: roundToTick(pendingSetup.bb.middle),
        scaled: false,
        remainingQty: CONFIG.numberOfContracts,
        scalePnL: 0,
        feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
        fastScaleFilled: true, // DISABLED: Skip 50 SMA logic
        fastScaleTarget: null, // DISABLED: No 50 SMA target
        bias: CONFIG.svpBiasFilterEnabled ? getBiasForBar(bar.timestamp) : undefined,
        highestPrice: bar.high,
        lowestPrice: bar.low,
      };

      const entryAdxText = entryAdx !== null ? entryAdx.toFixed(1) : 'N/A';
      const fastSmaText = fastSmaForEntry !== null ? fastSmaForEntry.toFixed(2) : 'N/A';

      console.log(
        `[${bar.timestamp}] ${pendingSetup.side.toUpperCase()} entry ${CONFIG.symbol} @ ${entryPrice.toFixed(2)} ` +
        `(TTM Squeeze trigger, setup @ ${pendingSetup.setupPrice.toFixed(2)}, ` +
        `RSI ${pendingSetup.rsi.toFixed(1)}, ADX ${entryAdxText}, Fast SMA ${fastSmaText})`
      );

      pendingSetup = null;
    }
  }

  if (position) {
    const lastBar = bars[bars.length - 1];
    // End-of-data exit: use stop slippage (market exit)
    const closeSide = position.side === 'long' ? 'sell' : 'buy';
    const endExitPrice = roundToTick(fillStop(baseSymbol, closeSide, lastBar.close));
    const stopSlippagePoints = SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize;
    exitPosition(endExitPrice, lastBar.timestamp, 'end_of_data', stopSlippagePoints);
  }

  const winningTrades = trades.filter(t => t.pnl > 0);
  const losingTrades = trades.filter(t => t.pnl <= 0);
  const winRate = trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0;

  const avgWin = winningTrades.length
    ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length
    : 0;
  const avgLoss = losingTrades.length
    ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length)
    : 0;

  const grossProfit = trades
    .filter(t => t.grossPnl > 0)
    .reduce((sum, t) => sum + t.grossPnl, 0);
  const grossLoss = Math.abs(
    trades
      .filter(t => t.grossPnl <= 0)
      .reduce((sum, t) => sum + t.grossPnl, 0),
  );
  const totalFees = trades.reduce((sum, t) => sum + t.fees, 0);
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

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

  const scaledTrades = trades.filter(t => t.scaled);
  const avgScalePnL = scaledTrades.length
    ? scaledTrades.reduce((sum, t) => sum + (t.scalePnL || 0), 0) / scaledTrades.length
    : 0;

  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST SUMMARY');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${trades.length} | Wins: ${winningTrades.length} | Losses: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net Realized PnL: ${formatCurrency(realizedPnL)} USD | Fees Paid: $${totalFees.toFixed(2)}`);
  console.log(`Gross Profit (pre-fees): ${formatCurrency(grossProfit)} | Gross Loss: ${formatCurrency(grossLoss)}`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);
  console.log(`Scaled Trades: ${scaledTrades.length}/${trades.length} (${((scaledTrades.length / Math.max(1, trades.length)) * 100).toFixed(1)}%)`);
  console.log(`Avg Scale PnL: ${formatCurrency(avgScalePnL)}`);

  // Slippage Statistics
  if (trades.length > 0) {
    const totalSlippageCost = trades.reduce((sum, t) => sum + (t.slippageCost || 0), 0);
    const avgSlippageCost = totalSlippageCost / trades.length;
    const avgEntrySlippageTicks = trades.reduce((sum, t) => sum + (t.entrySlippageTicks || 0), 0) / trades.length;
    const avgExitSlippageTicks = trades.reduce((sum, t) => sum + (t.exitSlippageTicks || 0), 0) / trades.length;
    const avgTotalSlippageTicks = avgEntrySlippageTicks + avgExitSlippageTicks;

    console.log(`\nSlippage Impact (${baseSymbol}):`);
    console.log(`  Total Slippage Cost: ${formatCurrency(totalSlippageCost)} (${((totalSlippageCost / Math.abs(realizedPnL + totalSlippageCost)) * 100).toFixed(1)}% of gross PnL)`);
    console.log(`  Avg Per Trade: ${formatCurrency(avgSlippageCost)} | ${avgTotalSlippageTicks.toFixed(2)} ticks (Entry: ${avgEntrySlippageTicks.toFixed(2)}, Exit: ${avgExitSlippageTicks.toFixed(2)})`);
  }

  // Exit reason breakdown
  const exitReasons = trades.reduce((acc, t) => {
    acc[t.exitReason] = (acc[t.exitReason] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`\nExit Reasons: ${Object.entries(exitReasons).map(([r, c]) => `${r}=${c}`).join(', ')}`);
  console.log(`Setups Filtered by ATR: ${filteredByAtr} (bar range < ${(CONFIG.atrMinRangePercent * 100).toFixed(0)}% of ATR)`);
  console.log(`Setups Filtered by FVG: ${filteredByFvg} (no matching Fair Value Gap)`);
  console.log(`Setups Filtered by RSI Divergence: ${filteredByRsiDivergence} (TTM fired but no RSI higher-low/lower-high)`);
  console.log(`Setups Filtered by Volume: ${filteredByVolume} (insufficient directional volume or no volume increase)`);
  if (CONFIG.svpBiasFilterEnabled) {
    console.log(`Setups Filtered by SVP Bias: ${filteredBySvpBias} (setup misaligned with daily POC migration bias)`);

    // Bias breakdown
    const biasCounts = trades.reduce((acc, t) => {
      const bias = t.bias || 'unknown';
      acc[bias] = (acc[bias] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    console.log(`Trades by Bias: ${Object.entries(biasCounts).map(([b, c]) => `${b}=${c}`).join(', ')}`);
  }

  // ADX Statistics
  if (adxValues.length > 0) {
    const sortedAdx = adxValues.slice().sort((a, b) => a - b);
    const minAdx = sortedAdx[0];
    const maxAdx = sortedAdx[sortedAdx.length - 1];
    const avgAdx = adxValues.reduce((sum, val) => sum + val, 0) / adxValues.length;
    const medianAdx = sortedAdx[Math.floor(sortedAdx.length / 2)];
    const p25Adx = sortedAdx[Math.floor(sortedAdx.length * 0.25)];
    const p75Adx = sortedAdx[Math.floor(sortedAdx.length * 0.75)];
    const p90Adx = sortedAdx[Math.floor(sortedAdx.length * 0.90)];
    const p95Adx = sortedAdx[Math.floor(sortedAdx.length * 0.95)];

    console.log(`\nADX(${CONFIG.adxPeriod}) Statistics (${adxValues.length} samples):`);
    console.log(`  Min: ${minAdx.toFixed(2)} | Max: ${maxAdx.toFixed(2)} | Avg: ${avgAdx.toFixed(2)} | Median: ${medianAdx.toFixed(2)}`);
    console.log(`  P25: ${p25Adx.toFixed(2)} | P75: ${p75Adx.toFixed(2)} | P90: ${p90Adx.toFixed(2)} | P95: ${p95Adx.toFixed(2)}`);
    console.log(`  Below 10: ${sortedAdx.filter(v => v < 10).length} (${((sortedAdx.filter(v => v < 10).length / adxValues.length) * 100).toFixed(1)}%)`);
    console.log(`  Below 15: ${sortedAdx.filter(v => v < 15).length} (${((sortedAdx.filter(v => v < 15).length / adxValues.length) * 100).toFixed(1)}%)`);
    console.log(`  Below 20: ${sortedAdx.filter(v => v < 20).length} (${((sortedAdx.filter(v => v < 20).length / adxValues.length) * 100).toFixed(1)}%)`);
    console.log(`  Below 25: ${sortedAdx.filter(v => v < 25).length} (${((sortedAdx.filter(v => v < 25).length / adxValues.length) * 100).toFixed(1)}%)`);
  }

  if (entryAdxValues.length > 0) {
    const sortedEntryAdx = entryAdxValues.slice().sort((a, b) => a - b);
    const minEntryAdx = sortedEntryAdx[0];
    const maxEntryAdx = sortedEntryAdx[sortedEntryAdx.length - 1];
    const avgEntryAdx = entryAdxValues.reduce((sum, val) => sum + val, 0) / entryAdxValues.length;
    const medianEntryAdx = sortedEntryAdx[Math.floor(sortedEntryAdx.length / 2)];

    console.log(`\nADX at Trade Entries (${entryAdxValues.length} trades):`);
    console.log(`  Min: ${minEntryAdx.toFixed(2)} | Max: ${maxEntryAdx.toFixed(2)} | Avg: ${avgEntryAdx.toFixed(2)} | Median: ${medianEntryAdx.toFixed(2)}`);
  }

  if (trades.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('RECENT TRADES');
    console.log('='.repeat(80));
    trades.slice(-10).forEach(trade => {
      const scaleInfo = trade.scaled
        ? ` [SCALED: ${formatCurrency(trade.scalePnL || 0)} + ${formatCurrency(trade.finalPnL || 0)}]`
        : '';
      console.log(
        `${trade.side.toUpperCase().padEnd(5)} ${trade.entryTime} @ ${trade.entryPrice.toFixed(2)} -> ` +
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl).padStart(10)} (${trade.exitReason})${scaleInfo}`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('TopstepX mean reversion backtest failed:', err);
  process.exit(1);
});
