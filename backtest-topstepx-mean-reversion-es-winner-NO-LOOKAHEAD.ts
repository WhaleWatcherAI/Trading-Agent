#!/usr/bin/env tsx
/**
 * TopstepX Mean Reversion Backtest - FIXED NO LOOK-AHEAD BIAS
 *
 * KEY FIXES:
 * 1. Indicators calculated EXCLUDING current bar's close
 * 2. Entries occur at NEXT bar's open (not current bar's close)
 * 3. Signal detection uses PREVIOUS bar's completed data only
 *
 * This version properly simulates real-time trading conditions.
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
const DEFAULT_DAYS = 365;

const DEFAULT_MR_SYMBOL = process.env.TOPSTEPX_MR_SYMBOL || 'ESZ5';
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
  stopLossPercent: Number(process.env.TOPSTEPX_MR_STOP_LOSS_PERCENT || '0.0001'),
  takeProfitPercent: Number(process.env.TOPSTEPX_MR_TAKE_PROFIT_PERCENT || '0.0004'),
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

async function runBacktest() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX MEAN REVERSION BACKTEST - NO LOOK-AHEAD BIAS (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Period: ${CONFIG.start} -> ${CONFIG.end}`);
  console.log(`BB Period: ${CONFIG.bbPeriod} bars | Std Dev: ${CONFIG.bbStdDev}`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | Oversold: ${CONFIG.rsiOversold} | Overbought: ${CONFIG.rsiOverbought}`);
  console.log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`⚠️  FIXED: Entry at NEXT bar's OPEN (no look-ahead bias)`);
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

  const baseSymbol = getBaseSymbol(CONFIG.symbol);

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize}`);
  console.log(`Spread Model (${baseSymbol}): Avg spread ${SLIP_CONFIG.avgSpreadTicks[baseSymbol]} tick`);

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
    throw new Error('No 1-minute bars returned for configured window.');
  }

  bars.reverse();
  console.log(`Loaded ${bars.length} one-minute bars`);

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
  const volumes: number[] = [];

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
    highestPrice: number;
    lowestPrice: number;
  } | null = null;

  let realizedPnL = 0;

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
    };

    trades.push(trade);
    realizedPnL += netPnl;
    position = null;
  };

  // MAIN LOOP - FIXED FOR NO LOOK-AHEAD BIAS
  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];

    // Handle position management first (uses current bar OHLC)
    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // Check stop loss
      if (position.stopLoss !== null) {
        const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                       (direction === -1 && bar.high >= position.stopLoss);
        if (hitStop) {
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const stopExitPrice = roundToTick(fillStop(baseSymbol, closeSide, position.stopLoss));
          const stopSlippagePoints = SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize;
          exitPosition(stopExitPrice, bar.timestamp, 'stop', stopSlippagePoints);
          closes.push(bar.close);
          volumes.push(bar.volume ?? 0);
          adxIndicator.nextValue({ high: bar.high, low: bar.low, close: bar.close } as any);
          atrIndicator.nextValue({ high: bar.high, low: bar.low, close: bar.close });
          continue;
        }
      }

      // Check take profit
      if (position.target !== null) {
        const hitTarget = (direction === 1 && bar.high >= position.target) ||
                         (direction === -1 && bar.low <= position.target);
        if (hitTarget) {
          const basePrice = direction === 1
            ? Math.min(bar.high, position.target)
            : Math.max(bar.low, position.target);
          const closeSide = position.side === 'long' ? 'sell' : 'buy';
          const exitPrice = roundToTick(fillTP(baseSymbol, closeSide, basePrice));

          const S_ticks = SLIP_CONFIG.avgSpreadTicks[baseSymbol];
          const sigma_tp_ticks = SLIP_CONFIG.slipAvg.tp[baseSymbol];
          const p_passive = SLIP_CONFIG.p_tp_passive[baseSymbol];
          const E_tp_ticks = (1 - p_passive) * (S_ticks + sigma_tp_ticks);
          const tpSlippagePoints = E_tp_ticks * tickSize;

          exitPosition(exitPrice, bar.timestamp, 'target', tpSlippagePoints);
          closes.push(bar.close);
          volumes.push(bar.volume ?? 0);
          adxIndicator.nextValue({ high: bar.high, low: bar.low, close: bar.close } as any);
          atrIndicator.nextValue({ high: bar.high, low: bar.low, close: bar.close });
          continue;
        }
      }
    }

    // No immediate entry here - wait for TTM trigger below

    // Add current bar to history AFTER checking entries
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

    const atrResult = atrIndicator.nextValue({
      high: bar.high,
      low: bar.low,
      close: bar.close,
    });
    const currentATR = typeof atrResult === 'number' ? atrResult : null;

    if (!isTradingAllowed(bar.timestamp)) {
      if (position) {
        const closeSide = position.side === 'long' ? 'sell' : 'buy';
        const sessionExitPrice = roundToTick(fillStop(baseSymbol, closeSide, bar.close));
        const stopSlippagePoints = (SLIP_CONFIG.slipAvg.stop[baseSymbol] * tickSize);
        exitPosition(sessionExitPrice, bar.timestamp, 'session', stopSlippagePoints);
      }
      pendingSetup = null;
      continue;
    }

    if (closes.length < CONFIG.bbPeriod + 1) {
      continue;
    }

    // CRITICAL FIX: Calculate indicators on PREVIOUS bars only (exclude current)
    const prevCloses = closes.slice(0, -1);
    const bb = calculateBollingerBands(prevCloses, CONFIG.bbPeriod, CONFIG.bbStdDev);
    if (!bb) continue;

    const rsiValues = RSI.calculate({ values: prevCloses, period: CONFIG.rsiPeriod });
    const currentRSI = rsiValues[rsiValues.length - 1];
    if (currentRSI === undefined) continue;

    // Skip if already in position
    if (position) {
      continue;
    }

    // ADX filter
    if (!CONFIG.bypassAdx) {
      if (currentADX === null) {
        pendingSetup = null;
        continue;
      }
      const effectiveAdxThreshold = Math.max(0, CONFIG.adxThreshold - CONFIG.adxBuffer);
      if (currentADX >= effectiveAdxThreshold) {
        pendingSetup = null;
        continue;
      }
    }

    // Get PREVIOUS bar for comparison
    const prevBar = i > 0 ? bars[i - 1] : null;
    if (!prevBar) continue;

    // ATR filter using previous bar
    const barRange = prevBar.high - prevBar.low;
    const atrFilterPassed = currentATR === null || barRange >= (currentATR * CONFIG.atrMinRangePercent);

    // Volume filter
    const volumePeriod = Math.min(200, volumes.length - 1);
    const prevVolumes = volumes.slice(0, -1);
    const avgVolume = volumePeriod > 0
      ? prevVolumes.slice(-volumePeriod).reduce((sum, v) => sum + v, 0) / volumePeriod
      : 0;
    const prevVolume = prevBar.volume ?? 0;
    const volumeFilterPassed = avgVolume === 0 || prevVolume > avgVolume * 1.5;

    // STAGE 1: Setup detection using PREVIOUS bar's price action
    if (!pendingSetup) {
      const longSetupDetected = prevBar.low <= bb.lower && currentRSI < CONFIG.rsiOversold && atrFilterPassed && volumeFilterPassed;
      const shortSetupDetected = prevBar.high >= bb.upper && currentRSI > CONFIG.rsiOverbought && atrFilterPassed && volumeFilterPassed;

      if (longSetupDetected) {
        pendingSetup = {
          side: 'long',
          setupTime: prevBar.timestamp,
          setupPrice: prevBar.close,
          rsi: currentRSI,
          adx: currentADX,
          bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
        };
        console.log(
          `[${prevBar.timestamp}] LONG setup detected @ ${prevBar.close.toFixed(2)} ` +
          `(RSI ${currentRSI.toFixed(1)}) - awaiting TTM Squeeze`
        );
      } else if (shortSetupDetected) {
        pendingSetup = {
          side: 'short',
          setupTime: prevBar.timestamp,
          setupPrice: prevBar.close,
          rsi: currentRSI,
          adx: currentADX,
          bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
        };
        console.log(
          `[${prevBar.timestamp}] SHORT setup detected @ ${prevBar.close.toFixed(2)} ` +
          `(RSI ${currentRSI.toFixed(1)}) - awaiting TTM Squeeze`
        );
      }
    }

    // STAGE 2: Check for TTM Squeeze trigger on CURRENT bar
    const ttmBars = bars.slice(Math.max(0, i - 20), i); // Exclude current bar
    const ttmSqueeze = calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });

    if (pendingSetup && ttmSqueeze && ttmSqueeze.squeezeOn) {
      // CRITICAL FIX: Enter at NEXT bar's open
      // We need to peek ahead to get next bar's open
      if (i + 1 < bars.length) {
        const nextBar = bars[i + 1];
        const entrySide = pendingSetup.side === 'long' ? 'buy' : 'sell';
        const entryPrice = roundToTick(fillEntry(baseSymbol, entrySide, nextBar.open));

        position = {
          side: pendingSetup.side,
          entryPrice,
          entryTime: nextBar.timestamp,
          entryRSI: pendingSetup.rsi,
          entryADX: pendingSetup.adx,
          stopLoss: roundToTick(pendingSetup.side === 'long'
            ? pendingSetup.bb.lower * (1 - CONFIG.stopLossPercent)
            : pendingSetup.bb.upper * (1 + CONFIG.stopLossPercent)),
          target: roundToTick(pendingSetup.side === 'long'
            ? entryPrice * (1 + CONFIG.takeProfitPercent)
            : entryPrice * (1 - CONFIG.takeProfitPercent)),
          scaled: false,
          remainingQty: CONFIG.numberOfContracts,
          scalePnL: 0,
          feesPaid: addFees(baseSymbol, CONFIG.numberOfContracts),
          fastScaleFilled: true,
          fastScaleTarget: null,
          highestPrice: nextBar.high,
          lowestPrice: nextBar.low,
        };

        console.log(
          `[${bar.timestamp}] TTM Squeeze triggered - ${pendingSetup.side.toUpperCase()} entry @ next bar OPEN ${entryPrice.toFixed(2)} ` +
          `(setup @ ${pendingSetup.setupPrice.toFixed(2)}, RSI ${pendingSetup.rsi.toFixed(1)})`
        );

        pendingSetup = null;

        // Skip next bar since we've already processed it for entry
        i++;
        closes.push(nextBar.close);
        volumes.push(nextBar.volume ?? 0);
        adxIndicator.nextValue({ high: nextBar.high, low: nextBar.low, close: nextBar.close } as any);
        atrIndicator.nextValue({ high: nextBar.high, low: nextBar.low, close: nextBar.close });
      }
    }
  }

  // Close any remaining position
  if (position) {
    const lastBar = bars[bars.length - 1];
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

  console.log('\n' + '='.repeat(80));
  console.log('BACKTEST SUMMARY - NO LOOK-AHEAD BIAS');
  console.log('='.repeat(80));
  console.log(`Total Trades: ${trades.length} | Wins: ${winningTrades.length} | Losses: ${losingTrades.length}`);
  console.log(`Win Rate: ${winRate.toFixed(1)}%`);
  console.log(`Net Realized PnL: ${formatCurrency(realizedPnL)} USD | Fees Paid: $${totalFees.toFixed(2)}`);
  console.log(`Gross Profit (pre-fees): ${formatCurrency(grossProfit)} | Gross Loss: ${formatCurrency(grossLoss)}`);
  console.log(`Avg Win: ${formatCurrency(avgWin)} | Avg Loss: ${formatCurrency(avgLoss)}`);
  console.log(`Profit Factor: ${profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}`);
  console.log(`Max Drawdown: ${formatCurrency(maxDrawdown)} USD`);

  if (trades.length > 0) {
    const totalSlippageCost = trades.reduce((sum, t) => sum + (t.slippageCost || 0), 0);
    const avgSlippageCost = totalSlippageCost / trades.length;
    console.log(`\nSlippage Impact: Total ${formatCurrency(totalSlippageCost)} | Avg ${formatCurrency(avgSlippageCost)} per trade`);
  }

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
        `${trade.exitTime} @ ${trade.exitPrice.toFixed(2)} | ${formatCurrency(trade.pnl).padStart(10)} (${trade.exitReason})`
      );
    });
  }
}

runBacktest().catch(err => {
  console.error('TopstepX mean reversion backtest failed:', err);
  process.exit(1);
});
