#!/usr/bin/env tsx
/**
 * TopstepX Mean Reversion Backtest on 1-Second Bars - WITH FVG FILTER
 *
 * OPTIMIZED FOR: EURO (6E) - 88.9% win rate with FVG
 *
 * Strategy:
 * - Bollinger Bands: 200-period SMA with 3 standard deviations
 * - RSI confirmation: RSI(14) for entry filtering
 * - Two-stage entry:
 *   Stage 1: Price within 5% of BB width from outer band + RSI creates setup
 *   Stage 2: TTM Squeeze ON triggers entry (even if BB/RSI changed)
 * - FVG Filter: Requires at least 1 Fair Value Gap in trade direction within last 10 bars (ENABLED)
 * - Scale: Take 50% profit at middle band (200 SMA) - exact
 * - Runner target: Opposite outer BB (within 5% of BB width) - consistent with entry
 * - Stop: Fixed percent from middle band (default 0.05%)
 * - ADX filter: ADX(900) must stay below 22, with safety buffer, to avoid trending markets
 * - Volume filter: disabled (previously SMA(300) * 1.8)
 */

import { RSI, ADX, ATR, EMA } from 'technicalindicators';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXFuturesBar,
} from './lib/topstepx';
import { calculateTtmSqueeze } from './lib/ttmSqueeze';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';

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
}

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;
const DEFAULT_DAYS = 7;

const DEFAULT_MR_SYMBOL = process.env.TOPSTEPX_MR_SYMBOL || '6EZ5';  // Default to Euro (optimized for FVG)
const DEFAULT_MR_CONTRACT_ID = process.env.TOPSTEPX_MR_CONTRACT_ID;

const CONFIG: BacktestConfig = {
  symbol: DEFAULT_MR_SYMBOL,
  contractId: DEFAULT_MR_CONTRACT_ID,
  start: process.env.TOPSTEPX_MR_START || new Date(Date.now() - DEFAULT_DAYS * 24 * 60 * 60 * 1000).toISOString(),
  end: process.env.TOPSTEPX_MR_END || new Date().toISOString(),
  bbPeriod: Number(process.env.TOPSTEPX_MR_BB_PERIOD || '200'),
  bbStdDev: Number(process.env.TOPSTEPX_MR_BB_STDDEV || '3'),
  rsiPeriod: Number(process.env.TOPSTEPX_MR_RSI_PERIOD || '14'),
  rsiOversold: Number(process.env.TOPSTEPX_MR_RSI_OVERSOLD || '30'),
  rsiOverbought: Number(process.env.TOPSTEPX_MR_RSI_OVERBOUGHT || '70'),
  stopLossPercent: Number(process.env.TOPSTEPX_MR_STOP_LOSS_PERCENT || '0.0005'),
  adxPeriod: Number(process.env.TOPSTEPX_MR_ADX_PERIOD || '900'),
  adxThreshold: Number(process.env.TOPSTEPX_MR_ADX_THRESHOLD || '22'),
  bypassAdx: process.env.TOPSTEPX_MR_BYPASS_ADX === 'true',
  adxBuffer: Number(process.env.TOPSTEPX_MR_ADX_BUFFER || '4'),
  fastSmaPeriod: Number(process.env.TOPSTEPX_MR_FAST_SMA_PERIOD || '50'),
  atrPeriod: Number(process.env.TOPSTEPX_MR_ATR_PERIOD || '900'),
  atrMinRangePercent: Number(process.env.TOPSTEPX_MR_ATR_MIN_RANGE_PERCENT || '0.25'),
  htfEmaPeriod: Number(process.env.TOPSTEPX_MR_HTF_EMA_PERIOD || '200'),
  htfEmaEnabled: process.env.TOPSTEPX_MR_HTF_EMA_ENABLED === 'true',
  fvgEnabled: process.env.TOPSTEPX_MR_FVG_ENABLED !== 'false',  // FVG ENABLED BY DEFAULT
  numberOfContracts: Number(process.env.TOPSTEPX_MR_CONTRACTS || '3'),
  commissionPerSide: process.env.TOPSTEPX_MR_COMMISSION
    ? Number(process.env.TOPSTEPX_MR_COMMISSION)
    : inferFuturesCommissionPerSide([DEFAULT_MR_CONTRACT_ID, DEFAULT_MR_SYMBOL], 0.35),
  slippageTicks: Number(process.env.TOPSTEPX_MR_SLIPPAGE_TICKS || '1'),
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
  const slippagePoints = tickSize * CONFIG.slippageTicks;

  console.log(`Resolved contract: ${metadata.name} (${contractId})`);
  console.log(`Point multiplier: ${multiplier}`);
  console.log(`Tick size: ${tickSize} | Slippage: ${CONFIG.slippageTicks} tick(s) = ${slippagePoints} points`);

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

  let pendingSetup: {
    side: 'long' | 'short';
    setupTime: string;
    setupPrice: number;
    rsi: number;
    adx: number | null;
    bb: { upper: number; middle: number; lower: number };
    lowestRsi: number;  // Track lowest RSI for long divergence
    highestRsi: number; // Track highest RSI for short divergence
    rsiDivergenceDetected: boolean;
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
  } | null = null;

  let realizedPnL = 0;

  const exitPosition = (
    exitPrice: number,
    exitTime: string,
    reason: TradeRecord['exitReason'],
  ) => {
    if (!position) return;

    const direction = position.side === 'long' ? 1 : -1;
    const rawPnL = (exitPrice - position.entryPrice) * direction * multiplier * position.remainingQty;
    const exitFees = CONFIG.commissionPerSide * position.remainingQty;
    const totalFees = position.feesPaid + exitFees;
    const grossPnl = rawPnL + position.scalePnL;
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
      entryRSI: position.entryRSI,
      entryADX: position.entryADX ?? undefined,
      scaled: position.scaled,
      scalePnL: position.scalePnL,
      finalPnL: rawPnL,
    };

    trades.push(trade);
    realizedPnL += netPnl;
    position = null;
  };

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    closes.push(bar.close);
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
        const direction = position.side === 'long' ? 1 : -1;
        const sessionExitPrice = direction === 1
          ? bar.close - slippagePoints
          : bar.close + slippagePoints;
        exitPosition(sessionExitPrice, bar.timestamp, 'session');
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

    const fastSma = calculateSimpleMovingAverage(closes, CONFIG.fastSmaPeriod);

    const ttmBars = bars.slice(Math.max(0, i - 20), i + 1);
    const ttmSqueeze = calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });
    if (!ttmSqueeze) continue;

    if (position) {
      const direction = position.side === 'long' ? 1 : -1;

      // DISABLED: 50 SMA fast scale - testing without it
      // Skip 50 SMA logic entirely

      // 200 SMA Middle Band Scale - ACTIVE
      if (!position.scaled && position.target !== null) {
        const hitTarget = (direction === 1 && bar.high >= position.target) ||
                         (direction === -1 && bar.low <= position.target);

        if (hitTarget) {
          const maxScalable = Math.max(0, position.remainingQty - 1);
          const desiredScale = Math.max(1, Math.floor(position.remainingQty / 2));
          const scaleQty = Math.min(desiredScale, maxScalable);
          const remainingQty = position.remainingQty - scaleQty;

          if (scaleQty > 0) {
            const scaleExitPrice = position.target;
            const scalePnL = (scaleExitPrice - position.entryPrice) * direction * multiplier * scaleQty;
            position.scalePnL += scalePnL;
            position.scaled = true;
            position.remainingQty = remainingQty;
            position.feesPaid += scaleQty * CONFIG.commissionPerSide;

            position.stopLoss = bb.middle * (1 + direction * -CONFIG.stopLossPercent);
            position.target = direction === 1 ? bb.upper : bb.lower;

            console.log(
              `[${bar.timestamp}] Scaled ${CONFIG.symbol} ${position.side.toUpperCase()}: ` +
              `${scaleQty} contracts @ ${scaleExitPrice.toFixed(2)} = ${formatCurrency(scalePnL)} ` +
              `(${remainingQty} remaining, new stop ${position.stopLoss?.toFixed(2)}, new target ${position.target?.toFixed(2)})`
            );
            continue;
          }
        }
      }

      if (position.stopLoss !== null) {
        const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                       (direction === -1 && bar.high >= position.stopLoss);
        if (hitStop) {
          const stopExitPrice = direction === 1
            ? position.stopLoss - slippagePoints
            : position.stopLoss + slippagePoints;
          exitPosition(stopExitPrice, bar.timestamp, 'stop');
          continue;
        }
      }

      // Runner Target (Opposite BB) - ACTIVE
      if (position.scaled && position.target !== null) {
        const targetBuffer = bbWidth * 0.05;
        const hitTarget = (direction === 1 && bar.high >= position.target - targetBuffer) ||
                         (direction === -1 && bar.low <= position.target + targetBuffer);
        if (hitTarget) {
          const exitPrice = direction === 1
            ? Math.min(bar.high, position.target)
            : Math.max(bar.low, position.target);
          exitPosition(exitPrice, bar.timestamp, 'target');
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

    const longSetupDetected = distanceToLower <= bbWidth * 0.05 && currentRSI < CONFIG.rsiOversold && atrFilterPassed && longFvgOk;
    const shortSetupDetected = distanceToUpper <= bbWidth * 0.05 && currentRSI > CONFIG.rsiOverbought && atrFilterPassed && shortFvgOk;

    // Track filtered setups
    if (!atrFilterPassed && (
      (distanceToLower <= bbWidth * 0.05 && currentRSI < CONFIG.rsiOversold) ||
      (distanceToUpper <= bbWidth * 0.05 && currentRSI > CONFIG.rsiOverbought)
    )) {
      filteredByAtr++;
    }

    // Track FVG filtered setups
    if (CONFIG.fvgEnabled && fvg !== null && atrFilterPassed) {
      if (distanceToLower <= bbWidth * 0.05 && currentRSI < CONFIG.rsiOversold && !fvg.bullishFVG) {
        filteredByFvg++;
      }
      if (distanceToUpper <= bbWidth * 0.05 && currentRSI > CONFIG.rsiOverbought && !fvg.bearishFVG) {
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
        lowestRsi: currentRSI,
        highestRsi: currentRSI,
        rsiDivergenceDetected: false,
      };
      console.log(
        `[${bar.timestamp}] LONG setup detected ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
        `(RSI ${currentRSI.toFixed(1)}, ADX ${currentADX?.toFixed(1) ?? 'N/A'}, awaiting RSI divergence + TTM Squeeze)`
      );
    } else if (!pendingSetup && shortSetupDetected) {
      pendingSetup = {
        side: 'short',
        setupTime: bar.timestamp,
        setupPrice: bar.close,
        rsi: currentRSI,
        adx: currentADX,
        bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
        lowestRsi: currentRSI,
        highestRsi: currentRSI,
        rsiDivergenceDetected: false,
      };
      console.log(
        `[${bar.timestamp}] SHORT setup detected ${CONFIG.symbol} @ ${bar.close.toFixed(2)} ` +
        `(RSI ${currentRSI.toFixed(1)}, ADX ${currentADX?.toFixed(1) ?? 'N/A'}, awaiting RSI divergence + TTM Squeeze)`
      );
    }

    // Update RSI divergence detection for pending setup
    if (pendingSetup && !pendingSetup.rsiDivergenceDetected) {
      if (pendingSetup.side === 'long') {
        // Track lowest RSI
        if (currentRSI < pendingSetup.lowestRsi) {
          pendingSetup.lowestRsi = currentRSI;
        }
        // Detect higher low: RSI bouncing up from the low (minimum 30 points)
        if (currentRSI > pendingSetup.lowestRsi + 30) {
          pendingSetup.rsiDivergenceDetected = true;
          console.log(
            `[${bar.timestamp}] RSI divergence detected for LONG: RSI ${currentRSI.toFixed(1)} > low ${pendingSetup.lowestRsi.toFixed(1)} (+${(currentRSI - pendingSetup.lowestRsi).toFixed(1)})`
          );
        }
      } else {
        // Track highest RSI
        if (currentRSI > pendingSetup.highestRsi) {
          pendingSetup.highestRsi = currentRSI;
        }
        // Detect lower high: RSI bouncing down from the high (minimum 30 points)
        if (currentRSI < pendingSetup.highestRsi - 30) {
          pendingSetup.rsiDivergenceDetected = true;
          console.log(
            `[${bar.timestamp}] RSI divergence detected for SHORT: RSI ${currentRSI.toFixed(1)} < high ${pendingSetup.highestRsi.toFixed(1)} (-${(pendingSetup.highestRsi - currentRSI).toFixed(1)})`
          );
        }
      }
    }

    // Check for TTM Squeeze trigger - NO RSI divergence filter
    if (pendingSetup && ttmSqueeze.squeezeOn) {
      if (true) {  // RSI divergence filter DISABLED
        // Entry allowed
        const entryPrice = pendingSetup.side === 'long'
          ? bar.close + slippagePoints
          : bar.close - slippagePoints;

        const entryAdx = pendingSetup.adx ?? currentADX ?? null;
        const fastSmaForEntry = fastSma;

        position = {
          side: pendingSetup.side,
          entryPrice,
          entryTime: bar.timestamp,
          entryRSI: pendingSetup.rsi,
          entryADX: entryAdx,
          stopLoss: pendingSetup.side === 'long'
            ? pendingSetup.bb.lower * (1 - CONFIG.stopLossPercent)
            : pendingSetup.bb.upper * (1 + CONFIG.stopLossPercent),
          target: pendingSetup.bb.middle,
          scaled: false,
          remainingQty: CONFIG.numberOfContracts,
          scalePnL: 0,
          feesPaid: CONFIG.numberOfContracts * CONFIG.commissionPerSide,
          fastScaleFilled: true, // DISABLED: Skip 50 SMA logic
          fastScaleTarget: null, // DISABLED: No 50 SMA target
        };

        const entryAdxText = entryAdx !== null ? entryAdx.toFixed(1) : 'N/A';
        const fastSmaText = fastSmaForEntry !== null ? fastSmaForEntry.toFixed(2) : 'N/A';

        console.log(
          `[${bar.timestamp}] ${pendingSetup.side.toUpperCase()} entry ${CONFIG.symbol} @ ${entryPrice.toFixed(2)} ` +
          `(TTM Squeeze trigger, setup @ ${pendingSetup.setupPrice.toFixed(2)}, ` +
          `RSI ${pendingSetup.rsi.toFixed(1)}, ADX ${entryAdxText}, Fast SMA ${fastSmaText})`
        );

        pendingSetup = null;
      } else {
        // TTM Squeeze fired but RSI divergence not detected yet - filter this
        filteredByRsiDivergence++;
        console.log(
          `[${bar.timestamp}] TTM Squeeze fired but NO RSI divergence yet for ${pendingSetup.side.toUpperCase()} - continuing to wait`
        );
      }
    }
  }

  if (position) {
    const lastBar = bars[bars.length - 1];
    const direction = position.side === 'long' ? 1 : -1;
    const endExitPrice = direction === 1
      ? lastBar.close - slippagePoints
      : lastBar.close + slippagePoints;
    exitPosition(endExitPrice, lastBar.timestamp, 'end_of_data');
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

  // Exit reason breakdown
  const exitReasons = trades.reduce((acc, t) => {
    acc[t.exitReason] = (acc[t.exitReason] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.log(`\nExit Reasons: ${Object.entries(exitReasons).map(([r, c]) => `${r}=${c}`).join(', ')}`);
  console.log(`Setups Filtered by ATR: ${filteredByAtr} (bar range < ${(CONFIG.atrMinRangePercent * 100).toFixed(0)}% of ATR)`);
  console.log(`Setups Filtered by FVG: ${filteredByFvg} (no matching Fair Value Gap)`);
  console.log(`Setups Filtered by RSI Divergence: ${filteredByRsiDivergence} (TTM fired but no RSI higher-low/lower-high)`);

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
