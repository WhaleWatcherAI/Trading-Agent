/**
 * Keltner Channel + RSI Scalping Strategy
 *
 * Entry:
 * - LONG: RSI < 25 AND price closes outside lower Keltner(20, 1.5)
 * - SHORT: RSI > 75 AND price closes outside upper Keltner(20, 1.5)
 *
 * Exit:
 * - Target: Middle Keltner band
 * - OR RSI returns to 50 (neutral)
 *
 * Filter:
 * - Skip trades if ADX > 25 (market is trending)
 */

import { RSI, ATR, EMA, ADX } from 'technicalindicators';

export interface KeltnerScalpConfig {
  symbol: string;
  rsiPeriod?: number;         // Default: 14
  rsiOversold?: number;       // Default: 25
  rsiOverbought?: number;     // Default: 75
  rsiNeutral?: number;        // Default: 50
  keltnerPeriod?: number;     // Default: 20
  keltnerMultiplier?: number; // Default: 1.5
  adxPeriod?: number;         // Default: 14
  adxThreshold?: number;      // Default: 25
  stopLossPercent?: number;   // Default: 0.002 (0.2%)
}

export interface KeltnerScalpSignal {
  symbol: string;
  timestamp: string;
  direction: 'long' | 'short' | 'none';
  entryPrice: number;
  stopLoss: number | null;
  target: number | null;
  rsi: number | null;
  adx: number | null;
  keltnerUpper: number | null;
  keltnerMiddle: number | null;
  keltnerLower: number | null;
  rationale: string[];
}

export interface KeltnerBands {
  upper: number;
  middle: number;
  lower: number;
}

/**
 * Calculate Keltner Channels
 * Middle = EMA(period)
 * Upper = Middle + (multiplier × ATR(period))
 * Lower = Middle - (multiplier × ATR(period))
 */
export function calculateKeltnerChannels(
  highs: number[],
  lows: number[],
  closes: number[],
  period: number = 20,
  multiplier: number = 1.5,
): KeltnerBands[] {
  if (highs.length < period || lows.length < period || closes.length < period) {
    return [];
  }

  // Calculate EMA (middle band)
  const emaValues = EMA.calculate({ values: closes, period });

  // Calculate ATR
  const atrInput = highs.map((high, i) => ({
    high,
    low: lows[i],
    close: closes[i],
  }));
  const atrValues = ATR.calculate({ high: highs, low: lows, close: closes, period });

  // Build Keltner bands
  const keltnerBands: KeltnerBands[] = [];
  const offset = closes.length - emaValues.length;

  for (let i = 0; i < emaValues.length; i++) {
    const middle = emaValues[i];
    const atr = atrValues[i] || 0;
    const channelWidth = atr * multiplier;

    keltnerBands.push({
      middle,
      upper: middle + channelWidth,
      lower: middle - channelWidth,
    });
  }

  return keltnerBands;
}

/**
 * Generate scalping signal based on Keltner + RSI
 */
export function generateKeltnerScalpSignal(
  symbol: string,
  currentPrice: number,
  priceHistory: { high: number; low: number; close: number }[],
  config: KeltnerScalpConfig = {},
): KeltnerScalpSignal {
  const {
    rsiPeriod = 14,
    rsiOversold = 25,
    rsiOverbought = 75,
    rsiNeutral = 50,
    keltnerPeriod = 20,
    keltnerMultiplier = 1.5,
    adxPeriod = 14,
    adxThreshold = 25,
    stopLossPercent = 0.002, // 0.2% default for scalping
  } = config;

  const timestamp = new Date().toISOString();

  const defaultSignal: KeltnerScalpSignal = {
    symbol,
    timestamp,
    direction: 'none',
    entryPrice: currentPrice,
    stopLoss: null,
    target: null,
    rsi: null,
    adx: null,
    keltnerUpper: null,
    keltnerMiddle: null,
    keltnerLower: null,
    rationale: [],
  };

  // Need enough price history
  const minBars = Math.max(rsiPeriod, keltnerPeriod, adxPeriod) + 5;
  if (priceHistory.length < minBars) {
    defaultSignal.rationale.push(
      `Insufficient data (${priceHistory.length} bars, need ${minBars})`,
    );
    return defaultSignal;
  }

  const closes = priceHistory.map(p => p.close);
  const highs = priceHistory.map(p => p.high);
  const lows = priceHistory.map(p => p.low);

  // Calculate RSI
  const rsiValues = RSI.calculate({ values: closes, period: rsiPeriod });
  const currentRSI = rsiValues[rsiValues.length - 1];
  if (!currentRSI) {
    defaultSignal.rationale.push('Failed to calculate RSI');
    return defaultSignal;
  }

  // Calculate Keltner Channels
  const keltnerBands = calculateKeltnerChannels(
    highs,
    lows,
    closes,
    keltnerPeriod,
    keltnerMultiplier,
  );
  const currentKeltner = keltnerBands[keltnerBands.length - 1];
  if (!currentKeltner) {
    defaultSignal.rationale.push('Failed to calculate Keltner Channels');
    return defaultSignal;
  }

  // Calculate ADX (trend strength filter)
  const adxInput = highs.map((high, i) => ({
    high,
    low: lows[i],
    close: closes[i],
  }));
  const adxValues = ADX.calculate({ high: highs, low: lows, close: closes, period: adxPeriod });
  const adxResult = adxValues[adxValues.length - 1];
  const currentADX = adxResult && typeof adxResult === 'object' ? adxResult.adx : adxResult;

  const rationale: string[] = [
    `RSI: ${currentRSI.toFixed(1)}`,
    `Keltner: ${currentKeltner.lower.toFixed(2)} - ${currentKeltner.middle.toFixed(2)} - ${currentKeltner.upper.toFixed(2)}`,
    `ADX: ${currentADX?.toFixed(1) || 'N/A'}`,
    `Price: ${currentPrice.toFixed(2)}`,
  ];

  // Filter: Skip if trending (ADX > threshold)
  if (currentADX && currentADX > adxThreshold) {
    rationale.push(`⚠️ ADX too high (${currentADX.toFixed(1)} > ${adxThreshold}) - market trending, skip`);
    return {
      ...defaultSignal,
      rsi: currentRSI,
      adx: currentADX,
      keltnerUpper: currentKeltner.upper,
      keltnerMiddle: currentKeltner.middle,
      keltnerLower: currentKeltner.lower,
      rationale,
    };
  }

  // LONG Setup: RSI oversold + price outside lower Keltner
  if (currentRSI < rsiOversold && currentPrice < currentKeltner.lower) {
    const stopLoss = currentPrice * (1 - stopLossPercent);
    const target = currentKeltner.middle;

    rationale.push(`🟢 LONG SCALP: RSI ${currentRSI.toFixed(1)} < ${rsiOversold} + price below Keltner`);
    rationale.push(`Stop: ${stopLoss.toFixed(2)} | Target: ${target.toFixed(2)}`);

    return {
      symbol,
      timestamp,
      direction: 'long',
      entryPrice: currentPrice,
      stopLoss,
      target,
      rsi: currentRSI,
      adx: currentADX,
      keltnerUpper: currentKeltner.upper,
      keltnerMiddle: currentKeltner.middle,
      keltnerLower: currentKeltner.lower,
      rationale,
    };
  }

  // SHORT Setup: RSI overbought + price outside upper Keltner
  if (currentRSI > rsiOverbought && currentPrice > currentKeltner.upper) {
    const stopLoss = currentPrice * (1 + stopLossPercent);
    const target = currentKeltner.middle;

    rationale.push(`🔴 SHORT SCALP: RSI ${currentRSI.toFixed(1)} > ${rsiOverbought} + price above Keltner`);
    rationale.push(`Stop: ${stopLoss.toFixed(2)} | Target: ${target.toFixed(2)}`);

    return {
      symbol,
      timestamp,
      direction: 'short',
      entryPrice: currentPrice,
      stopLoss,
      target,
      rsi: currentRSI,
      adx: currentADX,
      keltnerUpper: currentKeltner.upper,
      keltnerMiddle: currentKeltner.middle,
      keltnerLower: currentKeltner.lower,
      rationale,
    };
  }

  // No setup
  rationale.push('No scalp setup - waiting for RSI extreme + Keltner breakout');

  return {
    ...defaultSignal,
    rsi: currentRSI,
    adx: currentADX,
    keltnerUpper: currentKeltner.upper,
    keltnerMiddle: currentKeltner.middle,
    keltnerLower: currentKeltner.lower,
    rationale,
  };
}

/**
 * Check if position should exit
 */
export function checkKeltnerScalpExit(
  currentPrice: number,
  currentRSI: number,
  position: {
    direction: 'long' | 'short';
    entryPrice: number;
    target: number;
    stopLoss: number;
  },
  config: { rsiNeutral?: number } = {},
): { shouldExit: boolean; reason: string; exitPrice: number } | null {
  const { rsiNeutral = 50 } = config;

  // Check stop loss
  if (
    (position.direction === 'long' && currentPrice <= position.stopLoss) ||
    (position.direction === 'short' && currentPrice >= position.stopLoss)
  ) {
    return {
      shouldExit: true,
      reason: 'stop',
      exitPrice: position.stopLoss,
    };
  }

  // Check target
  if (
    (position.direction === 'long' && currentPrice >= position.target) ||
    (position.direction === 'short' && currentPrice <= position.target)
  ) {
    return {
      shouldExit: true,
      reason: 'target',
      exitPrice: position.target,
    };
  }

  // Check RSI neutral (mean reversion complete)
  const rsiNeutralRange = 5; // ±5 around neutral
  if (Math.abs(currentRSI - rsiNeutral) < rsiNeutralRange) {
    return {
      shouldExit: true,
      reason: 'rsi_neutral',
      exitPrice: currentPrice,
    };
  }

  return null;
}
