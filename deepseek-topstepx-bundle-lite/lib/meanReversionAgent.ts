import {
  calculateGexForSymbol,
  GexCalculationResult,
  GexMode,
} from './gexCalculator';
import { BollingerBands, RSI, SMA } from 'technicalindicators';

export interface MeanReversionConfig {
  symbol: string;
  date?: string; // For historical backtesting
  mode?: GexMode; // Default 'intraday'
  rsiPeriod?: number; // RSI period (default 14)
  rsiOversold?: number; // RSI oversold threshold (default 30)
  rsiOverbought?: number; // RSI overbought threshold (default 70)
  bbPeriod?: number; // Bollinger Band period (default 20)
  bbStdDev?: number; // Bollinger Band standard deviations (default 2)
  bbThreshold?: number; // How far outside bands to trigger (default 0.002 = 0.2%)
  stopLossPercent?: number; // Stop loss as % of entry (default 0.01 = 1%)
  targetPercent?: number; // Profit target as % of entry (default 0.015 = 1.5%)
}

export interface MeanReversionSignal {
  symbol: string;
  timestamp: string;
  action: 'buy' | 'sell' | 'none';
  direction: 'long' | 'short' | 'none';
  entryPrice: number;
  stopLoss: number | null;
  target: number | null;
  rsi: number | null;
  bbUpper: number | null;
  bbMiddle: number | null;
  bbLower: number | null;
  bbPercentB: number | null; // Where price is in the bands (0 = lower, 1 = upper)
  netGex: number;
  regime: 'positive_gex' | 'negative_gex';
  rationale: string[];
}

export interface TechnicalContext {
  rsi: number;
  bbUpper: number;
  bbMiddle: number;
  bbLower: number;
  bbWidth: number;
  sma20: number;
}

/**
 * Calculate technical indicators from price history
 */
export function calculateTechnicals(
  closes: number[],
  rsiPeriod: number = 14,
  bbPeriod: number = 20,
  bbStdDev: number = 2,
): TechnicalContext | null {
  if (closes.length < Math.max(rsiPeriod, bbPeriod)) {
    return null;
  }

  // Calculate RSI
  const rsiValues = RSI.calculate({ values: closes, period: rsiPeriod });
  const rsi = rsiValues[rsiValues.length - 1];

  // Calculate Bollinger Bands
  const bbValues = BollingerBands.calculate({
    values: closes,
    period: bbPeriod,
    stdDev: bbStdDev,
  });
  const latestBB = bbValues[bbValues.length - 1];

  // Calculate SMA 20
  const smaValues = SMA.calculate({ values: closes, period: bbPeriod });
  const sma20 = smaValues[smaValues.length - 1];

  if (!rsi || !latestBB || !sma20) {
    return null;
  }

  return {
    rsi,
    bbUpper: latestBB.upper,
    bbMiddle: latestBB.middle,
    bbLower: latestBB.lower,
    bbWidth: latestBB.upper - latestBB.lower,
    sma20,
  };
}

/**
 * Generate mean reversion signal based on technical indicators only
 * (Assumes GEX regime has already been checked)
 */
export function generateMeanReversionSignalFromTechnicals(
  symbol: string,
  currentPrice: number,
  priceHistory: number[],
  netGex: number,
  config: {
    rsiPeriod?: number;
    rsiOversold?: number;
    rsiOverbought?: number;
    bbPeriod?: number;
    bbStdDev?: number;
    bbThreshold?: number;
    stopLossPercent?: number;
    targetPercent?: number;
  } = {},
): MeanReversionSignal {
  const {
    rsiPeriod = 14,
    rsiOversold = 30,
    rsiOverbought = 70,
    bbPeriod = 20,
    bbStdDev = 2,
    bbThreshold = 0.005, // 0.5%
    stopLossPercent = 0.01, // 1%
  } = config;

  const timestamp = new Date().toISOString();
  const regime: 'positive_gex' | 'negative_gex' = netGex > 0 ? 'positive_gex' : 'negative_gex';

  // Initialize default signal (no trade)
  const defaultSignal: MeanReversionSignal = {
    symbol,
    timestamp,
    action: 'none',
    direction: 'none',
    entryPrice: currentPrice,
    stopLoss: null,
    target: null,
    rsi: null,
    bbUpper: null,
    bbMiddle: null,
    bbLower: null,
    bbPercentB: null,
    netGex,
    regime,
    rationale: [],
  };

  // Need price history to calculate indicators
  if (!priceHistory || priceHistory.length < Math.max(rsiPeriod, bbPeriod)) {
    defaultSignal.rationale.push(
      `Insufficient price history (${priceHistory?.length || 0} bars, need ${Math.max(rsiPeriod, bbPeriod)})`,
    );
    return defaultSignal;
  }

  // Calculate technical indicators
  const technicals = calculateTechnicals(priceHistory, rsiPeriod, bbPeriod, bbStdDev);
  if (!technicals) {
    defaultSignal.rationale.push('Failed to calculate technical indicators');
    return defaultSignal;
  }

  const { rsi, bbUpper, bbMiddle, bbLower, bbWidth } = technicals;
  const bbPercentB = (currentPrice - bbLower) / (bbWidth || 1);

  const rationale: string[] = [
    `Net GEX: $${(netGex / 1_000_000).toFixed(1)}M`,
    `RSI: ${rsi.toFixed(1)} | BB: $${bbLower.toFixed(2)} - $${bbMiddle.toFixed(2)} - $${bbUpper.toFixed(2)}`,
    `Current price: $${currentPrice.toFixed(2)} (BB %B: ${(bbPercentB * 100).toFixed(1)}%)`,
  ];

  // LONG Setup: RSI oversold + price at/below lower Bollinger Band
  const distanceFromLowerBB = Math.abs(currentPrice - bbLower) / currentPrice;
  if (rsi < rsiOversold && distanceFromLowerBB < bbThreshold) {
    const stopLoss = currentPrice * (1 - stopLossPercent);
    const target = bbMiddle; // Target middle band

    rationale.push(`🟢 LONG: RSI oversold (${rsi.toFixed(1)}) + price at lower BB`);
    rationale.push(`Stop: $${stopLoss.toFixed(2)} | Target: $${target.toFixed(2)}`);

    return {
      symbol,
      timestamp,
      action: 'buy',
      direction: 'long',
      entryPrice: currentPrice,
      stopLoss,
      target,
      rsi,
      bbUpper,
      bbMiddle,
      bbLower,
      bbPercentB,
      netGex,
      regime,
      rationale,
    };
  }

  // SHORT Setup: RSI overbought + price at/above upper Bollinger Band
  const distanceFromUpperBB = Math.abs(bbUpper - currentPrice) / currentPrice;
  if (rsi > rsiOverbought && distanceFromUpperBB < bbThreshold) {
    const stopLoss = currentPrice * (1 + stopLossPercent);
    const target = bbMiddle; // Target middle band

    rationale.push(`🔴 SHORT: RSI overbought (${rsi.toFixed(1)}) + price at upper BB`);
    rationale.push(`Stop: $${stopLoss.toFixed(2)} | Target: $${target.toFixed(2)}`);

    return {
      symbol,
      timestamp,
      action: 'sell',
      direction: 'short',
      entryPrice: currentPrice,
      stopLoss,
      target,
      rsi,
      bbUpper,
      bbMiddle,
      bbLower,
      bbPercentB,
      netGex,
      regime,
      rationale,
    };
  }

  // No entry setup
  if (rsi >= rsiOversold && rsi <= rsiOverbought) {
    rationale.push(`RSI neutral (${rsi.toFixed(1)}) - waiting for oversold/overbought`);
  } else if (distanceFromLowerBB >= bbThreshold && distanceFromUpperBB >= bbThreshold) {
    rationale.push(`Price not at Bollinger Band extremes - waiting`);
  } else {
    rationale.push(`No clear mean reversion setup`);
  }

  return {
    symbol,
    timestamp,
    action: 'none',
    direction: 'none',
    entryPrice: currentPrice,
    stopLoss: null,
    target: null,
    rsi,
    bbUpper,
    bbMiddle,
    bbLower,
    bbPercentB,
    netGex,
    regime,
    rationale,
  };
}

/**
 * Generate mean reversion signal based on positive net GEX day + technical indicators
 * (Includes GEX calculation - use this for initial setup)
 */
export async function generateMeanReversionSignal(
  config: MeanReversionConfig,
  priceHistory?: number[], // Optional pre-calculated price history
): Promise<MeanReversionSignal> {
  const {
    symbol,
    date,
    mode = 'intraday',
  } = config;

  const bypassGex = process.env.BYPASS_GEX === 'true';
  const historicalPrice = priceHistory && priceHistory.length > 0
    ? priceHistory[priceHistory.length - 1]
    : undefined;

  let currentPrice: number;
  let netGex: number;
  let regime: 'positive_gex' | 'negative_gex';
  let gex: GexCalculationResult | null = null;

  if (bypassGex && Number.isFinite(historicalPrice ?? NaN)) {
    currentPrice = historicalPrice as number;
    netGex = 1;
    regime = 'positive_gex';
  } else {
    gex = await calculateGexForSymbol(symbol, mode, date);
    currentPrice = gex.stockPrice;
    netGex = gex.summary.netGex;
    regime = netGex > 0 ? 'positive_gex' : 'negative_gex';
  }

  if (!Number.isFinite(currentPrice) || currentPrice <= 0) {
    throw new Error(`Invalid reference price for ${symbol}`);
  }

  const timestamp = new Date().toISOString();

  // Initialize default signal (no trade)
  const defaultSignal: MeanReversionSignal = {
    symbol,
    timestamp,
    action: 'none',
    direction: 'none',
    entryPrice: currentPrice,
    stopLoss: null,
    target: null,
    rsi: null,
    bbUpper: null,
    bbMiddle: null,
    bbLower: null,
    bbPercentB: null,
    netGex,
    regime,
    rationale: [],
  };

  // Only trade on POSITIVE net GEX days (pinning regime)
  if (!bypassGex && netGex <= 0) {
    defaultSignal.rationale.push(
      `Net GEX is negative ($${(netGex / 1_000_000).toFixed(1)}M) - no mean reversion regime`,
    );
    return defaultSignal;
  }

  // If no price history provided, just return GEX check result
  if (!priceHistory) {
    if (bypassGex) {
      defaultSignal.rationale.push('BYPASS_GEX enabled (GEX checks skipped)');
    }
    return defaultSignal;
  }

  if (bypassGex) {
    defaultSignal.rationale.push('BYPASS_GEX enabled (GEX checks skipped)');
  } else if (gex) {
    defaultSignal.rationale.push(
      `Net GEX: $${(netGex / 1_000_000).toFixed(1)}M (regime ${regime})`,
    );
  }

  // Generate signal from technicals
  return generateMeanReversionSignalFromTechnicals(symbol, currentPrice, priceHistory, netGex, config);
}

/**
 * Check if current price should trigger an exit
 */
export function checkMeanReversionExit(
  currentPrice: number,
  signal: MeanReversionSignal,
  currentRSI?: number,
): {
  shouldExit: boolean;
  exitReason: 'stop' | 'target' | 'rsi_neutral' | 'none';
  exitPrice: number;
} {
  const { direction, stopLoss, target } = signal;

  // No active trade
  if (direction === 'none') {
    return { shouldExit: false, exitReason: 'none', exitPrice: currentPrice };
  }

  // Check stop loss
  if (stopLoss !== null) {
    if (direction === 'long' && currentPrice <= stopLoss) {
      return { shouldExit: true, exitReason: 'stop', exitPrice: currentPrice };
    }
    if (direction === 'short' && currentPrice >= stopLoss) {
      return { shouldExit: true, exitReason: 'stop', exitPrice: currentPrice };
    }
  }

  // Check target (middle Bollinger Band)
  if (target !== null) {
    if (direction === 'long' && currentPrice >= target) {
      return { shouldExit: true, exitReason: 'target', exitPrice: currentPrice };
    }
    if (direction === 'short' && currentPrice <= target) {
      return { shouldExit: true, exitReason: 'target', exitPrice: currentPrice };
    }
  }

  // Optional: Exit if RSI returns to neutral zone (50) - BYPASSED
  // if (currentRSI !== undefined) {
  //   if (direction === 'long' && currentRSI > 50) {
  //     return { shouldExit: true, exitReason: 'rsi_neutral', exitPrice: currentPrice };
  //   }
  //   if (direction === 'short' && currentRSI < 50) {
  //     return { shouldExit: true, exitReason: 'rsi_neutral', exitPrice: currentPrice };
  //   }
  // }

  return { shouldExit: false, exitReason: 'none', exitPrice: currentPrice };
}
