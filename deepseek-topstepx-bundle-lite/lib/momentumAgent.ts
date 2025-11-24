import { SMA } from 'technicalindicators';

export interface MomentumConfig {
  symbol: string;
  date?: string;
  fastPeriod?: number; // Default 9
  slowPeriod?: number; // Default 20
  atrPeriod?: number; // Default 14 for stop loss
  atrMultiplier?: number; // Default 2 for stop loss
}

export interface MomentumSignal {
  symbol: string;
  timestamp: string;
  action: 'buy_call' | 'buy_put' | 'close' | 'none';
  direction: 'bullish' | 'bearish' | 'neutral';
  currentPrice: number;
  fastSMA: number | null;
  slowSMA: number | null;
  trend: 'golden_cross' | 'death_cross' | 'none';
  netGex: number;
  regime: 'negative_gex' | 'positive_gex';
  rationale: string[];
}

export interface TrendContext {
  fastSMA: number;
  slowSMA: number;
  prevFastSMA: number;
  prevSlowSMA: number;
  atr: number | null;
}

/**
 * Calculate SMAs and determine trend
 */
export function calculateTrend(
  closes: number[],
  fastPeriod: number = 9,
  slowPeriod: number = 20,
): TrendContext | null {
  if (closes.length < slowPeriod) {
    return null;
  }

  // Calculate SMAs
  const fastSMAValues = SMA.calculate({ values: closes, period: fastPeriod });
  const slowSMAValues = SMA.calculate({ values: closes, period: slowPeriod });

  if (fastSMAValues.length < 2 || slowSMAValues.length < 2) {
    return null;
  }

  const fastSMA = fastSMAValues[fastSMAValues.length - 1];
  const slowSMA = slowSMAValues[slowSMAValues.length - 1];
  const prevFastSMA = fastSMAValues[fastSMAValues.length - 2];
  const prevSlowSMA = slowSMAValues[slowSMAValues.length - 2];

  // Calculate ATR for stop loss (simplified: use high-low range)
  const atrPeriod = 14;
  if (closes.length >= atrPeriod) {
    const recentCloses = closes.slice(-atrPeriod);
    const ranges = [];
    for (let i = 1; i < recentCloses.length; i++) {
      ranges.push(Math.abs(recentCloses[i] - recentCloses[i - 1]));
    }
    const atr = ranges.reduce((sum, r) => sum + r, 0) / ranges.length;

    return {
      fastSMA,
      slowSMA,
      prevFastSMA,
      prevSlowSMA,
      atr,
    };
  }

  return {
    fastSMA,
    slowSMA,
    prevFastSMA,
    prevSlowSMA,
    atr: null,
  };
}

/**
 * Detect crossover events
 */
export function detectCrossover(trend: TrendContext): 'golden_cross' | 'death_cross' | 'none' {
  const { fastSMA, slowSMA, prevFastSMA, prevSlowSMA } = trend;

  // Golden Cross: Fast SMA crosses ABOVE Slow SMA
  if (prevFastSMA <= prevSlowSMA && fastSMA > slowSMA) {
    return 'golden_cross';
  }

  // Death Cross: Fast SMA crosses BELOW Slow SMA
  if (prevFastSMA >= prevSlowSMA && fastSMA < slowSMA) {
    return 'death_cross';
  }

  return 'none';
}

/**
 * Generate momentum/trend following signal
 * Only trades on NEGATIVE GEX days (trending regime)
 */
export function generateMomentumSignal(
  symbol: string,
  currentPrice: number,
  priceHistory: number[],
  netGex: number,
  config: {
    fastPeriod?: number;
    slowPeriod?: number;
  } = {},
): MomentumSignal {
  const {
    fastPeriod = 9,
    slowPeriod = 20,
  } = config;

  const timestamp = new Date().toISOString();
  const regime: 'negative_gex' | 'positive_gex' = netGex < 0 ? 'negative_gex' : 'positive_gex';

  // Initialize default signal
  const defaultSignal: MomentumSignal = {
    symbol,
    timestamp,
    action: 'none',
    direction: 'neutral',
    currentPrice,
    fastSMA: null,
    slowSMA: null,
    trend: 'none',
    netGex,
    regime,
    rationale: [],
  };

  // TEMPORARILY DISABLED: Only trade on NEGATIVE GEX days (trending markets)
  // TODO: Re-enable once we confirm negative GEX days exist in dataset
  // if (netGex >= 0) {
  //   defaultSignal.rationale.push(
  //     `Net GEX is positive ($${(netGex / 1_000_000).toFixed(1)}M) - no trending regime`,
  //   );
  //   return defaultSignal;
  // }

  // Need sufficient price history
  if (!priceHistory || priceHistory.length < slowPeriod) {
    defaultSignal.rationale.push(
      `Insufficient price history (${priceHistory?.length || 0} bars, need ${slowPeriod})`,
    );
    return defaultSignal;
  }

  // Calculate trend
  const trend = calculateTrend(priceHistory, fastPeriod, slowPeriod);
  if (!trend) {
    defaultSignal.rationale.push('Failed to calculate trend indicators');
    return defaultSignal;
  }

  const { fastSMA, slowSMA } = trend;
  const crossover = detectCrossover(trend);

  const rationale: string[] = [
    `Net GEX: $${(netGex / 1_000_000).toFixed(1)}M (NEGATIVE - trending regime)`,
    `SMA(${fastPeriod}): $${fastSMA.toFixed(2)} | SMA(${slowPeriod}): $${slowSMA.toFixed(2)}`,
    `Current price: $${currentPrice.toFixed(2)}`,
  ];

  // GOLDEN CROSS: Buy Calls (bullish trend starting)
  if (crossover === 'golden_cross') {
    rationale.push(`🟢 GOLDEN CROSS: SMA(${fastPeriod}) crossed above SMA(${slowPeriod})`);
    rationale.push(`Action: Buy CALL options`);

    return {
      symbol,
      timestamp,
      action: 'buy_call',
      direction: 'bullish',
      currentPrice,
      fastSMA,
      slowSMA,
      trend: 'golden_cross',
      netGex,
      regime,
      rationale,
    };
  }

  // DEATH CROSS: Buy Puts (bearish trend starting)
  if (crossover === 'death_cross') {
    rationale.push(`🔴 DEATH CROSS: SMA(${fastPeriod}) crossed below SMA(${slowPeriod})`);
    rationale.push(`Action: Buy PUT options`);

    return {
      symbol,
      timestamp,
      action: 'buy_put',
      direction: 'bearish',
      currentPrice,
      fastSMA,
      slowSMA,
      trend: 'death_cross',
      netGex,
      regime,
      rationale,
    };
  }

  // No crossover - check if we should hold existing trend
  if (fastSMA > slowSMA) {
    rationale.push(`Bullish trend in progress (SMA${fastPeriod} > SMA${slowPeriod}) - hold calls`);
    return {
      symbol,
      timestamp,
      action: 'none',
      direction: 'bullish',
      currentPrice,
      fastSMA,
      slowSMA,
      trend: 'none',
      netGex,
      regime,
      rationale,
    };
  } else if (fastSMA < slowSMA) {
    rationale.push(`Bearish trend in progress (SMA${fastPeriod} < SMA${slowPeriod}) - hold puts`);
    return {
      symbol,
      timestamp,
      action: 'none',
      direction: 'bearish',
      currentPrice,
      fastSMA,
      slowSMA,
      trend: 'none',
      netGex,
      regime,
      rationale,
    };
  }

  rationale.push(`No clear trend or crossover`);
  return {
    symbol,
    timestamp,
    action: 'none',
    direction: 'neutral',
    currentPrice,
    fastSMA,
    slowSMA,
    trend: 'none',
    netGex,
    regime,
    rationale,
  };
}

/**
 * Check if current position should exit (opposite crossover)
 */
export function checkMomentumExit(
  currentDirection: 'bullish' | 'bearish',
  trend: TrendContext,
): {
  shouldExit: boolean;
  exitReason: 'opposite_cross' | 'stop_loss' | 'none';
} {
  const crossover = detectCrossover(trend);

  // Exit on opposite crossover
  if (currentDirection === 'bullish' && crossover === 'death_cross') {
    return { shouldExit: true, exitReason: 'opposite_cross' };
  }

  if (currentDirection === 'bearish' && crossover === 'golden_cross') {
    return { shouldExit: true, exitReason: 'opposite_cross' };
  }

  return { shouldExit: false, exitReason: 'none' };
}
