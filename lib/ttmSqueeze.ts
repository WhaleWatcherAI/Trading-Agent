type Sentiment = 'bullish' | 'bearish' | 'neutral';

export interface TtmSqueezeConfig {
  lookback?: number;
  bbStdDev?: number;
  atrMultiplier?: number;
  momentumThreshold?: number;
}

export interface PriceBar {
  high: number;
  low: number;
  close: number;
}

export interface TtmSqueezeResult {
  squeezeOn: boolean;
  squeezeOff: boolean;
  momentum: number;
  sentiment: Sentiment;
  bbUpper: number;
  bbLower: number;
  kcUpper: number;
  kcLower: number;
}

/**
 * Calculates the latest TTM Squeeze state.
 */
export function calculateTtmSqueeze(
  bars: PriceBar[],
  config: TtmSqueezeConfig = {}
): TtmSqueezeResult | null {
  const lookback = config.lookback ?? 20;
  const bbStdDev = config.bbStdDev ?? 2;
  const atrMultiplier = config.atrMultiplier ?? 1.5;
  const momentumThreshold = config.momentumThreshold ?? 1e-5;

  if (bars.length < lookback + 1) {
    return null;
  }

  const recent = bars.slice(-lookback);
  const closes = recent.map(bar => bar.close);

  const sma = closes.reduce((acc, close) => acc + close, 0) / lookback;
  const variance =
    closes.reduce((acc, close) => acc + Math.pow(close - sma, 2), 0) /
    lookback;
  const stdDev = Math.sqrt(variance);

  const bbUpper = sma + bbStdDev * stdDev;
  const bbLower = sma - bbStdDev * stdDev;

  let trueRangeSum = 0;
  for (let i = bars.length - lookback; i < bars.length; i++) {
    const current = bars[i];
    const prev = bars[i - 1];
    const trueRange = Math.max(
      current.high - current.low,
      Math.abs(current.high - prev.close),
      Math.abs(current.low - prev.close)
    );
    trueRangeSum += trueRange;
  }

  const atr = trueRangeSum / lookback;
  const kcUpper = sma + atrMultiplier * atr;
  const kcLower = sma - atrMultiplier * atr;

  const squeezeOn = bbUpper <= kcUpper && bbLower >= kcLower;
  const squeezeOff = bbUpper >= kcUpper && bbLower <= kcLower;

  const n = closes.length;
  const indices = Array.from({ length: n }, (_, idx) => idx);
  const sumX = indices.reduce((acc, val) => acc + val, 0);
  const sumY = closes.reduce((acc, val) => acc + val, 0);
  const sumXY = closes.reduce((acc, y, idx) => acc + idx * y, 0);
  const sumX2 = indices.reduce((acc, val) => acc + val * val, 0);
  const denominator = n * sumX2 - sumX * sumX;
  const slope =
    denominator === 0 ? 0 : (n * sumXY - sumX * sumY) / denominator;

  let sentiment: Sentiment = 'neutral';
  if (slope > momentumThreshold) {
    sentiment = 'bullish';
  } else if (slope < -momentumThreshold) {
    sentiment = 'bearish';
  }

  return {
    squeezeOn,
    squeezeOff,
    momentum: slope,
    sentiment,
    bbUpper,
    bbLower,
    kcUpper,
    kcLower,
  };
}
