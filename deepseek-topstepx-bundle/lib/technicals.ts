import axios from 'axios';
import { RSI, SMA, EMA, MACD, BollingerBands } from 'technicalindicators';
import { getHistoricalTimesales } from './tradier';

const TRADIER_API_KEY = process.env.TRADIER_API_KEY || '';
const TRADIER_BASE_URL = process.env.TRADIER_BASE_URL || 'https://api.tradier.com/v1';

const tradierClient = axios.create({
  baseURL: TRADIER_BASE_URL,
  headers: {
    'Authorization': `Bearer ${TRADIER_API_KEY}`,
    'Accept': 'application/json',
  },
});

export interface TechnicalIndicators {
  symbol: string;
  currentPrice: number;
  rsi: number;
  sma_20: number;
  sma_50: number;
  sma_200: number;
  ema_12: number;
  ema_26: number;
  macd: {
    value: number;
    signal: number;
    histogram: number;
  };
  bollingerBands: {
    upper: number;
    middle: number;
    lower: number;
    width: number;
    percentB: number; // Where price is within the bands (0-1)
  };
  supportResistance: {
    resistance: number;
    support: number;
    distanceToResistance: number; // percentage
    distanceToSupport: number; // percentage
  };
  trendSignals: {
    rsiSignal: 'oversold' | 'overbought' | 'neutral';
    macdSignal: 'bullish' | 'bearish' | 'neutral';
    smaSignal: 'bullish' | 'bearish' | 'neutral';
    bollingerSignal: 'oversold' | 'overbought' | 'squeeze' | 'neutral';
  };
  momentum: number; // -1 to 1
  trend: 'bullish' | 'bearish' | 'neutral';
}

interface HistoricalDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Fetches historical price data from Tradier
 */
export async function getHistoricalData(symbol: string, interval: string = 'daily', lookbackDays: number = 200, date?: string): Promise<HistoricalDataPoint[]> {
  try {
    // For minute-level intervals, use the timesales endpoint
    if (interval.endsWith('min')) {
      const intervalMinutes = parseInt(interval.replace('min', '')) || 1;
      const allBars: HistoricalDataPoint[] = [];

      // Calculate date range
      const endDate = date ? new Date(date) : new Date();
      const startDate = new Date(endDate);
      startDate.setDate(startDate.getDate() - lookbackDays);

      // Fetch timesales data for each day
      const currentDate = new Date(startDate);
      while (currentDate <= endDate) {
        // Skip weekends
        const dayOfWeek = currentDate.getDay();
        if (dayOfWeek !== 0 && dayOfWeek !== 6) {
          const dateStr = currentDate.toISOString().split('T')[0];
          try {
            const dayBars = await getHistoricalTimesales(symbol, dateStr, intervalMinutes);
            const convertedBars = dayBars.map(bar => ({
              date: bar.timestamp,
              open: bar.open,
              high: bar.high,
              low: bar.low,
              close: bar.close,
              volume: bar.volume,
            }));
            allBars.push(...convertedBars);
          } catch (error) {
            // Skip days with no data (e.g., holidays, market closed)
            console.warn(`No data for ${symbol} on ${dateStr}`);
          }
        }
        currentDate.setDate(currentDate.getDate() + 1);
      }

      return allBars;
    }

    // For daily/weekly/monthly intervals, use the /markets/history endpoint
    const endDate = date ? new Date(date) : new Date();
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - lookbackDays);

    const response = await tradierClient.get('/markets/history', {
      params: {
        symbol,
        interval,
        start: startDate.toISOString().split('T')[0],
        end: endDate.toISOString().split('T')[0],
      },
    });

    const history = response.data.history;
    if (!history || !history.day) {
      console.warn(`No historical data for ${symbol}`);
      return [];
    }

    // Normalize response (can be array or single object)
    const historicalDays = Array.isArray(history.day) ? history.day : [history.day];

    return historicalDays.map((d: any) => ({
      date: d.date,
      open: parseFloat(d.open),
      high: parseFloat(d.high),
      low: parseFloat(d.low),
      close: parseFloat(d.close),
      volume: parseInt(d.volume),
    }));
  } catch (error) {
    console.error(`Error fetching historical data for ${symbol}:`, error);
    return [];
  }
}

/**
 * Calculate support and resistance levels using pivot points and local extremes
 */
function calculateSupportResistance(
  highs: number[],
  lows: number[],
  closes: number[],
  currentPrice: number
): TechnicalIndicators['supportResistance'] {
  // Use last 20 days for recent support/resistance
  const recentPeriod = 20;
  const recentHighs = highs.slice(-recentPeriod);
  const recentLows = lows.slice(-recentPeriod);

  // Find local peaks (resistance) and valleys (support)
  const resistance = Math.max(...recentHighs);
  const support = Math.min(...recentLows);

  const distanceToResistance = ((resistance - currentPrice) / currentPrice) * 100;
  const distanceToSupport = ((currentPrice - support) / currentPrice) * 100;

  return {
    resistance,
    support,
    distanceToResistance,
    distanceToSupport,
  };
}

/**
 * Determine trend signals from indicators
 */
function analyzeTrendSignals(
  rsi: number,
  macdValue: number,
  macdSignal: number,
  currentPrice: number,
  sma20: number,
  sma50: number,
  sma200: number,
  bbUpper: number,
  bbLower: number,
  bbWidth: number
): TechnicalIndicators['trendSignals'] {
  // RSI Signal
  let rsiSignal: 'oversold' | 'overbought' | 'neutral' = 'neutral';
  if (rsi < 30) rsiSignal = 'oversold'; // Bullish reversal potential
  if (rsi > 70) rsiSignal = 'overbought'; // Bearish reversal potential

  // MACD Signal (histogram crossover)
  const macdHistogram = macdValue - macdSignal;
  let macdSignalType: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (macdHistogram > 0) macdSignalType = 'bullish';
  if (macdHistogram < 0) macdSignalType = 'bearish';

  // SMA Trend (price above/below moving averages)
  let smaSignal: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (currentPrice > sma20 && currentPrice > sma50 && sma50 > sma200) {
    smaSignal = 'bullish'; // Golden cross setup
  } else if (currentPrice < sma20 && currentPrice < sma50 && sma50 < sma200) {
    smaSignal = 'bearish'; // Death cross setup
  }

  // Bollinger Band Signal
  let bollingerSignal: 'oversold' | 'overbought' | 'squeeze' | 'neutral' = 'neutral';
  if (currentPrice <= bbLower) bollingerSignal = 'oversold'; // Potential bounce
  if (currentPrice >= bbUpper) bollingerSignal = 'overbought'; // Potential pullback
  if (bbWidth < (currentPrice * 0.05)) bollingerSignal = 'squeeze'; // Volatility compression

  return {
    rsiSignal,
    macdSignal: macdSignalType,
    smaSignal,
    bollingerSignal,
  };
}

/**
 * Calculate overall momentum score (-1 to 1)
 */
function calculateMomentum(
  rsi: number,
  macdHistogram: number,
  priceVsSMA: number
): number {
  // RSI contribution (-1 to 1)
  const rsiScore = (rsi - 50) / 50; // 0 = neutral, -1 = oversold, 1 = overbought

  // MACD contribution (normalized)
  const macdScore = Math.tanh(macdHistogram); // Squash to -1 to 1

  // Price vs SMA contribution
  const smaScore = Math.tanh(priceVsSMA / 100); // Normalize percentage difference

  // Weighted average
  const momentum = (rsiScore * 0.3 + macdScore * 0.4 + smaScore * 0.3);

  return Math.max(-1, Math.min(1, momentum));
}

/**
 * Main function to calculate all technical indicators
 */
export async function getTechnicalIndicators(symbol: string): Promise<TechnicalIndicators | null> {
  try {
    // Get historical data (200 days for 200 SMA)
    const history = await getHistoricalData(symbol, 'daily', 200);

    if (history.length < 50) {
      console.warn(`Insufficient historical data for ${symbol} (${history.length} days)`);
      return null;
    }

    // Extract price arrays
    const closes = history.map(d => d.close);
    const highs = history.map(d => d.high);
    const lows = history.map(d => d.low);
    const currentPrice = closes[closes.length - 1];

    // Calculate RSI (14-period)
    const rsiValues = RSI.calculate({ values: closes, period: 14 });
    const rsi = rsiValues[rsiValues.length - 1] || 50;

    // Calculate SMAs
    const sma20Values = SMA.calculate({ values: closes, period: 20 });
    const sma50Values = SMA.calculate({ values: closes, period: 50 });
    const sma200Values = SMA.calculate({ values: closes, period: 200 });

    const sma_20 = sma20Values[sma20Values.length - 1] || currentPrice;
    const sma_50 = sma50Values[sma50Values.length - 1] || currentPrice;
    const sma_200 = sma200Values[sma200Values.length - 1] || currentPrice;

    // Calculate EMAs (12 and 26 for MACD)
    const ema12Values = EMA.calculate({ values: closes, period: 12 });
    const ema26Values = EMA.calculate({ values: closes, period: 26 });

    const ema_12 = ema12Values[ema12Values.length - 1] || currentPrice;
    const ema_26 = ema26Values[ema26Values.length - 1] || currentPrice;

    // Calculate MACD
    const macdValues = MACD.calculate({
      values: closes,
      fastPeriod: 12,
      slowPeriod: 26,
      signalPeriod: 9,
      SimpleMAOscillator: false,
      SimpleMASignal: false,
    });

    const latestMACD = macdValues[macdValues.length - 1] || { MACD: 0, signal: 0, histogram: 0 };
    const macd = {
      value: latestMACD.MACD || 0,
      signal: latestMACD.signal || 0,
      histogram: latestMACD.histogram || 0,
    };

    // Calculate Bollinger Bands (20-period, 2 std dev)
    const bbValues = BollingerBands.calculate({
      values: closes,
      period: 20,
      stdDev: 2,
    });

    const latestBB = bbValues[bbValues.length - 1] || { upper: currentPrice, middle: currentPrice, lower: currentPrice };
    const bbWidth = latestBB.upper - latestBB.lower;
    const percentB = (currentPrice - latestBB.lower) / (bbWidth || 1); // 0 = at lower band, 1 = at upper band

    const bollingerBands = {
      upper: latestBB.upper,
      middle: latestBB.middle,
      lower: latestBB.lower,
      width: bbWidth,
      percentB,
    };

    // Calculate support and resistance
    const supportResistance = calculateSupportResistance(highs, lows, closes, currentPrice);

    // Analyze trend signals
    const trendSignals = analyzeTrendSignals(
      rsi,
      macd.value,
      macd.signal,
      currentPrice,
      sma_20,
      sma_50,
      sma_200,
      bollingerBands.upper,
      bollingerBands.lower,
      bbWidth
    );

    // Calculate momentum
    const priceVsSMA = ((currentPrice - sma_50) / sma_50) * 100;
    const momentum = calculateMomentum(rsi, macd.histogram, priceVsSMA);

    // Determine overall trend
    let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    const bullishCount = [
      trendSignals.rsiSignal === 'oversold' ? 1 : 0,
      trendSignals.macdSignal === 'bullish' ? 1 : 0,
      trendSignals.smaSignal === 'bullish' ? 1 : 0,
    ].reduce((a, b) => a + b, 0);

    const bearishCount = [
      trendSignals.rsiSignal === 'overbought' ? 1 : 0,
      trendSignals.macdSignal === 'bearish' ? 1 : 0,
      trendSignals.smaSignal === 'bearish' ? 1 : 0,
    ].reduce((a, b) => a + b, 0);

    if (bullishCount >= 2) trend = 'bullish';
    if (bearishCount >= 2) trend = 'bearish';

    return {
      symbol,
      currentPrice,
      rsi,
      sma_20,
      sma_50,
      sma_200,
      ema_12,
      ema_26,
      macd,
      bollingerBands,
      supportResistance,
      trendSignals,
      momentum,
      trend,
    };
  } catch (error) {
    console.error(`Error calculating technical indicators for ${symbol}:`, error);
    return null;
  }
}

/**
 * Calculate technical score (1-10) for a symbol
 */
export function calculateTechnicalScore(indicators: TechnicalIndicators): {
  bullishScore: number;
  bearishScore: number;
  importance: number;
  factors: string[];
} {
  const factors: string[] = [];
  let bullishPoints = 0;
  let bearishPoints = 0;

  // RSI Analysis (weight: 2)
  if (indicators.trendSignals.rsiSignal === 'oversold') {
    bullishPoints += 2;
    factors.push(`RSI oversold at ${indicators.rsi.toFixed(1)} (bullish reversal)`);
  } else if (indicators.trendSignals.rsiSignal === 'overbought') {
    bearishPoints += 2;
    factors.push(`RSI overbought at ${indicators.rsi.toFixed(1)} (bearish reversal)`);
  }

  // MACD Analysis (weight: 3)
  if (indicators.trendSignals.macdSignal === 'bullish') {
    bullishPoints += 3;
    factors.push(`MACD bullish crossover (histogram: ${indicators.macd.histogram.toFixed(2)})`);
  } else if (indicators.trendSignals.macdSignal === 'bearish') {
    bearishPoints += 3;
    factors.push(`MACD bearish crossover (histogram: ${indicators.macd.histogram.toFixed(2)})`);
  }

  // SMA Trend Analysis (weight: 3)
  if (indicators.trendSignals.smaSignal === 'bullish') {
    bullishPoints += 3;
    factors.push(`Price above key moving averages (bullish trend)`);
  } else if (indicators.trendSignals.smaSignal === 'bearish') {
    bearishPoints += 3;
    factors.push(`Price below key moving averages (bearish trend)`);
  }

  // Bollinger Bands Analysis (weight: 2)
  if (indicators.trendSignals.bollingerSignal === 'oversold') {
    bullishPoints += 2;
    factors.push(`Price at lower Bollinger Band (bounce potential)`);
  } else if (indicators.trendSignals.bollingerSignal === 'overbought') {
    bearishPoints += 2;
    factors.push(`Price at upper Bollinger Band (pullback risk)`);
  } else if (indicators.trendSignals.bollingerSignal === 'squeeze') {
    factors.push(`Bollinger Band squeeze detected (breakout pending)`);
  }

  // Support/Resistance Analysis (weight: 2)
  if (indicators.supportResistance.distanceToSupport < 2) {
    bullishPoints += 2;
    factors.push(`Near support level at $${indicators.supportResistance.support.toFixed(2)}`);
  }
  if (indicators.supportResistance.distanceToResistance < 2) {
    bearishPoints += 1;
    factors.push(`Near resistance at $${indicators.supportResistance.resistance.toFixed(2)}`);
  }

  // Normalize to 1-10 scale
  const maxPoints = 10;
  const bullishScore = Math.min(10, Math.round((bullishPoints / maxPoints) * 10));
  const bearishScore = Math.min(10, Math.round((bearishPoints / maxPoints) * 10));

  // Importance based on signal strength
  const importance = Math.min(10, Math.max(bullishPoints, bearishPoints));

  return {
    bullishScore,
    bearishScore,
    importance,
    factors,
  };
}
