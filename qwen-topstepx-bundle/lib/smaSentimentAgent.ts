import { SMA } from 'technicalindicators';
import { getOptionsChain } from './tradier';
import { getWhaleFlowAlerts, getPutCallRatio } from './unusualwhales';
import { getHistoricalTimesales } from './tradier';
import { TradeSignal, OptionsTrade } from '@/types';
import { getCached, setCache } from './dataCache';

interface SmaSentimentStrategyOptions {
  symbol: string;
  smaPeriod?: number;
  date?: string;
}

interface HistoricalDataPoint {
  date: string;
  time: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface SentimentAnalysis {
  overallSentiment: 'bullish' | 'bearish' | 'neutral';
  dailyCallPutRatio: number;
  flowSentiment: 'bullish' | 'bearish' | 'neutral';
  aggregatedSentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  reasoning: string[];
}

const HISTORICAL_DATA_CACHE_BUDGET = 5 * 60 * 1000; // 5 minutes
const OPTIONS_CHAIN_CACHE_BUDGET = 10 * 60 * 1000; // 10 minutes
const SENTIMENT_CACHE_BUDGET = 10 * 60 * 1000; // 10 minutes (poll every 10 min)

/**
 * Analyze sentiment from options flow data
 * Looks at call/put ratios and aggregated ask/bid data to determine who's winning
 */
async function analyzeSentiment(symbol: string, date?: string): Promise<SentimentAnalysis> {
  const cacheKey = `sentiment_${symbol}_${date || 'live'}`;
  const cached = getCached<SentimentAnalysis>(cacheKey);

  if (cached?.data) {
    return cached.data;
  }

  const reasoning: string[] = [];

  // 1. Get the day's official call/put ratio for the ticker
  const dailyCallPutRatio = await getPutCallRatio(symbol);
  reasoning.push(`Daily Call/Put Ratio: ${dailyCallPutRatio.toFixed(2)}`);

  // 2. Get options flow data from last 10 minutes
  const flowAlerts = await getWhaleFlowAlerts({
    symbols: [symbol],
    lookbackMinutes: 10,
    minPremium: 50000, // Lower threshold to get more data
    date,
  });

  // 3. Aggregate sentiment from options flow (call/put at ask/bid)
  let bullishFlow = 0;
  let bearishFlow = 0;
  let totalFlowValue = 0;

  flowAlerts.forEach(alert => {
    const value = alert.premium;
    totalFlowValue += value;

    // Determine sentiment based on option type and direction
    // Calls at ask = bullish, Puts at ask = bearish
    // Calls at bid = bearish, Puts at bid = bearish
    if (alert.optionType === 'call') {
      if (alert.direction === 'bullish') {
        bullishFlow += value;
        reasoning.push(`Bullish call flow: $${(value / 1000).toFixed(0)}k`);
      } else if (alert.direction === 'bearish') {
        bearishFlow += value;
        reasoning.push(`Bearish call flow: $${(value / 1000).toFixed(0)}k`);
      }
    } else if (alert.optionType === 'put') {
      if (alert.direction === 'bullish') {
        bearishFlow += value;
        reasoning.push(`Bearish put flow: $${(value / 1000).toFixed(0)}k`);
      } else if (alert.direction === 'bearish') {
        bullishFlow += value;
        reasoning.push(`Bullish put flow: $${(value / 1000).toFixed(0)}k`);
      }
    }
  });

  // Calculate flow sentiment
  let flowSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (totalFlowValue > 0) {
    const bullishPercentage = bullishFlow / totalFlowValue;
    const bearishPercentage = bearishFlow / totalFlowValue;

    if (bullishPercentage > 0.6) {
      flowSentiment = 'bullish';
    } else if (bearishPercentage > 0.6) {
      flowSentiment = 'bearish';
    }

    reasoning.push(`Flow Sentiment: ${flowSentiment} (${(bullishPercentage * 100).toFixed(0)}% bullish, ${(bearishPercentage * 100).toFixed(0)}% bearish)`);
  } else {
    reasoning.push('No significant flow data available');
  }

  // Determine daily sentiment from call/put ratio
  // Lower ratio (< 0.7) = more calls = bullish
  // Higher ratio (> 1.3) = more puts = bearish
  let aggregatedSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (dailyCallPutRatio < 0.7) {
    aggregatedSentiment = 'bullish';
    reasoning.push('Daily ratio indicates bullish sentiment (more calls)');
  } else if (dailyCallPutRatio > 1.3) {
    aggregatedSentiment = 'bearish';
    reasoning.push('Daily ratio indicates bearish sentiment (more puts)');
  } else {
    reasoning.push('Daily ratio is neutral');
  }

  // Combine both sentiments
  let overallSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  let confidence = 0.5;

  if (flowSentiment === 'bullish' && aggregatedSentiment === 'bullish') {
    overallSentiment = 'bullish';
    confidence = 0.9;
    reasoning.push('✓ STRONG BULLISH: Both flow and daily ratio agree');
  } else if (flowSentiment === 'bearish' && aggregatedSentiment === 'bearish') {
    overallSentiment = 'bearish';
    confidence = 0.9;
    reasoning.push('✓ STRONG BEARISH: Both flow and daily ratio agree');
  } else if (flowSentiment === 'bullish' || aggregatedSentiment === 'bullish') {
    overallSentiment = 'bullish';
    confidence = 0.6;
    reasoning.push('Moderate bullish: One indicator is bullish');
  } else if (flowSentiment === 'bearish' || aggregatedSentiment === 'bearish') {
    overallSentiment = 'bearish';
    confidence = 0.6;
    reasoning.push('Moderate bearish: One indicator is bearish');
  } else {
    reasoning.push('✗ NEUTRAL: No clear directional bias');
  }

  const analysis: SentimentAnalysis = {
    overallSentiment,
    dailyCallPutRatio,
    flowSentiment,
    aggregatedSentiment,
    confidence,
    reasoning,
  };

  setCache(cacheKey, analysis, 'sma_sentiment_agent', SENTIMENT_CACHE_BUDGET);

  return analysis;
}

export async function runSmaSentimentStrategy(options: SmaSentimentStrategyOptions): Promise<TradeSignal | null> {
  const { symbol, smaPeriod = 20, date } = options;

  // Get 5-minute candle data
  const historyCacheKey = `sma_sentiment_history_${symbol}_${date || 'live'}`;
  let history: HistoricalDataPoint[] | null = getCached<HistoricalDataPoint[]>(historyCacheKey)?.data || null;

  if (!history) {
    if (date) {
      // Get historical 5-minute data for the specific date
      const bars = await getHistoricalTimesales(symbol, date, 5, 'all');
      history = bars.map(bar => ({
        date: bar.timestamp.split(' ')[0],
        time: bar.time,
        timestamp: bar.timestamp,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume,
      }));
    } else {
      // For live trading, we'd need to use a different data source or accumulate 5-min bars
      // For now, we'll throw an error
      throw new Error('Live 5-minute data not yet implemented. Please provide a date for backtesting.');
    }

    if (history && history.length > 0) {
      setCache(historyCacheKey, history, 'sma_sentiment_agent', HISTORICAL_DATA_CACHE_BUDGET);
    }
  }

  if (!history || history.length < smaPeriod + 1) {
    console.warn(`Insufficient historical data for ${symbol} to calculate SMA(${smaPeriod}). Need ${smaPeriod + 1}, got ${history?.length || 0}`);
    return null;
  }

  // Calculate SMA
  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: smaPeriod });

  if (smaValues.length < 2) {
    console.warn(`Insufficient SMA values for ${symbol}. Need at least 2, got ${smaValues.length}`);
    return null;
  }

  const currentSma = smaValues[smaValues.length - 1];
  const previousSma = smaValues[smaValues.length - 2];

  const currentPrice = closes[closes.length - 1];
  const previousPrice = closes[closes.length - 2];

  // Detect SMA crossover
  const crossesUp = previousPrice <= previousSma && currentPrice > currentSma;
  const crossesDown = previousPrice >= previousSma && currentPrice < currentSma;

  if (!crossesUp && !crossesDown) {
    return null;
  }

  const direction = crossesUp ? 'bullish' : 'bearish';

  // Analyze sentiment
  const sentiment = await analyzeSentiment(symbol, date);

  console.log(`\n📊 SMA Crossover detected for ${symbol}: ${direction}`);
  console.log(`💭 Sentiment Analysis: ${sentiment.overallSentiment} (confidence: ${(sentiment.confidence * 100).toFixed(0)}%)`);
  console.log(`   ${sentiment.reasoning.join('\n   ')}`);

  // Only take trades when sentiment aligns with SMA direction
  if (direction === 'bullish' && sentiment.overallSentiment !== 'bullish') {
    console.log(`✗ Skipping bullish trade - sentiment is ${sentiment.overallSentiment}`);
    return null;
  }

  if (direction === 'bearish' && sentiment.overallSentiment !== 'bearish') {
    console.log(`✗ Skipping bearish trade - sentiment is ${sentiment.overallSentiment}`);
    return null;
  }

  console.log(`✓ Trade approved - sentiment aligns with direction`);

  // Get options chain
  const optionType = crossesUp ? 'call' : 'put';
  const optionsChainCacheKey = `sma_sentiment_options_${symbol}_${date || 'live'}`;
  let optionsChain: OptionsTrade[] | null = getCached<OptionsTrade[]>(optionsChainCacheKey)?.data || null;

  if (!optionsChain) {
    optionsChain = await getOptionsChain(symbol, undefined, date);
    if (optionsChain && optionsChain.length > 0) {
      setCache(optionsChainCacheKey, optionsChain, 'sma_sentiment_agent', OPTIONS_CHAIN_CACHE_BUDGET);
    }
  }

  if (!optionsChain || optionsChain.length === 0) {
    console.warn(`No options chain for ${symbol}`);
    return null;
  }

  // Find best option (closest ITM)
  const inTheMoneyOptions = optionsChain.filter(o => {
    if (o.type !== optionType) return false;
    if (optionType === 'call') {
      return o.strike < currentPrice;
    } else { // put
      return o.strike > currentPrice;
    }
  });

  if (inTheMoneyOptions.length === 0) {
    console.warn(`No in-the-money ${optionType} options found for ${symbol}`);
    return null;
  }

  // Sort by strike price to find the first ITM option
  inTheMoneyOptions.sort((a, b) => {
    if (optionType === 'call') {
      return b.strike - a.strike; // Closest to the money (highest strike)
    } else { // put
      return a.strike - b.strike; // Closest to the money (lowest strike)
    }
  });

  const bestOption = inTheMoneyOptions[0];

  const signal: TradeSignal = {
    symbol: bestOption.symbol,
    underlying: symbol,
    contract: `${symbol} ${bestOption.expiration} ${bestOption.strike}${optionType.toUpperCase()}`,
    strike: bestOption.strike,
    expiration: bestOption.expiration,
    type: optionType,
    action: 'buy',
    strategy: 'sma_sentiment',
    currentPrice: bestOption.premium,
    rating: Math.round(sentiment.confidence * 10),
    confidence: sentiment.confidence,
    reasoning: `Price crossed ${direction === 'bullish' ? 'above' : 'below'} the ${smaPeriod}-period SMA with ${sentiment.overallSentiment} sentiment.`,
    factors: [
      `Price: ${currentPrice.toFixed(2)}`,
      `SMA(${smaPeriod}): ${currentSma.toFixed(2)}`,
      `Direction: ${direction}`,
      `Sentiment: ${sentiment.overallSentiment} (${(sentiment.confidence * 100).toFixed(0)}% confidence)`,
      `Daily P/C Ratio: ${sentiment.dailyCallPutRatio.toFixed(2)}`,
      ...sentiment.reasoning.slice(0, 3), // Include top 3 reasoning points
    ],
    timestamp: new Date(history[history.length - 1].timestamp),
  };

  return signal;
}
