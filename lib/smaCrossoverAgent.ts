
import { SMA } from 'technicalindicators';
import { getOptionsChain } from './tradier';
import { getHistoricalData } from './technicals';
import { TradeSignal, OptionsTrade } from '@/types';
import { getCached, setCache } from './dataCache';

interface SmaCrossoverStrategyOptions {
  symbol: string;
  smaPeriod?: number;
  date?: string; // Optional date for historical analysis
}

interface HistoricalDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const HISTORICAL_DATA_CACHE_BUDGET = 5 * 60 * 1000; // 5 minutes
const OPTIONS_CHAIN_CACHE_BUDGET = 10 * 60 * 1000; // 10 minutes

export async function runSmaCrossoverStrategy(options: SmaCrossoverStrategyOptions): Promise<TradeSignal | null> {
  const { symbol, smaPeriod = 9, date } = options;

  const historyCacheKey = `sma_crossover_history_${symbol}_${date || 'live'}`;
  let history: HistoricalDataPoint[] | null = getCached<HistoricalDataPoint[]>(historyCacheKey)?.data || null;

  if (!history) {
    history = await getHistoricalData(symbol, '1min', date ? 7 : 2, date);
    if (history && history.length > 0) {
      setCache(historyCacheKey, history, 'sma_crossover_agent');
    }
  }

  if (!history || history.length < smaPeriod) {
    console.warn(`Insufficient historical data for ${symbol} to calculate SMA(${smaPeriod})`);
    return null;
  }

  const closes = history.map(d => d.close);
  const smaValues = SMA.calculate({ values: closes, period: smaPeriod });
  const currentSma = smaValues[smaValues.length - 1];
  const previousSma = smaValues[smaValues.length - 2];

  const currentPrice = closes[closes.length - 1];
  const previousPrice = closes[closes.length - 2];

  const crossesUp = previousPrice <= previousSma && currentPrice > currentSma;
  const crossesDown = previousPrice >= previousSma && currentPrice < currentSma;

  if (!crossesUp && !crossesDown) {
    return null;
  }

  const direction = crossesUp ? 'bullish' : 'bearish';
  const optionType = crossesUp ? 'call' : 'put';

  const optionsChainCacheKey = `sma_crossover_options_${symbol}_${date || 'live'}`;
  let optionsChain: OptionsTrade[] | null = getCached<OptionsTrade[]>(optionsChainCacheKey)?.data || null;

  if (!optionsChain) {
    optionsChain = await getOptionsChain(symbol, undefined, date);
    if (optionsChain && optionsChain.length > 0) {
      setCache(optionsChainCacheKey, optionsChain, 'sma_crossover_agent');
    }
  }

  if (!optionsChain || optionsChain.length === 0) {
    console.warn(`No options chain for ${symbol}`);
    return null;
  }

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
    strategy: 'sma_crossover',
    currentPrice: bestOption.premium,
    rating: 7, // Static rating for now
    confidence: 0.7, // Static confidence
    reasoning: `Price crossed ${direction === 'bullish' ? 'above' : 'below'} the ${smaPeriod}-period SMA.`,
    factors: [
        `Price: ${currentPrice.toFixed(2)}`,
        `SMA(${smaPeriod}): ${currentSma.toFixed(2)}`,
        `Direction: ${direction}`
    ],
    timestamp: new Date(),
  };

  return signal;
}
