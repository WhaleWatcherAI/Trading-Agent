import { OptionsTrade, InstitutionalTrade, NewsItem, MarketData, TradeSignal, BullBearSignal } from '@/types';
import { analyzeOptionFlowSentiment } from './unusualwhales';

interface DataPoint {
  timestamp: Date;
  value: number;
}

/**
 * Creates an importance curve that weights recent data more heavily
 * Uses exponential decay with recency bias
 */
export function calculateImportanceCurve(dataPoints: DataPoint[]): Map<string, number> {
  const now = Date.now();
  const importanceMap = new Map<string, number>();

  // Sort by timestamp descending (most recent first)
  const sorted = [...dataPoints].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  sorted.forEach((point, index) => {
    const ageMs = now - point.timestamp.getTime();
    const ageHours = ageMs / (1000 * 60 * 60);

    // Exponential decay: more recent = higher importance
    // Half-life of 2 hours for intraday
    const decayFactor = Math.exp(-ageHours / 2);

    // Recency bonus for the most recent items
    const recencyBonus = index < 3 ? 0.2 : 0;

    const importance = Math.min(1.0, decayFactor + recencyBonus);

    importanceMap.set(point.timestamp.toISOString(), importance);
  });

  return importanceMap;
}

/**
 * Analyzes news sentiment and calculates aggregate impact
 */
export function analyzeNewsSentiment(news: NewsItem[], symbol: string): {
  sentiment: BullBearSignal;
  score: number;
  importance: number;
} {
  // Filter news for the symbol
  const relevantNews = news.filter(n => n.symbols.includes(symbol));

  if (relevantNews.length === 0) {
    return { sentiment: 'neutral', score: 0, importance: 0 };
  }

  // Calculate importance curve
  const dataPoints = relevantNews.map(n => ({
    timestamp: n.timestamp,
    value: n.importance,
  }));

  const importanceCurve = calculateImportanceCurve(dataPoints);

  let bullishScore = 0;
  let bearishScore = 0;
  let totalWeight = 0;

  relevantNews.forEach(newsItem => {
    const importance = importanceCurve.get(newsItem.timestamp.toISOString()) || 0.5;
    const weight = newsItem.importance * importance;

    if (newsItem.sentiment === 'bullish') {
      bullishScore += weight;
    } else if (newsItem.sentiment === 'bearish') {
      bearishScore += weight;
    }

    totalWeight += weight;
  });

  const netScore = (bullishScore - bearishScore) / (totalWeight || 1);
  const sentiment: BullBearSignal = netScore > 0.1 ? 'bull' : netScore < -0.1 ? 'bear' : 'neutral';

  return {
    sentiment,
    score: netScore,
    importance: totalWeight / relevantNews.length,
  };
}

/**
 * Analyzes institutional activity for a symbol
 */
export function analyzeInstitutionalActivity(
  trades: InstitutionalTrade[],
  symbol: string
): {
  signal: BullBearSignal;
  score: number;
  totalValue: number;
} {
  const relevantTrades = trades.filter(t => t.symbol === symbol);

  if (relevantTrades.length === 0) {
    return { signal: 'neutral', score: 0, totalValue: 0 };
  }

  let buyValue = 0;
  let sellValue = 0;

  relevantTrades.forEach(trade => {
    if (trade.side === 'buy') {
      buyValue += trade.value;
    } else {
      sellValue += trade.value;
    }
  });

  const totalValue = buyValue + sellValue;
  const netValue = buyValue - sellValue;
  const score = netValue / (totalValue || 1);

  const signal: BullBearSignal = score > 0.2 ? 'bull' : score < -0.2 ? 'bear' : 'neutral';

  return { signal, score, totalValue };
}

/**
 * Analyzes options flow for a symbol
 */
export function analyzeOptionsFlow(
  optionsTrades: OptionsTrade[],
  symbol: string
): {
  signal: BullBearSignal;
  score: number;
  unusualActivity: boolean;
} {
  const relevantTrades = optionsTrades.filter(t => t.underlying === symbol && t.side !== 'mid');

  if (relevantTrades.length === 0) {
    return { signal: 'neutral', score: 0, unusualActivity: false };
  }

  let bullishScore = 0;
  let bearishScore = 0;
  let hasUnusual = false;

  relevantTrades.forEach(trade => {
    const sentiment = analyzeOptionFlowSentiment(trade);

    if (sentiment.signal === 'bull') {
      bullishScore += sentiment.strength * (trade.volume * trade.premium);
    } else if (sentiment.signal === 'bear') {
      bearishScore += sentiment.strength * (trade.volume * trade.premium);
    }

    if (trade.unusual) {
      hasUnusual = true;
    }
  });

  const totalScore = bullishScore + bearishScore;
  const netScore = (bullishScore - bearishScore) / (totalScore || 1);

  const signal: BullBearSignal = netScore > 0.15 ? 'bull' : netScore < -0.15 ? 'bear' : 'neutral';

  return {
    signal,
    score: netScore,
    unusualActivity: hasUnusual,
  };
}

/**
 * Determines market tide from market data
 */
export function analyzeMarketTide(marketData: MarketData): BullBearSignal {
  // Use put/call ratio and VIX to determine market sentiment
  // High put/call ratio (>1.2) = bearish
  // Low put/call ratio (<0.8) = bullish
  // High VIX (>25) = fear/bearish
  // Low VIX (<15) = complacency/bullish

  let bullishFactors = 0;
  let bearishFactors = 0;

  if (marketData.putCallRatio < 0.8) bullishFactors++;
  if (marketData.putCallRatio > 1.2) bearishFactors++;

  // VIX interpretation (inverse relationship with market)
  const vixLevel = marketData.vix || 20;
  if (vixLevel < 15) bullishFactors++;
  if (vixLevel > 25) bearishFactors++;

  if (marketData.marketTide === 'bullish') bullishFactors += 2;
  if (marketData.marketTide === 'bearish') bearishFactors += 2;

  if (bullishFactors > bearishFactors) return 'bull';
  if (bearishFactors > bullishFactors) return 'bear';
  return 'neutral';
}

/**
 * Calculates composite trade score (1-10 scale)
 * Negative for bearish, positive for bullish
 */
export function calculateTradeScore(
  newsAnalysis: ReturnType<typeof analyzeNewsSentiment>,
  institutionalAnalysis: ReturnType<typeof analyzeInstitutionalActivity>,
  optionsAnalysis: ReturnType<typeof analyzeOptionsFlow>,
  marketTide: BullBearSignal,
  putCallRatio: number
): {
  rating: number;
  confidence: number;
  factors: {
    newsImpact: number;
    institutionalActivity: number;
    optionsFlow: number;
    marketTide: number;
    technicals: number;
  };
} {
  // Weight factors
  const NEWS_WEIGHT = 0.30;
  const INSTITUTIONAL_WEIGHT = 0.25;
  const OPTIONS_WEIGHT = 0.25;
  const MARKET_TIDE_WEIGHT = 0.15;
  const TECHNICALS_WEIGHT = 0.05;

  // Convert signals to numeric scores
  const signalToScore = (signal: BullBearSignal) => {
    if (signal === 'bull') return 1;
    if (signal === 'bear') return -1;
    return 0;
  };

  const newsScore = newsAnalysis.score * newsAnalysis.importance;
  const instScore = institutionalAnalysis.score;
  const optionsScore = optionsAnalysis.score;
  const marketScore = signalToScore(marketTide);
  const technicalScore = putCallRatio < 1 ? 0.5 : putCallRatio > 1 ? -0.5 : 0;

  // Composite score (-1 to 1)
  const compositeScore =
    newsScore * NEWS_WEIGHT +
    instScore * INSTITUTIONAL_WEIGHT +
    optionsScore * OPTIONS_WEIGHT +
    marketScore * MARKET_TIDE_WEIGHT +
    technicalScore * TECHNICALS_WEIGHT;

  // Convert to 1-10 scale (bearish = negative)
  // -1.0 = -10, 0 = 0, 1.0 = 10
  const rating = Math.round(compositeScore * 10);

  // Calculate confidence based on agreement between factors
  const factors = [
    Math.sign(newsScore),
    Math.sign(instScore),
    Math.sign(optionsScore),
    Math.sign(marketScore),
  ];

  const agreement = factors.filter(f => f === Math.sign(compositeScore)).length / factors.length;
  const confidence = Math.round(agreement * 100);

  return {
    rating,
    confidence,
    factors: {
      newsImpact: newsScore,
      institutionalActivity: instScore,
      optionsFlow: optionsScore,
      marketTide: marketScore,
      technicals: technicalScore,
    },
  };
}
