import { OptionsTrade, InstitutionalTrade, NewsItem, GreekFlow, SectorFlow, SectorTide, SpotGEX, VolatilityStats } from '@/types';
import { TechnicalIndicators } from './technicals';

export type Strategy = 'scalp' | 'intraday' | 'swing' | 'leap';

// TESTING MODE: Set to true to bypass time decay (useful when markets are closed)
// GPT NOTE: When BYPASS_TIME_DECAY is true, all data is treated as fresh regardless of age
// This allows testing with older data but should be FALSE in production
export const BYPASS_TIME_DECAY = process.env.BYPASS_TIME_DECAY === 'true';

export interface DataPoint {
  timestamp: Date;
  value: number;
  magnitude: number; // Percentile: 0-1 (0.95 = 95th percentile)
  type: 'flow' | 'news' | 'institutional' | 'structure' | 'technical';
}

export interface StrategyConfig {
  name: Strategy;
  halfLife: number; // minutes
  weights: {
    flow: number;
    news: number;
    institutional: number;
    structure: number; // Greek flow, GEX, volatility
    technical: number;
  };
}

// Strategy-specific configurations
export const STRATEGY_CONFIGS: Record<Strategy, StrategyConfig> = {
  scalp: {
    name: 'scalp',
    halfLife: 10, // 10 minutes
    weights: {
      flow: 0.40,      // Options flow is critical for scalping
      news: 0.30,      // Breaking news moves fast
      structure: 0.20, // GEX and gamma matter
      institutional: 0.05,
      technical: 0.05,
    },
  },
  intraday: {
    name: 'intraday',
    halfLife: 30, // 30 minutes
    weights: {
      flow: 0.35,
      news: 0.25,
      structure: 0.25,
      institutional: 0.10,
      technical: 0.05,
    },
  },
  swing: {
    name: 'swing',
    halfLife: 60, // 1 hour
    weights: {
      news: 0.30,
      flow: 0.25,
      structure: 0.20,
      technical: 0.15,
      institutional: 0.10,
    },
  },
  leap: {
    name: 'leap',
    halfLife: 120, // 2 hours
    weights: {
      structure: 0.30, // Long-term volatility and sector trends
      news: 0.25,      // Major catalysts
      technical: 0.20, // Long-term trends (SMA crossovers)
      institutional: 0.15,
      flow: 0.10,
    },
  },
};

/**
 * Calculate time decay factor using exponential decay
 * Returns value between 0 and 1
 */
export function calculateTimeDecay(timestamp: Date, halfLifeMinutes: number): number {
  // BYPASS MODE: Treat all data as fresh for testing
  if (BYPASS_TIME_DECAY) {
    console.log('⚠️  BYPASS MODE: Time decay disabled - treating all data as fresh');
    return 1.0; // No decay
  }

  const now = Date.now();
  const ageMs = now - timestamp.getTime();
  const ageMinutes = ageMs / (1000 * 60);

  // Exponential decay: e^(-t/halfLife)
  const decayFactor = Math.exp(-ageMinutes / halfLifeMinutes);

  return Math.max(0, Math.min(1, decayFactor));
}

/**
 * Calculate magnitude boost based on data size/importance
 */
export function calculateMagnitudeBoost(magnitude: number): number {
  // magnitude is 0-1 (percentile)
  if (magnitude < 0.25) return 0.8;  // Small events: 80% weight
  if (magnitude < 0.75) return 1.0;  // Normal events: 100% weight
  if (magnitude < 0.95) return 1.3;  // Large events: 130% weight
  return 1.6;                        // Extreme events: 160% weight
}

/**
 * Calculate weighted importance score for a data point
 */
export function calculateImportance(
  dataPoint: DataPoint,
  strategy: Strategy
): number {
  const config = STRATEGY_CONFIGS[strategy];

  // Time decay
  const timeFactor = calculateTimeDecay(dataPoint.timestamp, config.halfLife);

  // Magnitude boost
  const magnitudeBoost = calculateMagnitudeBoost(dataPoint.magnitude);

  // Base importance
  const baseImportance = dataPoint.value;

  // Combined importance
  const importance = baseImportance * timeFactor * magnitudeBoost;

  return Math.max(0, Math.min(10, importance));
}

/**
 * Analyze options flow data and generate data points
 */
export function analyzeFlowData(
  optionsTrades: OptionsTrade[],
  symbol: string,
  strategy: Strategy
): {
  bullishScore: number;
  bearishScore: number;
  factors: string[];
  dataPoints: DataPoint[];
} {
  const relevantTrades = optionsTrades.filter(t => t.underlying === symbol && t.side !== 'mid');

  if (relevantTrades.length === 0) {
    return { bullishScore: 0, bearishScore: 0, factors: [], dataPoints: [] };
  }

  const dataPoints: DataPoint[] = [];
  let bullishWeight = 0;
  let bearishWeight = 0;
  const factors: string[] = [];

  // Calculate total volume for percentile ranking
  const volumes = relevantTrades.map(t => t.volume * t.premium);
  const maxVolume = Math.max(...volumes);

  relevantTrades.forEach(trade => {
    const tradeValue = trade.volume * trade.premium;
    const magnitude = tradeValue / (maxVolume || 1); // Normalize to 0-1

    // Determine if bullish or bearish
    let isBullish = false;
    let isBearish = false;

    if (trade.type === 'call' && trade.side === 'ask') {
      isBullish = true; // Buying calls
    } else if (trade.type === 'put' && trade.side === 'bid') {
      isBullish = true; // Selling puts
    } else if (trade.type === 'put' && trade.side === 'ask') {
      isBearish = true; // Buying puts
    } else if (trade.type === 'call' && trade.side === 'bid') {
      isBearish = true; // Selling calls
    }

    // Calculate importance
    const importance = calculateImportance(
      { timestamp: trade.timestamp, value: 8, magnitude, type: 'flow' },
      strategy
    );

    const dataPoint: DataPoint = {
      timestamp: trade.timestamp,
      value: importance,
      magnitude,
      type: 'flow',
    };

    dataPoints.push(dataPoint);

    if (isBullish) {
      bullishWeight += importance * (trade.unusual ? 1.3 : 1.0);
      if (trade.unusual && magnitude > 0.7) {
        factors.push(
          `Large ${trade.type} sweep at ${trade.side} ($${(tradeValue / 1000000).toFixed(2)}M) ` +
          `${Math.round((Date.now() - trade.timestamp.getTime()) / 60000)}min ago`
        );
      }
    } else if (isBearish) {
      bearishWeight += importance * (trade.unusual ? 1.3 : 1.0);
      if (trade.unusual && magnitude > 0.7) {
        factors.push(
          `Large bearish ${trade.type} at ${trade.side} ($${(tradeValue / 1000000).toFixed(2)}M) ` +
          `${Math.round((Date.now() - trade.timestamp.getTime()) / 60000)}min ago`
        );
      }
    }
  });

  // Normalize to 1-10 scale
  const totalWeight = bullishWeight + bearishWeight;
  const bullishScore = totalWeight > 0 ? Math.min(10, Math.round((bullishWeight / totalWeight) * 10)) : 0;
  const bearishScore = totalWeight > 0 ? Math.min(10, Math.round((bearishWeight / totalWeight) * 10)) : 0;

  return { bullishScore, bearishScore, factors, dataPoints };
}

/**
 * Analyze news sentiment with time decay
 */
export function analyzeNewsData(
  news: NewsItem[],
  symbol: string,
  strategy: Strategy
): {
  bullishScore: number;
  bearishScore: number;
  factors: string[];
  dataPoints: DataPoint[];
} {
  const relevantNews = news.filter(n => n.symbols.includes(symbol));

  if (relevantNews.length === 0) {
    return { bullishScore: 0, bearishScore: 0, factors: [], dataPoints: [] };
  }

  const dataPoints: DataPoint[] = [];
  let bullishWeight = 0;
  let bearishWeight = 0;
  const factors: string[] = [];

  relevantNews.forEach(newsItem => {
    const magnitude = newsItem.importance; // Already 0-1

    const baseImportance = newsItem.importance * 10; // Convert to 1-10 scale
    const importance = calculateImportance(
      { timestamp: newsItem.timestamp, value: baseImportance, magnitude, type: 'news' },
      strategy
    );

    const dataPoint: DataPoint = {
      timestamp: newsItem.timestamp,
      value: importance,
      magnitude,
      type: 'news',
    };

    dataPoints.push(dataPoint);

    const ageMinutes = Math.round((Date.now() - newsItem.timestamp.getTime()) / 60000);

    if (newsItem.sentiment === 'bullish') {
      bullishWeight += importance;
      if (importance > 5 && ageMinutes < 60) {
        factors.push(`Bullish news: "${newsItem.title.substring(0, 80)}..." (${ageMinutes}min ago)`);
      }
    } else if (newsItem.sentiment === 'bearish') {
      bearishWeight += importance;
      if (importance > 5 && ageMinutes < 60) {
        factors.push(`Bearish news: "${newsItem.title.substring(0, 80)}..." (${ageMinutes}min ago)`);
      }
    }
  });

  // Normalize to 1-10 scale
  const totalWeight = bullishWeight + bearishWeight;
  const bullishScore = totalWeight > 0 ? Math.min(10, Math.round((bullishWeight / totalWeight) * 10)) : 5;
  const bearishScore = totalWeight > 0 ? Math.min(10, Math.round((bearishWeight / totalWeight) * 10)) : 5;

  return { bullishScore, bearishScore, factors, dataPoints };
}

/**
 * Analyze institutional activity
 */
export function analyzeInstitutionalData(
  trades: InstitutionalTrade[],
  symbol: string,
  strategy: Strategy
): {
  bullishScore: number;
  bearishScore: number;
  factors: string[];
  dataPoints: DataPoint[];
} {
  const relevantTrades = trades.filter(t => t.symbol === symbol);

  if (relevantTrades.length === 0) {
    return { bullishScore: 0, bearishScore: 0, factors: [], dataPoints: [] };
  }

  const dataPoints: DataPoint[] = [];
  let bullishWeight = 0;
  let bearishWeight = 0;
  const factors: string[] = [];

  const values = relevantTrades.map(t => t.value);
  const maxValue = Math.max(...values);

  relevantTrades.forEach(trade => {
    const magnitude = trade.value / (maxValue || 1);

    const baseImportance = 7; // Institutional trades are important
    const importance = calculateImportance(
      { timestamp: trade.timestamp, value: baseImportance, magnitude, type: 'institutional' },
      strategy
    );

    const dataPoint: DataPoint = {
      timestamp: trade.timestamp,
      value: importance,
      magnitude,
      type: 'institutional',
    };

    dataPoints.push(dataPoint);

    const ageMinutes = Math.round((Date.now() - trade.timestamp.getTime()) / 60000);

    if (trade.side === 'buy') {
      bullishWeight += importance;
      if (trade.value > 10000000 && ageMinutes < 180) {
        factors.push(
          `${trade.institution} bought $${(trade.value / 1000000).toFixed(1)}M (${ageMinutes}min ago)`
        );
      }
    } else {
      bearishWeight += importance;
      if (trade.value > 10000000 && ageMinutes < 180) {
        factors.push(
          `${trade.institution} sold $${(trade.value / 1000000).toFixed(1)}M (${ageMinutes}min ago)`
        );
      }
    }
  });

  // Normalize to 1-10 scale
  const totalWeight = bullishWeight + bearishWeight;
  const bullishScore = totalWeight > 0 ? Math.min(10, Math.round((bullishWeight / totalWeight) * 10)) : 5;
  const bearishScore = totalWeight > 0 ? Math.min(10, Math.round((bearishWeight / totalWeight) * 10)) : 5;

  return { bullishScore, bearishScore, factors, dataPoints };
}

/**
 * Analyze market structure (Greek flow, GEX, volatility)
 */
export function analyzeStructureData(
  greekFlow: GreekFlow[],
  spotGEX: SpotGEX[],
  volStats: VolatilityStats[],
  sectorTide: SectorTide[],
  symbol: string,
  strategy: Strategy
): {
  bullishScore: number;
  bearishScore: number;
  factors: string[];
  dataPoints: DataPoint[];
} {
  const factors: string[] = [];
  const dataPoints: DataPoint[] = [];
  let bullishWeight = 0;
  let bearishWeight = 0;

  // Analyze GEX for the symbol
  const gexData = spotGEX.filter(g => g.ticker === symbol);
  if (gexData.length > 0) {
    const latest = gexData[gexData.length - 1];
    const gammaImpact = latest.gamma_per_one_percent_move_oi;

    const magnitude = Math.min(1, Math.abs(gammaImpact) / 1000000); // Normalize
    const importance = calculateImportance(
      { timestamp: latest.time, value: 6, magnitude, type: 'structure' },
      strategy
    );

    if (gammaImpact > 0) {
      bullishWeight += importance;
      factors.push(`Positive gamma exposure: ${(gammaImpact / 1000000).toFixed(2)}M (supportive)`);
    } else if (gammaImpact < 0) {
      bearishWeight += importance;
      factors.push(`Negative gamma exposure: ${(gammaImpact / 1000000).toFixed(2)}M (volatile)`);
    }

    dataPoints.push({
      timestamp: latest.time,
      value: importance,
      magnitude,
      type: 'structure',
    });
  }

  // Analyze volatility stats
  const volData = volStats.find(v => v.ticker === symbol);
  if (volData) {
    const ivRank = volData.iv_rank;

    if (ivRank > 75) {
      factors.push(`High IV rank (${ivRank.toFixed(0)}%) - options expensive`);
      bearishWeight += 2; // Expensive options = bearish for buyers
    } else if (ivRank < 25) {
      factors.push(`Low IV rank (${ivRank.toFixed(0)}%) - options cheap`);
      bullishWeight += 2; // Cheap options = bullish opportunity
    }
  }

  // Analyze sector tide
  // Try to find the symbol's sector from the data
  const relevantTide = sectorTide.find(t => t.sector.includes('Technology') || t.sector.includes('Financial'));
  if (relevantTide) {
    const netFlow = relevantTide.net_call_premium - Math.abs(relevantTide.net_put_premium);

    const magnitude = Math.min(1, Math.abs(netFlow) / 10000000);
    const importance = calculateImportance(
      { timestamp: relevantTide.timestamp, value: 5, magnitude, type: 'structure' },
      strategy
    );

    if (netFlow > 0) {
      bullishWeight += importance * 0.5; // Sector is less direct
      factors.push(`Sector showing bullish flow (net: $${(netFlow / 1000000).toFixed(1)}M)`);
    } else {
      bearishWeight += importance * 0.5;
      factors.push(`Sector showing bearish flow (net: $${(netFlow / 1000000).toFixed(1)}M)`);
    }
  }

  // Normalize to 1-10 scale
  const totalWeight = bullishWeight + bearishWeight;
  const bullishScore = totalWeight > 0 ? Math.min(10, Math.round((bullishWeight / totalWeight) * 10)) : 5;
  const bearishScore = totalWeight > 0 ? Math.min(10, Math.round((bearishWeight / totalWeight) * 10)) : 5;

  return { bullishScore, bearishScore, factors, dataPoints };
}

/**
 * Analyze technical indicators
 */
export function analyzeTechnicalData(
  technicals: TechnicalIndicators | null,
  strategy: Strategy
): {
  bullishScore: number;
  bearishScore: number;
  factors: string[];
} {
  if (!technicals) {
    return { bullishScore: 5, bearishScore: 5, factors: [] };
  }

  const factors: string[] = [];
  let bullishPoints = 0;
  let bearishPoints = 0;

  // RSI
  if (technicals.trendSignals.rsiSignal === 'oversold') {
    bullishPoints += 2;
    factors.push(`RSI oversold at ${technicals.rsi.toFixed(1)}`);
  } else if (technicals.trendSignals.rsiSignal === 'overbought') {
    bearishPoints += 2;
    factors.push(`RSI overbought at ${technicals.rsi.toFixed(1)}`);
  }

  // MACD
  if (technicals.trendSignals.macdSignal === 'bullish') {
    bullishPoints += 2;
    factors.push(`MACD bullish crossover`);
  } else if (technicals.trendSignals.macdSignal === 'bearish') {
    bearishPoints += 2;
    factors.push(`MACD bearish crossover`);
  }

  // SMA Trend
  if (technicals.trendSignals.smaSignal === 'bullish') {
    bullishPoints += 2;
    factors.push(`Price above key MAs (bullish trend)`);
  } else if (technicals.trendSignals.smaSignal === 'bearish') {
    bearishPoints += 2;
    factors.push(`Price below key MAs (bearish trend)`);
  }

  // Support/Resistance
  if (technicals.supportResistance.distanceToSupport < 2) {
    bullishPoints += 1;
    factors.push(`Near support at $${technicals.supportResistance.support.toFixed(2)}`);
  }
  if (technicals.supportResistance.distanceToResistance < 2) {
    bearishPoints += 1;
    factors.push(`Near resistance at $${technicals.supportResistance.resistance.toFixed(2)}`);
  }

  // Normalize to 1-10
  const maxPoints = 7;
  const bullishScore = Math.min(10, Math.round((bullishPoints / maxPoints) * 10));
  const bearishScore = Math.min(10, Math.round((bearishPoints / maxPoints) * 10));

  return { bullishScore, bearishScore, factors };
}

/**
 * Calculate final composite score for a strategy
 */
export function calculateCompositeScore(
  flowAnalysis: ReturnType<typeof analyzeFlowData>,
  newsAnalysis: ReturnType<typeof analyzeNewsData>,
  institutionalAnalysis: ReturnType<typeof analyzeInstitutionalData>,
  structureAnalysis: ReturnType<typeof analyzeStructureData>,
  technicalAnalysis: ReturnType<typeof analyzeTechnicalData>,
  strategy: Strategy
): {
  bullishScore: number;
  bearishScore: number;
  confidence: number;
  dominantDirection: 'bullish' | 'bearish' | 'neutral';
  factorBreakdown: {
    flow: { bullishScore: number; bearishScore: number; weight: number };
    news: { bullishScore: number; bearishScore: number; weight: number };
    institutional: { bullishScore: number; bearishScore: number; weight: number };
    structure: { bullishScore: number; bearishScore: number; weight: number };
    technical: { bullishScore: number; bearishScore: number; weight: number };
  };
} {
  const config = STRATEGY_CONFIGS[strategy];

  // Calculate weighted scores
  const bullishScore =
    (flowAnalysis.bullishScore * config.weights.flow) +
    (newsAnalysis.bullishScore * config.weights.news) +
    (institutionalAnalysis.bullishScore * config.weights.institutional) +
    (structureAnalysis.bullishScore * config.weights.structure) +
    (technicalAnalysis.bullishScore * config.weights.technical);

  const bearishScore =
    (flowAnalysis.bearishScore * config.weights.flow) +
    (newsAnalysis.bearishScore * config.weights.news) +
    (institutionalAnalysis.bearishScore * config.weights.institutional) +
    (structureAnalysis.bearishScore * config.weights.structure) +
    (technicalAnalysis.bearishScore * config.weights.technical);

  // Determine dominant direction
  let dominantDirection: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (bullishScore >= 7 && bullishScore > bearishScore) {
    dominantDirection = 'bullish';
  } else if (bearishScore >= 7 && bearishScore > bullishScore) {
    dominantDirection = 'bearish';
  }

  // Calculate confidence (agreement between factors)
  const factors = [
    flowAnalysis.bullishScore > flowAnalysis.bearishScore ? 1 : -1,
    newsAnalysis.bullishScore > newsAnalysis.bearishScore ? 1 : -1,
    institutionalAnalysis.bullishScore > institutionalAnalysis.bearishScore ? 1 : -1,
    structureAnalysis.bullishScore > structureAnalysis.bearishScore ? 1 : -1,
    technicalAnalysis.bullishScore > technicalAnalysis.bearishScore ? 1 : -1,
  ];

  const dominantSign = bullishScore > bearishScore ? 1 : -1;
  const agreement = factors.filter(f => f === dominantSign).length / factors.length;
  const confidence = Math.round(agreement * 100);

  return {
    bullishScore: Math.round(bullishScore),
    bearishScore: Math.round(bearishScore),
    confidence,
    dominantDirection,
    factorBreakdown: {
      flow: {
        bullishScore: flowAnalysis.bullishScore,
        bearishScore: flowAnalysis.bearishScore,
        weight: config.weights.flow,
      },
      news: {
        bullishScore: newsAnalysis.bullishScore,
        bearishScore: newsAnalysis.bearishScore,
        weight: config.weights.news,
      },
      institutional: {
        bullishScore: institutionalAnalysis.bullishScore,
        bearishScore: institutionalAnalysis.bearishScore,
        weight: config.weights.institutional,
      },
      structure: {
        bullishScore: structureAnalysis.bullishScore,
        bearishScore: structureAnalysis.bearishScore,
        weight: config.weights.structure,
      },
      technical: {
        bullishScore: technicalAnalysis.bullishScore,
        bearishScore: technicalAnalysis.bearishScore,
        weight: config.weights.technical,
      },
    },
  };
}
