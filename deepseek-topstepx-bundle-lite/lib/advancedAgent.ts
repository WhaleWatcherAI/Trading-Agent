import OpenAI from 'openai';
import {
  OptionsTrade,
  InstitutionalTrade,
  NewsItem,
  GreekFlow,
  SpotGEX,
  VolatilityStats,
  MarketData,
} from '@/types';
import {
  getGreekFlow,
  getSpotGEX,
  getVolatilityStats,
  getInstitutionalActivity,
  getNewsHeadlines,
  getUnusualOptionsActivity,
  getPutCallRatio,
} from './unusualwhales';
import { getMarketData } from './tradier';
import { getTechnicalIndicators, TechnicalIndicators } from './technicals';
import {
  Strategy,
  analyzeFlowData,
  analyzeNewsData,
  analyzeInstitutionalData,
  analyzeStructureData,
  analyzeTechnicalData,
  calculateCompositeScore,
} from './scoringEngine';
import {
  getStrategyParams,
  meetsThreshold,
  calculatePositionSize,
  calculateUrgency,
  generateRiskWarnings,
  getStrategyGuidelines,
} from './strategyEngine';
import {
  selectBestContract,
  getStockPrice,
  ContractRecommendation,
} from './contractSelector';
import {
  generateTradeAnalysisPrompt,
  parseAIResponse,
  PromptContext,
} from './aiPrompts';
import {
  formatTradeRecommendation,
  formatMultiStrategyResponse,
  TradeRecommendation,
  MultiStrategyResponse,
} from './responseFormatter';
import {
  getCached,
  setCache,
  getWithStaleness,
  STALENESS_CONFIG,
} from './dataCache';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface AnalysisContext {
  symbol: string;
  optionsTrades: OptionsTrade[];
  institutionalTrades: InstitutionalTrade[];
  news: NewsItem[];
  greekFlow: GreekFlow[];
  spotGEX: SpotGEX[];
  volStats: VolatilityStats[];
  marketData: MarketData;
  technicals: TechnicalIndicators | null;
  currentPrice: number;
}

/**
 * Helper function to add delay between API calls to respect rate limits
 * 120 req/min = ~2 req/sec, so 600ms delay = ~1.67 req/sec (safe margin)
 */
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Aggregate all market data for analysis
 * Uses cache-first approach for fast, non-blocking reads
 */
async function aggregateMarketData(symbols?: string[]): Promise<{
  optionsTrades: OptionsTrade[];
  institutionalTrades: InstitutionalTrade[];
  news: NewsItem[];
  greekFlow: GreekFlow[];
  spotGEX: SpotGEX[];
  volStats: VolatilityStats[];
  marketData: MarketData;
}> {
  console.log('📦 Aggregating market data from cache...');

  // Helper to get cached data or fetch if stale
  async function getCachedOrFetch<T>(
    key: string,
    fetcher: () => Promise<T>,
    fallback: T
  ): Promise<T> {
    const cached = getWithStaleness<T>(key, STALENESS_CONFIG[key]?.budget || 60000);

    if (cached.isFresh && cached.data) {
      console.log(`✅ Using cached ${key} (fresh for ${Math.round((STALENESS_CONFIG[key]?.budget - cached.staleFor) / 1000)}s)`);
      return cached.data;
    }

    // Cache miss or stale - fetch fresh data
    console.log(`🔄 Cache ${cached.data ? 'stale' : 'miss'} for ${key}, fetching...`);
    try {
      const data = await fetcher();
      setCache(key, data, 'analyze-api');
      return data;
    } catch (error) {
      console.warn(`⚠️  Failed to fetch ${key}, using ${cached.data ? 'stale cache' : 'fallback'}`);
      return cached.data || fallback;
    }
  }

  // Fetch all data from cache or API (no delays needed - cache handles rate limiting via polling service)
  const [
    optionsTrades,
    institutionalTrades,
    news,
    greekFlow,
    spotGEX,
    volStats,
    marketData,
    putCallRatio,
  ] = await Promise.all([
    getCachedOrFetch('sector_tide', getUnusualOptionsActivity, []),
    getCachedOrFetch('institutional', getInstitutionalActivity, []),
    getCachedOrFetch('news_headlines', getNewsHeadlines, []),
    getCachedOrFetch('greek_flow', getGreekFlow, []),
    getCachedOrFetch('spot_gex', getSpotGEX, []),
    getCachedOrFetch('volatility_stats', getVolatilityStats, []),
    getCachedOrFetch('market_data', getMarketData, { putCallRatio: 1.0, spy: 0, vix: 20 }),
    getCachedOrFetch('put_call_ratio', getPutCallRatio, 1.0),
  ]);

  console.log('✅ All market data aggregated from cache');

  // Determine market tide
  let marketTide: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (marketData.vix < 15 && putCallRatio < 0.8) {
    marketTide = 'bullish';
  } else if (marketData.vix > 25 && putCallRatio > 1.2) {
    marketTide = 'bearish';
  }

  return {
    optionsTrades,
    institutionalTrades,
    news,
    greekFlow,
    spotGEX,
    volStats,
    marketData: {
      ...marketData,
      putCallRatio,
      marketTide,
      timestamp: new Date(),
    },
  };
}

/**
 * Analyze a single symbol for a specific strategy
 */
async function analyzeSymbolForStrategy(
  context: AnalysisContext,
  strategy: Strategy,
  threshold: number = 7
): Promise<TradeRecommendation | null> {
  const { symbol, currentPrice } = context;

  console.log(`Analyzing ${symbol} for ${strategy} strategy...`);

  // Run all analysis modules
  const flowAnalysis = analyzeFlowData(context.optionsTrades, symbol, strategy);
  const newsAnalysis = analyzeNewsData(context.news, symbol, strategy);
  const institutionalAnalysis = analyzeInstitutionalData(context.institutionalTrades, symbol, strategy);
  const structureAnalysis = analyzeStructureData(
    context.greekFlow,
    context.spotGEX,
    context.volStats,
    [], // No sector tide data
    symbol,
    strategy
  );
  const technicalAnalysis = analyzeTechnicalData(context.technicals, strategy);

  // Calculate composite scores
  const compositeScore = calculateCompositeScore(
    flowAnalysis,
    newsAnalysis,
    institutionalAnalysis,
    structureAnalysis,
    technicalAnalysis,
    strategy
  );

  // Check if meets threshold
  const thresholdCheck = meetsThreshold(compositeScore.bullishScore, compositeScore.bearishScore, threshold);

  if (!thresholdCheck.passes) {
    console.log(`${symbol} ${strategy} does not meet threshold (score: ${thresholdCheck.score})`);
    return null;
  }

  const direction = thresholdCheck.direction!;
  console.log(`${symbol} ${strategy} passed threshold: ${direction} (score: ${thresholdCheck.score})`);

  // Select best contract
  const contractRec = await selectBestContract(symbol, currentPrice, direction, strategy);

  if (!contractRec) {
    console.log(`No suitable contracts found for ${symbol} ${strategy}`);
    return null;
  }

  // Calculate position sizing (assuming $100k account)
  const positionSizing = calculatePositionSize(compositeScore.confidence, 100000, strategy);

  // Get strategy parameters
  const strategyParams = getStrategyParams(strategy);

  // Calculate data freshness
  const allTimestamps = [
    ...context.optionsTrades.map(t => t.timestamp.getTime()),
    ...context.news.map(n => n.timestamp.getTime()),
    ...context.institutionalTrades.map(t => t.timestamp.getTime()),
  ];
  const mostRecentTimestamp = allTimestamps.length > 0 ? Math.max(...allTimestamps) : Date.now();
  const mostRecentDataAge = Math.round((Date.now() - mostRecentTimestamp) / 60000); // minutes

  // Generate AI reasoning
  const allFactors = [
    ...flowAnalysis.factors,
    ...newsAnalysis.factors,
    ...institutionalAnalysis.factors,
    ...structureAnalysis.factors,
    ...technicalAnalysis.factors,
  ];

  const promptContext: PromptContext = {
    symbol,
    strategy,
    direction,
    bullishScore: compositeScore.bullishScore,
    bearishScore: compositeScore.bearishScore,
    confidence: compositeScore.confidence,
    flowFactors: flowAnalysis.factors,
    newsFactors: newsAnalysis.factors,
    institutionalFactors: institutionalAnalysis.factors,
    structureFactors: structureAnalysis.factors,
    technicalFactors: technicalAnalysis.factors,
    contract: contractRec.contract,
    currentPrice,
    vix: context.marketData.vix,
    putCallRatio: context.marketData.putCallRatio,
    ivRank: context.volStats.find(v => v.ticker === symbol)?.iv_rank,
    mostRecentDataAge,
  };

  let reasoning = {
    primaryCatalyst: 'Analyzing trade setup based on multi-factor analysis...',
    supportingFactors: allFactors.slice(0, 5),
    riskConsiderations: ['Monitor position closely', 'Use appropriate position sizing'],
    timeSensitivity: 'MEDIUM - Based on recent data',
  };

  try {
    const prompt = generateTradeAnalysisPrompt(promptContext);
    const response = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      messages: [
        {
          role: 'system',
          content: 'You are an expert options trader with 20 years of experience analyzing market data and recommending trades.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.7,
      max_tokens: 1000,
    });

    const aiResponse = response.choices[0]?.message?.content || '';
    const parsedResponse = parseAIResponse(aiResponse);

    if (parsedResponse.primaryCatalyst) {
      reasoning = parsedResponse;
    }
  } catch (error) {
    console.error('Error generating AI reasoning:', error);
    // Use fallback reasoning
  }

  // Calculate urgency
  const urgency = calculateUrgency(strategy, mostRecentDataAge);

  // Generate risk warnings
  const nearResistance = context.technicals
    ? context.technicals.supportResistance.distanceToResistance < 2
    : false;
  const technicalDivergence = context.technicals
    ? context.technicals.trend !== direction
    : false;

  const riskWarnings = generateRiskWarnings(
    strategy,
    promptContext.ivRank,
    promptContext.vix,
    nearResistance,
    technicalDivergence
  );

  // Get trading guidelines
  const tradingGuidelines = getStrategyGuidelines(strategy);

  // Format data freshness
  const dataFreshness = {
    mostRecentFlow: `${mostRecentDataAge} minutes ago`,
    mostRecentNews: context.news.length > 0
      ? `${Math.round((Date.now() - Math.max(...context.news.map(n => n.timestamp.getTime()))) / 60000)} minutes ago`
      : 'N/A',
    lastPriceUpdate: '< 1 minute ago',
  };

  // Create final recommendation
  const recommendation = formatTradeRecommendation(
    symbol,
    strategy,
    direction,
    compositeScore.bullishScore,
    compositeScore.bearishScore,
    compositeScore.confidence,
    contractRec,
    strategyParams.holdTimeEstimate,
    reasoning,
    compositeScore.factorBreakdown,
    positionSizing,
    tradingGuidelines,
    dataFreshness,
    urgency,
    riskWarnings
  );

  return recommendation;
}

/**
 * Main analysis function - analyzes symbols across multiple strategies
 */
export async function analyzeMultiStrategy(
  symbols: string[],
  strategies: Strategy[] = ['scalp', 'intraday', 'swing', 'leap'],
  threshold: number = 7
): Promise<MultiStrategyResponse[]> {
  console.log(`Starting multi-strategy analysis for ${symbols.length} symbols...`);

  // Aggregate all market data once
  const marketDataCache = await aggregateMarketData(symbols);

  const results: MultiStrategyResponse[] = [];

  // Analyze each symbol
  for (const symbol of symbols) {
    console.log(`\n=== Analyzing ${symbol} ===`);

    // Get current price
    const currentPrice = await getStockPrice(symbol);
    if (currentPrice === 0) {
      console.warn(`Could not fetch price for ${symbol}, skipping...`);
      continue;
    }

    // Get technical indicators from cache or fetch fresh
    let technicals = null;
    const cachedTechnicals = getCached<Record<string, any>>('technicals');
    if (cachedTechnicals && cachedTechnicals.data[symbol]) {
      console.log(`✅ Using cached technicals for ${symbol}`);
      technicals = cachedTechnicals.data[symbol];
    } else {
      console.log(`🔄 Fetching fresh technicals for ${symbol}`);
      technicals = await getTechnicalIndicators(symbol);
    }

    // Build analysis context
    const context: AnalysisContext = {
      symbol,
      ...marketDataCache,
      technicals,
      currentPrice,
    };

    // Analyze for each strategy
    const recommendations: TradeRecommendation[] = [];

    for (const strategy of strategies) {
      try {
        const rec = await analyzeSymbolForStrategy(context, strategy, threshold);
        if (rec) {
          recommendations.push(rec);
        }
      } catch (error) {
        console.error(`Error analyzing ${symbol} for ${strategy}:`, error);
      }

      // Small delay between strategies to avoid rate limits
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Create response for this symbol
    if (recommendations.length > 0) {
      const response = formatMultiStrategyResponse(
        symbol,
        currentPrice,
        {
          vix: marketDataCache.marketData.vix,
          putCallRatio: marketDataCache.marketData.putCallRatio,
          marketTide: marketDataCache.marketData.marketTide,
        },
        recommendations
      );

      results.push(response);
    } else {
      console.log(`No recommendations for ${symbol} meet the threshold`);
    }
  }

  return results;
}

/**
 * Simpler function for single-strategy analysis (backward compatibility)
 */
export async function analyzeSingleStrategy(
  symbols: string[],
  strategy: Strategy,
  threshold: number = 7
): Promise<TradeRecommendation[]> {
  const results = await analyzeMultiStrategy(symbols, [strategy], threshold);

  // Flatten recommendations
  const allRecommendations: TradeRecommendation[] = [];
  results.forEach(result => {
    allRecommendations.push(...result.recommendations);
  });

  return allRecommendations;
}
