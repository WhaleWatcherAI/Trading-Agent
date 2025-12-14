import OpenAI from 'openai';
import {
  TradeSignal,
  AnalysisRequest,
  AnalysisResponse,
  MarketData,
  NewsItem,
  OptionsTrade,
  InstitutionalTrade,
} from '@/types';
import { getStockPrice, getOptionsChain, getMarketData as getTradierMarketData } from './tradier';
import {
  getUnusualOptionsActivity, getInstitutionalTrades, getMarketNews, getPutCallRatio,
  getGreekFlow, getSectorFlow, getSectorTide, getSpotGEX, getVolatilityStats,
  getInstitutionalActivity, getNewsHeadlines
} from './unusualwhales';
import {
  analyzeNewsSentiment,
  analyzeInstitutionalActivity,
  analyzeOptionsFlow,
  analyzeMarketTide,
  calculateTradeScore,
} from './analyzer';
import { addNews, addOptionsTrades, addInstitutionalTrades } from './storage/dataStore';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface AggregatedData {
  news: NewsItem[];
  optionsTrades: OptionsTrade[];
  institutionalTrades: InstitutionalTrade[];
  marketData: MarketData;
  symbols: string[];
}

async function aggregateMarketData(): Promise<AggregatedData> {
  const today = new Date().toISOString().split('T')[0];

  try {
    // Fetch data with error handling for each endpoint
    let news: any[] = [];
    let optionsTrades: any[] = [];
    let institutionalTrades: any[] = [];
    let putCallRatio = 1.0;

    // Try to fetch news
    try {
      news = await getMarketNews(today);
      console.log(`✅ News API working: ${news.length} items`);
    } catch (error) {
      console.warn('⚠️  News API failed:', error);
    }

    // Try to fetch options
    try {
      optionsTrades = await getUnusualOptionsActivity(today);
      console.log(`✅ Options API working: ${optionsTrades.length} items`);
    } catch (error) {
      console.warn('⚠️  Options API failed:', error);
    }

    // Try to fetch institutional trades
    try {
      institutionalTrades = await getInstitutionalTrades(today);
      console.log(`✅ Institutional API working: ${institutionalTrades.length} items`);
    } catch (error) {
      console.warn('⚠️  Institutional API failed:', error);
    }

    // Try to fetch put/call ratio
    try {
      putCallRatio = await getPutCallRatio();
      console.log(`✅ Put/Call Ratio: ${putCallRatio.toFixed(2)}`);
    } catch (error) {
      console.warn('⚠️  Put/Call ratio API failed:', error);
    }

    // Try Tradier, but don't fail if it doesn't work
    let tradierMarket = { spy: 480, vix: 15, putCallRatio: 1.0 }; // Defaults
    try {
      tradierMarket = await getTradierMarketData();
      console.log('✅ Tradier API working');
    } catch (error) {
      console.warn('⚠️  Tradier API failed, using defaults');
    }

    // Store data with deduplication (only new items are added)
    const newsStats = addNews(news);
    const optionsStats = addOptionsTrades(optionsTrades);
    const instStats = addInstitutionalTrades(institutionalTrades);

    console.log('📊 Data Storage Stats:');
    console.log(`  News: ${newsStats.added} new, ${newsStats.total} total`);
    console.log(`  Options: ${optionsStats.added} new, ${optionsStats.total} total`);
    console.log(`  Institutional: ${instStats.added} new, ${instStats.total} total`);

    // Extract unique symbols from all sources
    const symbolsSet = new Set<string>();

    news.forEach(n => n.symbols.forEach((s: string) => symbolsSet.add(s)));
    optionsTrades.forEach(t => symbolsSet.add(t.underlying));
    institutionalTrades.forEach(t => symbolsSet.add(t.symbol));

    const symbols = Array.from(symbolsSet);

    // Determine market tide
    let marketTide: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (tradierMarket.vix < 15 && putCallRatio < 0.9) {
      marketTide = 'bullish';
    } else if (tradierMarket.vix > 25 || putCallRatio > 1.2) {
      marketTide = 'bearish';
    }

    const marketData: MarketData = {
      putCallRatio,
      vix: tradierMarket.vix,
      spy: tradierMarket.spy,
      marketTide,
      timestamp: new Date(),
    };

    return {
      news,
      optionsTrades,
      institutionalTrades,
      marketData,
      symbols,
    };
  } catch (error) {
    console.error('Error aggregating market data:', error);
    throw error;
  }
}

async function analyzeSymbol(
  symbol: string,
  data: AggregatedData,
  strategy: 'scalp' | 'intraday' | 'swing' | 'leap' | 'all' = 'intraday'
): Promise<TradeSignal | null> {
  try {
    // Analyze all factors first (don't need price for this)
    const newsAnalysis = analyzeNewsSentiment(data.news, symbol);
    const institutionalAnalysis = analyzeInstitutionalActivity(data.institutionalTrades, symbol);
    const optionsAnalysis = analyzeOptionsFlow(data.optionsTrades, symbol);
    const marketTide = analyzeMarketTide(data.marketData);

    // Calculate composite score
    const scoreResult = calculateTradeScore(
      newsAnalysis,
      institutionalAnalysis,
      optionsAnalysis,
      marketTide,
      data.marketData.putCallRatio
    );

    // Skip if score is too weak (between -3 and 3)
    if (Math.abs(scoreResult.rating) < 3) {
      return null;
    }

    // Get stock price (with fallback if Tradier fails)
    let stockPrice;
    try {
      stockPrice = await getStockPrice(symbol);
    } catch (error) {
      // Estimate from options data if available
      const relevantOptions = data.optionsTrades.filter(t => t.underlying === symbol);
      if (relevantOptions.length > 0) {
        // Use average strike as rough price estimate
        const avgStrike = relevantOptions.reduce((sum, o) => sum + o.strike, 0) / relevantOptions.length;
        stockPrice = { symbol, price: avgStrike, change: 0, changePercent: 0, volume: 0, timestamp: new Date() };
        console.warn(`⚠️  Using estimated price for ${symbol}: $${avgStrike.toFixed(2)}`);
      } else {
        console.warn(`⚠️  Skipping ${symbol} - no price data available`);
        return null;
      }
    }

    // Determine if bearish or bullish
    const isBearish = scoreResult.rating < 0;
    const optionType = isBearish ? 'put' : 'call';

    // Find best option contract (try to get from Tradier, fallback to Unusual Whales data)
    let optionsChain;
    try {
      optionsChain = await getOptionsChain(symbol);
    } catch (error) {
      // Use options from Unusual Whales data as fallback
      optionsChain = data.optionsTrades.filter(t => t.underlying === symbol);
      console.warn(`⚠️  Using Unusual Whales options data for ${symbol}`);
    }

    const filteredOptions = optionsChain.filter(o => o.type === optionType);

    if (filteredOptions.length === 0) {
      return null;
    }

    // Select ATM or slightly OTM option
    const targetStrike = isBearish
      ? Math.floor(stockPrice.price * 0.98) // Slightly below current for puts
      : Math.ceil(stockPrice.price * 1.02); // Slightly above current for calls

    const bestOption = filteredOptions.reduce((best, current) => {
      const currentDiff = Math.abs(current.strike - targetStrike);
      const bestDiff = Math.abs(best.strike - targetStrike);
      return currentDiff < bestDiff ? current : best;
    });

    // Generate reasoning
    const reasoning = await generateTradeReasoning(
      symbol,
      newsAnalysis,
      institutionalAnalysis,
      optionsAnalysis,
      marketTide,
      scoreResult
    );

    // Map 'all' to 'intraday' for backward compatibility
    const finalStrategy: 'scalp' | 'intraday' | 'swing' | 'leap' = strategy === 'all' ? 'intraday' : strategy;

    const signal: TradeSignal = {
      symbol: bestOption.symbol,
      underlying: symbol,
      contract: `${symbol} ${bestOption.expiration} ${bestOption.strike}${optionType.toUpperCase()}`,
      strike: bestOption.strike,
      expiration: bestOption.expiration,
      type: optionType,
      action: 'buy',
      strategy: finalStrategy,
      currentPrice: bestOption.premium,
      rating: scoreResult.rating,
      confidence: scoreResult.confidence,
      reasoning,
      factors: scoreResult.factors,
      timestamp: new Date(),
    };

    return signal;
  } catch (error) {
    console.error(`Error analyzing symbol ${symbol}:`, error);
    return null;
  }
}

async function generateTradeReasoning(
  symbol: string,
  newsAnalysis: ReturnType<typeof analyzeNewsSentiment>,
  institutionalAnalysis: ReturnType<typeof analyzeInstitutionalActivity>,
  optionsAnalysis: ReturnType<typeof analyzeOptionsFlow>,
  marketTide: any,
  scoreResult: any
): Promise<string> {
  const prompt = `Generate a concise trading reasoning (2-3 sentences) for ${symbol} based on:

News Sentiment: ${newsAnalysis.sentiment} (score: ${newsAnalysis.score.toFixed(2)})
Institutional Activity: ${institutionalAnalysis.signal} ($${(institutionalAnalysis.totalValue / 1_000_000).toFixed(1)}M)
Options Flow: ${optionsAnalysis.signal} ${optionsAnalysis.unusualActivity ? '(UNUSUAL ACTIVITY)' : ''}
Market Tide: ${marketTide}
Overall Rating: ${scoreResult.rating}/10

Focus on the most impactful factors and provide actionable insight.`;

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are a professional trading analyst. Provide concise, actionable insights.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      max_tokens: 150,
      temperature: 0.7,
    });

    return response.choices[0].message.content || 'Analysis complete.';
  } catch (error) {
    console.error('Error generating reasoning:', error);
    return 'Based on market analysis, this trade shows potential based on current data patterns.';
  }
}

export async function analyzeMarket(request: AnalysisRequest = {}): Promise<AnalysisResponse> {
  const { strategy = 'intraday', limit = 5 } = request;

  // Aggregate all market data
  const data = await aggregateMarketData();

  // Determine which symbols to analyze
  let symbolsToAnalyze = data.symbols;

  if (request.symbols && request.symbols.length > 0) {
    symbolsToAnalyze = request.symbols;
  }

  // Limit to top symbols by activity if too many
  if (symbolsToAnalyze.length > 50) {
    // Count activity for each symbol
    const activityCount = new Map<string, number>();

    symbolsToAnalyze.forEach(symbol => {
      const count =
        data.optionsTrades.filter(t => t.underlying === symbol).length +
        data.institutionalTrades.filter(t => t.symbol === symbol).length +
        data.news.filter(n => n.symbols.includes(symbol)).length;

      activityCount.set(symbol, count);
    });

    symbolsToAnalyze = Array.from(activityCount.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 50)
      .map(([symbol]) => symbol);
  }

  // Analyze each symbol
  const analysisPromises = symbolsToAnalyze.map(symbol => analyzeSymbol(symbol, data, strategy));

  const results = await Promise.all(analysisPromises);

  // Filter out nulls and sort by absolute rating
  const validTrades = results
    .filter((trade): trade is TradeSignal => trade !== null)
    .sort((a, b) => Math.abs(b.rating) - Math.abs(a.rating))
    .slice(0, limit);

  return {
    trades: validTrades,
    marketOverview: data.marketData,
    timestamp: new Date(),
  };
}
