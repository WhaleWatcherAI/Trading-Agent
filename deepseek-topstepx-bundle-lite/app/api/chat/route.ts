import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import {
  getUnusualOptionsActivity,
  getInstitutionalActivity,
  getNewsHeadlines,
  getGreekFlow,
  getSpotGEX,
  getVolatilityStats,
  getPutCallRatio,
} from '@/lib/unusualwhales';
import { getMarketData, getStockPrice as getTradierPrice } from '@/lib/tradier';
import { getTechnicalIndicators } from '@/lib/technicals';
import {
  getCached,
  setCache,
  getWithStaleness,
  STALENESS_CONFIG,
} from '@/lib/dataCache';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * AI Financial Analyst Chat Bot
 *
 * Conversational interface to query market data, analyze stocks, and get trading insights
 * Persona: Expert hedge fund trader and financial analyst
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { message, conversationHistory = [] } = body;

    if (!message) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    console.log(`💬 Chat query: "${message}"`);

    // Determine if user is asking about specific symbol(s)
    const symbolPattern = /\b([A-Z]{1,5})\b/g;
    const potentialSymbols = message.match(symbolPattern) || [];
    const knownSymbols = potentialSymbols.filter((s: string) =>
      s.length >= 1 && s.length <= 5 && s !== 'I' && s !== 'A'
    );

    // Fetch relevant market data
    let marketContext = '';

    // Helper to get cached data or fetch if stale
    async function getCachedOrFetch<T>(
      key: string,
      fetcher: () => Promise<T>,
      fallback: T
    ): Promise<T> {
      const cached = getWithStaleness<T>(key, STALENESS_CONFIG[key]?.budget || 60000);

      if (cached.isFresh && cached.data) {
        console.log(`✅ Chat using cached ${key}`);
        return cached.data;
      }

      // Cache miss or stale - fetch fresh data
      console.log(`🔄 Chat cache ${cached.data ? 'stale' : 'miss'} for ${key}, fetching...`);
      try {
        const data = await fetcher();
        setCache(key, data, 'chat-api');
        return data;
      } catch (error) {
        console.warn(`⚠️  Chat failed to fetch ${key}, using ${cached.data ? 'stale cache' : 'fallback'}`);
        return cached.data || fallback;
      }
    }

    try {
      // Fetch all data from cache or API (parallel - much faster!)
      const [
        marketData,
        putCallRatio,
        news,
        greekFlow,
        spotGEX,
        volStats,
        institutionalActivity,
        optionsFlow,
      ] = await Promise.all([
        getCachedOrFetch('market_data', getMarketData, { spy: 0, vix: 20, putCallRatio: 1.0 }),
        getCachedOrFetch('put_call_ratio', getPutCallRatio, 1.0),
        getCachedOrFetch('news_headlines', getNewsHeadlines, []),
        getCachedOrFetch('greek_flow', getGreekFlow, []),
        getCachedOrFetch('spot_gex', getSpotGEX, []),
        getCachedOrFetch('volatility_stats', getVolatilityStats, []),
        getCachedOrFetch('institutional', getInstitutionalActivity, []),
        getCachedOrFetch('sector_tide', getUnusualOptionsActivity, []),
      ]);

      marketContext += `\n## Current Market Conditions
- SPY: $${marketData.spy.toFixed(2)}
- VIX: ${marketData.vix || 'N/A'}
- Put/Call Ratio: ${putCallRatio.toFixed(2)}
- Market Tide: ${putCallRatio > 1.1 ? 'Bearish' : putCallRatio < 0.9 ? 'Bullish' : 'Neutral'}

## Gamma Exposure (GEX) - Live Data
${spotGEX.slice(0, 4).map(g =>
  `- ${g.ticker}: Gamma/1% = ${g.gamma_per_one_percent_move_oi.toFixed(2)}M, Vanna = ${g.vanna_per_one_percent_move_oi.toFixed(2)}M`
).join('\n') || '- No GEX data available'}

## Volatility Stats
${volStats.slice(0, 4).map(v =>
  `- ${v.ticker}: IV Rank = ${v.iv_rank.toFixed(0)}%, IV = ${(v.iv * 100).toFixed(1)}%, RV = ${(v.rv * 100).toFixed(1)}%`
).join('\n') || '- No volatility data available'}

## Greek Flow (Sector-Level Options Activity)
${greekFlow.length > 0 ? greekFlow.slice(0, 5).map(g =>
  `- ${g.flow_group}: Call Premium = $${(g.net_call_premium / 1000000).toFixed(1)}M, Put Premium = $${(g.net_put_premium / 1000000).toFixed(1)}M, Delta Flow = ${(g.dir_delta_flow / 1000000).toFixed(1)}M`
).join('\n') : '- No greek flow data available (may be after hours or weekend)'}

## Unusual Options Activity (Market-Wide Flow)
${optionsFlow.length > 0 ? optionsFlow.slice(0, 20).filter(f => f && f.type && f.underlying).map(f =>
  `- ${f.underlying}: ${f.type?.toUpperCase() || 'N/A'} $${f.strike || 0} exp ${f.expiration} - ${f.unusual ? '🔥 UNUSUAL' : ''} Premium: $${((f.premium || 0) / 1000).toFixed(0)}K, Vol: ${(f.volume || 0).toLocaleString()}, Side: ${f.side?.toUpperCase() || 'N/A'}`
).join('\n') : '- No unusual options flow data available (may be after hours or weekend)'}

## Institutional Activity (Recent Large Trades)
${institutionalActivity.slice(0, 10).map(t =>
  `- ${t.institution}: ${t.side.toUpperCase()} ${t.shares.toLocaleString()} ${t.symbol} @ $${t.price.toFixed(2)} (Value: $${(t.value / 1000000).toFixed(1)}M)`
).join('\n') || '- No institutional activity available'}

## All Market News Headlines (Latest 15)
${news.slice(0, 15).map((n, idx) =>
  `${idx + 1}. ${n.title}${n.symbols.length > 0 ? ` [${n.symbols.slice(0, 3).join(', ')}${n.symbols.length > 3 ? '...' : ''}]` : ''} - ${(n.sentiment || 'neutral').toUpperCase()}`
).join('\n') || '- No recent news available'}
`;

      // If specific symbols mentioned, fetch detailed data
      if (knownSymbols.length > 0) {
        console.log(`📊 Fetching data for symbols: ${knownSymbols.join(', ')}`);

        for (const symbol of knownSymbols.slice(0, 3)) { // Max 3 symbols to avoid slowness
          try {
            const [price, technicals, optionsFlow, symbolVolStats] = await Promise.all([
              getTradierPrice(symbol).catch(() => 0),
              getTechnicalIndicators(symbol).catch(() => null),
              getUnusualOptionsActivity().then(flows =>
                flows.filter(f => f.underlying === symbol).slice(0, 5)
              ).catch(() => []),
              getVolatilityStats().then(stats =>
                stats.find(s => s.ticker === symbol)
              ).catch(() => null),
            ]);

            const priceNum = typeof price === 'number' ? price : 0;

            marketContext += `\n## ${symbol} Analysis
- Current Price: $${priceNum > 0 ? priceNum.toFixed(2) : 'N/A'}
${technicals ? `- RSI: ${technicals.rsi.toFixed(1)} (${technicals.trendSignals.rsiSignal})
- Trend: ${technicals.trend.toUpperCase()}
- Price vs SMA50: ${priceNum > 0 && technicals ? ((priceNum - technicals.sma_50) / technicals.sma_50 * 100).toFixed(2) : 'N/A'}%
- Support: $${technicals.supportResistance.support.toFixed(2)} | Resistance: $${technicals.supportResistance.resistance.toFixed(2)}` : ''}
${symbolVolStats ? `- IV Rank: ${symbolVolStats.iv_rank.toFixed(0)}%` : ''}
${optionsFlow.length > 0 ? `- Recent Unusual Options Flow:\n${optionsFlow.map(f =>
  `  • ${f.type.toUpperCase()} $${f.strike} ${f.expiration} - $${(f.premium / 1000).toFixed(0)}K premium (${f.side})`
).join('\n')}` : ''}
`;
          } catch (err) {
            console.error(`Error fetching data for ${symbol}:`, err);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching market context:', error);
      marketContext = '\n(Market data temporarily unavailable)';
    }

    // Build conversation with system prompt
    const messages: any[] = [
      {
        role: 'system',
        content: `You are an expert financial analyst and hedge fund trader with 20+ years of experience. Your specialties include:
- Options trading strategies (scalping, day trading, swing trading, LEAPs)
- Technical analysis (RSI, MACD, SMAs, Bollinger Bands, support/resistance)
- Market structure (gamma exposure, put/call ratios, volatility)
- Institutional flow and unusual options activity
- Risk management and position sizing

Your personality:
- Direct and actionable - no fluff
- Data-driven with clear reasoning
- Honest about risks and uncertainties
- Use precise numbers and percentages
- Cite specific indicators and levels

YOU HAVE ACCESS TO THE FOLLOWING LIVE DATA SOURCES:
1. **Market Conditions**: SPY price, VIX (when available), Put/Call ratio, market tide
2. **Gamma Exposure (GEX)**: Live gamma and vanna exposure for SPY, QQQ, AAPL, NVDA
3. **Volatility Stats**: IV Rank, Implied Volatility, Realized Volatility for major tickers
4. **Greek Flow**: Sector-level options activity (call/put premium, delta flow by sector)
5. **Unusual Options Activity**: Market-wide options flow showing unusual trades with premium, volume, strikes
6. **Institutional Activity**: Recent large trades from major institutions (Vanguard, BlackRock, State Street, JPMorgan, Goldman Sachs)
7. **Market News**: Latest 15 market headlines with sentiment and ticker tags
8. **Technical Indicators**: RSI, trend, SMAs, support/resistance (when specific symbols are mentioned)

NOTE: Some data may be unavailable after market hours or on weekends. VIX data may not be available in sandbox environment.

Current market data is provided below. Use it to give informed, actionable insights.

${marketContext}

IMPORTANT:
- Always reference specific data points from above when making claims
- You have access to EXTENSIVE data - use it! Reference GEX levels, institutional trades, greek flow, etc.
- Provide specific price levels, not vague suggestions
- Include risk warnings when appropriate
- Keep responses concise (2-4 paragraphs max) unless asked for detail
- If asked what data you have, list ALL the sources above`,
      },
    ];

    // Add conversation history
    conversationHistory.forEach((msg: any) => {
      messages.push({ role: msg.role, content: msg.content });
    });

    // Add current user message
    messages.push({ role: 'user', content: message });

    // Get AI response
    const response = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      messages,
      temperature: 0.7,
      max_tokens: 1000,
    });

    const aiMessage = response.choices[0].message.content || 'I apologize, I could not generate a response.';

    console.log(`✅ Chat response generated (${aiMessage.length} chars)`);

    return NextResponse.json({
      message: aiMessage,
      symbols: knownSymbols,
      timestamp: new Date().toISOString(),
    });

  } catch (error: any) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat message', details: error.message },
      { status: 500 }
    );
  }
}
