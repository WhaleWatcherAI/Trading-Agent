import OpenAI from 'openai';
import { jsonrepair } from 'jsonrepair';
import { NewsItem } from '@/types';
import { getNewsHeadlines } from './unusualwhales';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export interface SentimentAnalysis {
  symbol: string;
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  score: number; // 1-10, where 10 is strongest conviction
  reasoning: string;
  newsCount: number;
}

export interface MarketSentimentInsight {
  symbol: string;
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  score: number; // 1-10 scale
  confidence: number; // 0-100 percent
  reasoning: string;
  supportingHeadlines: string[];
}

export interface DailyMarketSentiment {
  generatedAt: string;
  marketSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  summary: string;
  insights: MarketSentimentInsight[];
  totalNewsAnalyzed: number;
  notes?: string;
}

export type AssetClass = 'stock' | 'crypto' | 'future' | 'index' | 'etf' | 'fx' | 'other';

export interface ArticleSymbolImpact {
  symbol: string;
  assetClass: AssetClass;
  score: number; // -10 to +10
  reasoning: string;
}

export interface ArticleImpact {
  index: number; // 1-based index from the digest
  marketScore: number; // -10 to +10
  symbolImpacts: ArticleSymbolImpact[];
}

export interface SymbolSentimentRating {
  symbol: string;
  assetClass: AssetClass;
  score: number; // -10 to +10
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  reasoning: string;
  supportingHeadlines: number[];
}

export interface NewsImpactAnalysis {
  generatedAt: string;
  newsWindowDays: number;
  marketScore: number; // -10 to +10
  marketSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  symbolRatings: SymbolSentimentRating[];
  articleImpacts: ArticleImpact[];
  totalNewsAnalyzed: number;
}

export async function analyzeSentiment(symbol: string, date?: string): Promise<SentimentAnalysis> {
  try {
    console.log(`📰 Analyzing sentiment for ${symbol}${date ? ` on ${date}` : ''}...`);

    // Get news headlines from Unusual Whales
    const allNews = await getNewsHeadlines(date);

    // Filter for symbol-specific news
    const symbolNews = allNews.filter(news =>
      news.title.toUpperCase().includes(symbol.toUpperCase()) ||
      news.source?.toUpperCase().includes(symbol.toUpperCase())
    );

    console.log(`   Found ${symbolNews.length} news items for ${symbol}`);

    if (symbolNews.length === 0) {
      console.log(`   ⚠️  No news found for ${symbol}, returning NEUTRAL`);
      return {
        symbol,
        sentiment: 'NEUTRAL',
        score: 5,
        reasoning: 'No news available',
        newsCount: 0,
      };
    }

    // Prepare headlines for OpenAI
    const headlines = symbolNews.slice(0, 10).map(n => n.title).join('\n');

    const prompt = `Analyze the following news headlines for ${symbol} stock and provide a sentiment score.

News Headlines:
${headlines}

Provide your analysis in the following JSON format:
{
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "score": <number 1-10, where 10 is strongest conviction>,
  "reasoning": "<brief explanation of your analysis>"
}

Rules:
- Score 1-3: Weak sentiment
- Score 4-6: Moderate sentiment
- Score 7-9: Strong sentiment
- Score 10: Extremely strong sentiment
- BULLISH scores: Positive news, growth, earnings beats, upgrades, etc.
- BEARISH scores: Negative news, losses, downgrades, regulatory issues, etc.
- NEUTRAL: Mixed or no clear direction

Respond ONLY with valid JSON, no additional text.`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'You are a financial sentiment analyst. Analyze news headlines and provide sentiment scores for stocks. Always respond with valid JSON only.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0.3,
      max_tokens: 200,
      response_format: {
        type: 'json_object',
      },
    });

    const responseText = completion.choices[0].message.content?.trim() || '{}';
    let analysis: any;

    try {
      analysis = JSON.parse(responseText);
    } catch (parseError) {
      analysis = JSON.parse(jsonrepair(responseText));
    }

    const result: SentimentAnalysis = {
      symbol,
      sentiment: analysis.sentiment || 'NEUTRAL',
      score: Math.min(10, Math.max(1, analysis.score || 5)),
      reasoning: analysis.reasoning || 'No analysis available',
      newsCount: symbolNews.length,
    };

    console.log(`   ✅ ${symbol} Sentiment: ${result.sentiment} (${result.score}/10) - ${result.reasoning}`);

    return result;

  } catch (error: any) {
    console.error(`   ❌ Error analyzing sentiment for ${symbol}:`, error.message);

    // Fallback to neutral
    return {
      symbol,
      sentiment: 'NEUTRAL',
      score: 5,
      reasoning: `Error: ${error.message}`,
      newsCount: 0,
    };
  }
}

// Batch analyze multiple symbols
export async function analyzeSentimentBatch(symbols: string[], date?: string): Promise<Map<string, SentimentAnalysis>> {
  const results = new Map<string, SentimentAnalysis>();

  for (const symbol of symbols) {
    const analysis = await analyzeSentiment(symbol, date);
    results.set(symbol, analysis);

    // Small delay to avoid rate limits
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  return results;
}

export function filterDailyNews(allNews: NewsItem[], date?: string, days: number = 1): NewsItem[] {
  const now = new Date();
  const end = date ? new Date(`${date}T23:59:59Z`) : now;
  const windowMs = Math.max(1, days) * 24 * 60 * 60 * 1000;
  const start = new Date(end.getTime() - windowMs);

  return allNews.filter(news => {
    const published = news.timestamp ? new Date(news.timestamp) : null;

    if (!published) {
      return false;
    }

    return published >= start && published <= end;
  });
}

export function formatNewsDigest(news: NewsItem[]): string {
  return news.map((item, index) => {
    const cleanSummary = (item.summary?.replace(/\s+/g, ' ').trim() || 'No summary available.').slice(0, 280);
    const symbols = item.symbols?.length ? item.symbols.join(', ') : 'Unknown';
    const sentimentTag = item.sentiment ? item.sentiment.toUpperCase() : 'NEUTRAL';

    return `${index + 1}. Headline: ${item.title}
Summary: ${cleanSummary}
Tickers Mentioned: ${symbols}
News Sentiment: ${sentimentTag}
Published: ${item.timestamp.toISOString()}`;
  }).join('\n\n');
}

interface SymbolNewsStats {
  bullish: number;
  bearish: number;
  neutral: number;
  positiveHits: number;
  negativeHits: number;
  headlines: {
    index: number;
    item: NewsItem;
  }[];
}

function buildSymbolStats(news: NewsItem[]): Map<string, SymbolNewsStats> {
  const stats = new Map<string, SymbolNewsStats>();
  const positiveKeywords = [
    'beat',
    'beats',
    'record',
    'surge',
    'strong',
    'growth',
    'expansion',
    'accelerat',
    'increase',
    'raises',
    'raise',
    'hike',
    'upgraded',
    'buyback',
    'repurchase',
    'profit',
    'profits',
    'positive',
    'outperform',
    'improved',
    'improves',
    'higher',
    'opens position',
    'initiates position',
    'initiates stake',
    'adds to stake',
    'accumulat',
    'buys',
    'buying',
  ];
  const negativeKeywords = [
    'miss',
    'misses',
    'cut',
    'cuts',
    'lower',
    'downgrade',
    'decline',
    'drop',
    'slump',
    'warning',
    'loss',
    'losses',
    'negative',
    'investigation',
    'lawsuit',
    'bankrupt',
    'guidance down',
    'reduction',
    'reduces',
    'delay',
    'layoff',
    'recall',
    'closes position',
    'closes stake',
    'sells',
    'selling',
    'trim',
    'trims',
    'reduces stake',
    'dump',
    'dumps',
  ];

  news.forEach((item, idx) => {
    const text = `${item.title} ${item.summary}`.toLowerCase();
    const positiveHits = positiveKeywords.reduce(
      (count, keyword) => (text.includes(keyword) ? count + 1 : count),
      0
    );
    const negativeHits = negativeKeywords.reduce(
      (count, keyword) => (text.includes(keyword) ? count + 1 : count),
      0
    );

    const storySentiment = item.sentiment?.toLowerCase();
    const targetSymbols = (item.symbols || []).map(symbol => symbol?.toUpperCase()).filter(Boolean);

    targetSymbols.forEach(symbol => {
      const entry = stats.get(symbol) || {
        bullish: 0,
        bearish: 0,
        neutral: 0,
        positiveHits: 0,
        negativeHits: 0,
        headlines: [],
      };

      if (storySentiment === 'bullish') {
        entry.bullish += 1;
      } else if (storySentiment === 'bearish') {
        entry.bearish += 1;
      } else {
        entry.neutral += 1;
      }

      entry.positiveHits += positiveHits;
      entry.negativeHits += negativeHits;
      entry.headlines.push({ index: idx + 1, item });

      stats.set(symbol, entry);
    });
  });

  return stats;
}

export async function analyzeDailyMarketSentiment(date?: string): Promise<DailyMarketSentiment> {
  try {
    console.log(`📰 Generating daily market sentiment report${date ? ` for ${date}` : ''}...`);

    const allNews = await getNewsHeadlines();
    const windowDays = 3;
    const dailyNews = filterDailyNews(allNews, date, windowDays);

    if (!dailyNews.length) {
      console.log('   ⚠️  No news found for the requested window, returning neutral state');
      const neutral: DailyMarketSentiment = {
        generatedAt: new Date().toISOString(),
        marketSentiment: 'NEUTRAL',
        summary: 'No qualifying news detected in the requested window.',
        insights: [],
        totalNewsAnalyzed: 0,
      };

      return neutral;
    }

    // Sort by importance and recency, then cap to keep prompt size manageable
    const sortedNews = [...dailyNews].sort((a, b) => {
      if (b.importance !== a.importance) {
        return b.importance - a.importance;
      }

      return b.timestamp.getTime() - a.timestamp.getTime();
    });

    const MAX_HEADLINES = 40;
    const selectedNews = sortedNews.slice(0, MAX_HEADLINES);
    const symbolStats = buildSymbolStats(selectedNews);
    const digest = formatNewsDigest(selectedNews);

    const prompt = `You are a market news analyst. Review the last 24 hours of Benzinga market headlines and summarize the overall market sentiment.

News Digest:
${digest}

Instructions:
- Identify up to 15 stocks (tickers) most likely to be impacted by this news flow.
- For each ticker, determine sentiment as BULLISH, BEARISH, or NEUTRAL and assign a 1-10 conviction score.
- Provide a short reasoning that references the catalysts or themes.
- Include the most relevant headlines (by number) that support each insight.
- Offer an overall market sentiment with a one sentence summary.
- If a ticker is not explicitly mentioned but is clearly implied, infer it.

Respond ONLY with valid JSON matching this TypeScript type:
{
  "marketSentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "summary": string,
  "insights": Array<{
    "symbol": string,
    "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
    "score": number,
    "confidence": number,
    "reasoning": string,
    "supportingHeadlines": number[]
  }>,
  "notes"?: string
}

Rules:
- Confidence must be 0-100.
- Use the full 1-10 scale: 1-3 indicates clear negative pressure, 4 is mild negative, 6-7 is constructive positive, 8-10 is strong positive conviction. Score 5 only when the news flow is genuinely mixed or lacks directional bias.
- Treat concrete positives (earnings beats, accelerating growth, margin expansion, large buybacks, bullish guidance, regulatory wins) as BULLISH with scores above 5. Treat concrete negatives (misses, lowered guidance, layoffs for distress, investigations, regulatory setbacks) as BEARISH with scores below 5.
- Avoid assigning identical scores and confidence to every ticker unless the news catalysts are truly indistinguishable in strength.
- Only include tickers with a clear connection to the provided headlines.
- Use headline numbers from the digest for supportingHeadlines.
- Do not include any additional text outside the JSON.`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'You are an expert equities analyst. You turn collections of news headlines into structured market sentiment insights. Always respond with valid JSON only.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.2,
      max_tokens: 700,
      response_format: {
        type: 'json_object',
      },
    });

    const raw = completion.choices[0].message.content?.trim() || '{}';
    let parsed: any;
    try {
      parsed = JSON.parse(raw);
    } catch (parseError) {
      parsed = JSON.parse(jsonrepair(raw));
    }

    const normalizedMarketSentiment = (parsed.marketSentiment || 'NEUTRAL')
      .toString()
      .toUpperCase();
    const marketSentiment: DailyMarketSentiment['marketSentiment'] =
      normalizedMarketSentiment === 'BULLISH' || normalizedMarketSentiment === 'BEARISH'
        ? normalizedMarketSentiment
        : 'NEUTRAL';

    const adjustmentNotes: string[] = [];

    const baseInsights = Array.isArray(parsed.insights)
      ? parsed.insights.slice(0, 15).map((item: any): MarketSentimentInsight => {
          const normalizedSymbol = (item.symbol || 'UNKNOWN').toString().toUpperCase();
          const normalizedSentiment = (item.sentiment || 'NEUTRAL').toString().toUpperCase();
          const stats = symbolStats.get(normalizedSymbol);

          let sentiment: MarketSentimentInsight['sentiment'] =
            normalizedSentiment === 'BULLISH' || normalizedSentiment === 'BEARISH' ? normalizedSentiment : 'NEUTRAL';
          let score = Math.min(10, Math.max(1, Number(item.score) || 5));
          let confidence = Math.min(100, Math.max(0, Number(item.confidence) || 50));
          let reasoning = (item.reasoning || 'No reasoning provided.').trim();

          let supportingHeadlines = Array.isArray(item.supportingHeadlines)
            ? item.supportingHeadlines
                .map((idx: any) => {
                  if (typeof idx === 'number' && idx >= 1 && idx <= selectedNews.length) {
                    const headline = selectedNews[idx - 1];
                    return `${idx}. ${headline.title}`;
                  }
                  return null;
                })
                .filter(Boolean) as string[]
            : [];

          if (!supportingHeadlines.length && stats) {
            supportingHeadlines = stats.headlines.slice(0, 3).map(h => `${h.index}. ${h.item.title}`);
          }

          if (stats) {
            const bullishWeight = stats.bullish * 2 + stats.positiveHits;
            const bearishWeight = stats.bearish * 2 + stats.negativeHits;

            if (sentiment === 'NEUTRAL') {
              if (bullishWeight > bearishWeight && bullishWeight > 0) {
                const adjustedScore = Math.min(9, Math.max(score > 5 ? score : 6, 6 + Math.floor(bullishWeight / 2)));
                const adjustedConfidence = Math.min(95, Math.max(confidence, 55 + bullishWeight * 5));
                sentiment = 'BULLISH';
                score = adjustedScore;
                confidence = adjustedConfidence;
                reasoning = `${reasoning} Positive skew from ${stats.bullish} bullish headlines and ${stats.positiveHits} positive signals.`
                  .trim();
                adjustmentNotes.push(`Auto-adjusted ${normalizedSymbol} to BULLISH based on aggregated headlines.`);
              } else if (bearishWeight > bullishWeight && bearishWeight > 0) {
                const adjustedScore = Math.max(
                  2,
                  Math.min(score < 5 ? score : 4, 4 - Math.floor(bearishWeight / 2))
                );
                const adjustedConfidence = Math.min(95, Math.max(confidence, 55 + bearishWeight * 5));
                sentiment = 'BEARISH';
                score = adjustedScore;
                confidence = adjustedConfidence;
                reasoning = `${reasoning} Negative skew from ${stats.bearish} bearish headlines and ${stats.negativeHits} negative signals.`
                  .trim();
                adjustmentNotes.push(`Auto-adjusted ${normalizedSymbol} to BEARISH based on aggregated headlines.`);
              }
            } else if (sentiment === 'BULLISH') {
              score = Math.min(9, Math.max(score > 5 ? score : 6, 6 + Math.floor(bullishWeight / 2)));
              confidence = Math.min(95, Math.max(confidence, 55 + bullishWeight * 5));
            } else if (sentiment === 'BEARISH') {
              score = Math.max(2, Math.min(score < 5 ? score : 4, 4 - Math.floor(bearishWeight / 2)));
              confidence = Math.min(95, Math.max(confidence, 55 + bearishWeight * 5));
            }
          }

          return {
            symbol: normalizedSymbol,
            sentiment,
            score,
            confidence,
            reasoning,
            supportingHeadlines,
          };
        })
      : [];

    const insightSymbols = new Set(baseInsights.map((insight: MarketSentimentInsight) => insight.symbol));
    const additionalInsights: MarketSentimentInsight[] = [];

    symbolStats.forEach((stats, symbol) => {
      if (insightSymbols.has(symbol)) {
        return;
      }

      const bullishWeight = stats.bullish * 2 + stats.positiveHits;
      const bearishWeight = stats.bearish * 2 + stats.negativeHits;
      const relevantHeadlines = stats.headlines;

      if (!relevantHeadlines.length) {
        return;
      }

      if (bullishWeight >= Math.max(2, bearishWeight + 1)) {
        const score = Math.min(9, 6 + Math.max(1, Math.floor(bullishWeight / 2)));
        const confidence = Math.min(95, 55 + bullishWeight * 5);
        const reasoningHeadline = relevantHeadlines[0];
        const reasoning = `Bullish momentum: ${reasoningHeadline.item.title} (plus ${Math.max(
          0,
          relevantHeadlines.length - 1,
        )} more supporting headline(s)).`;
        const supportingHeadlines = relevantHeadlines.slice(0, 3).map(h => `${h.index}. ${h.item.title}`);

        additionalInsights.push({
          symbol,
          sentiment: 'BULLISH',
          score,
          confidence,
          reasoning,
          supportingHeadlines,
        });
        adjustmentNotes.push(`Added ${symbol} as BULLISH from headline aggregation.`);
        return;
      }

      if (bearishWeight >= Math.max(2, bullishWeight + 1)) {
        const score = Math.max(2, 4 - Math.max(1, Math.floor(bearishWeight / 2)));
        const confidence = Math.min(95, 55 + bearishWeight * 5);
        const reasoningHeadline = relevantHeadlines[0];
        const reasoning = `Bearish pressure: ${reasoningHeadline.item.title} (plus ${Math.max(
          0,
          relevantHeadlines.length - 1,
        )} more supporting headline(s)).`;
        const supportingHeadlines = relevantHeadlines.slice(0, 3).map(h => `${h.index}. ${h.item.title}`);

        additionalInsights.push({
          symbol,
          sentiment: 'BEARISH',
          score,
          confidence,
          reasoning,
          supportingHeadlines,
        });
        adjustmentNotes.push(`Added ${symbol} as BEARISH from headline aggregation.`);
      }
    });

    const allInsights = [...baseInsights, ...additionalInsights].slice(0, 15);

    const result: DailyMarketSentiment = {
      generatedAt: new Date().toISOString(),
      marketSentiment,
      summary: parsed.summary || 'No summary provided.',
      insights: allInsights,
      totalNewsAnalyzed: dailyNews.length,
      notes:
        parsed.notes ||
        (adjustmentNotes.length ? adjustmentNotes.slice(0, 5).join(' | ') : undefined),
    };

    console.log(`   ✅ Generated market sentiment report with ${result.insights.length} insights (analyzed ${result.totalNewsAnalyzed} headlines)`);

    return result;
  } catch (error: any) {
    console.error('   ❌ Error generating daily market sentiment:', error.message);

    return {
      generatedAt: new Date().toISOString(),
      marketSentiment: 'NEUTRAL',
      summary: `Error generating market sentiment: ${error.message}`,
      insights: [],
      totalNewsAnalyzed: 0,
    };
  }
}

export async function analyzeNewsImpactMultiAsset(
  date?: string,
  windowDays: number = 3
): Promise<NewsImpactAnalysis> {
  try {
    console.log(
      `📰 Running multi-asset news impact analysis${date ? ` for ${date}` : ''} (window: ${windowDays} day(s))...`
    );

    const allNews = await getNewsHeadlines();
    const windowedNews = filterDailyNews(allNews, date, windowDays);

    if (!windowedNews.length) {
      console.log('   ⚠️  No news found for requested window, returning neutral snapshot');
      return {
        generatedAt: new Date().toISOString(),
        newsWindowDays: windowDays,
        marketScore: 0,
        marketSentiment: 'NEUTRAL',
        symbolRatings: [],
        articleImpacts: [],
        totalNewsAnalyzed: 0,
      };
    }

    const sortedNews = [...windowedNews].sort((a, b) => {
      if (b.importance !== a.importance) {
        return b.importance - a.importance;
      }

      return b.timestamp.getTime() - a.timestamp.getTime();
    });

    const MAX_HEADLINES = 40;
    const selectedNews = sortedNews.slice(0, MAX_HEADLINES);
    const keywordStats = buildSymbolStats(selectedNews);
    const digest = formatNewsDigest(selectedNews);

    const prompt = `You are a multi-asset macro and equity analyst. Review the following news headlines and determine how they impact the overall market and specific tradable instruments (stocks, ETFs, indices, futures, crypto, FX, etc.).

News Digest:
${digest}

Your task:
- For each headline, infer which tickers or instruments are impacted. This includes:
  - Explicit tickers mentioned in the headline or summary.
  - Implied tickers, sectors, or indices (e.g. \"chip tariffs\" impact NVDA, AMD, SMH, SOXX, etc.).
  - Macro news (rates, inflation, Fed, geopolitics) that moves broad indices (SPY, QQQ, IWM, ES, NQ, RTY, BTC, etc.).
- Classify impact direction and strength as a numeric score from -10 to +10:
  - -10 = extremely bearish, -5 = clearly bearish, 0 = neutral/mixed, +5 = clearly bullish, +10 = extremely bullish.
  - Use negative values for bearish pressure, positive for bullish, 0 when no clear directional bias.
- Some examples:
  - Chip tariffs increase or new restrictions → generally BEARISH for NVDA, AMD, SMH, semi sector.
  - Tariffs reduced or removed → BULLISH for affected exporters/importers.
  - Large rate cuts, clearly dovish Fed shift → BULLISH for broad indices unless headline is very negative for a particular name.
  - A small-cap winning a major partnership with a big company → BULLISH for the small-cap (and sometimes mildly bearish or neutral for the large partner).

Respond ONLY with JSON matching this type:
{
  \"articleImpacts\": Array<{
    \"index\": number,              // Headline number from the digest (1-based)
    \"marketScore\": number,        // -10 to +10 impact on the overall market
    \"symbolImpacts\": Array<{
      \"symbol\": string,           // e.g. \"NVDA\", \"AMD\", \"SPY\", \"BTC\", \"ES\"
      \"assetClass\": \"stock\" | \"crypto\" | \"future\" | \"index\" | \"etf\" | \"fx\" | \"other\",
      \"score\": number,            // -10 to +10 sentiment score for that symbol
      \"reasoning\": string         // short explanation (1-2 sentences)
    }>
  }>
}

Rules:
- Use numeric scores in the range -10 to +10 (decimals allowed).
- Use the full scale:
  - -1 to -2.5 or +1 to +2.5 = very mild bias,
  - -3 to -4.5 or +3 to +4.5 = clear but not extreme,
  - -5 to -7.5 or +5 to +7.5 = strong directional impact,
  - -8 to -10 or +8 to +10 = extremely strong, rare events.
- For every headline, you MUST:
  - If it has any directional bias at all, choose a non-zero marketScore (even if only -0.5 or +0.5).
  - Include at least 1 symbolImpact when any ticker, sector, asset, or macro theme is present.
- Calibrate scores *relative to other headlines in this digest*:
  - If one headline is the most negative in the set, its scores should be closer to -8 to -10 than the others.
  - If one headline is the most positive, its scores should be closer to +8 to +10.
  - Avoid giving every symbol the same score; reflect differences in strength and confidence.
- Across all symbolImpacts, avoid clustering everything at the same magnitude (e.g. all -5); use at least 3–4 distinct negative levels and 3–4 distinct positive levels whenever the news flow supports it.
- Macro or policy headlines that affect the whole market should include broad instruments like SPY, QQQ, IWM, ES, NQ, RTY, DXY, BTC as appropriate.
- When mapping to futures and other derivatives, treat:
  - S&P 500 news as impacting SPX/SPY and ES,
  - Nasdaq 100 news as impacting NDX/QQQ and NQ,
  - Russell 2000 news as impacting RUT/IWM and RTY,
  - Nikkei 225 news as impacting NKY/N225 and Nikkei futures,
  - Crude oil/OPEC/inventory news as impacting CL and related energy ETFs,
  - Gold/silver/metals news as impacting GC, SI and major miners/ETFs,
  - FX and dollar news as impacting DXY, EURUSD, USDJPY, and related FX futures,
  - Rates/bonds/Fed/treasury news as impacting ZN, ZB, TY, TLT and related bond futures.
- Company- or sector-specific headlines should at minimum include the directly mentioned ticker(s) and strongly implied peers (e.g. NVDA + AMD + SMH for chip tariffs).
- If a headline is nearly irrelevant for markets, you may set marketScore close to 0, but still add the company or sector symbol(s) if any are referenced.
- Avoid leaving symbolImpacts empty unless the headline is purely non-financial with no plausible market effect.
- Do not include any additional text outside the JSON.`;

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content:
            'You are a macro hedge fund trader and cross-asset news analyst. You must always derive directional biases from the news, quantifying how bullish or bearish each symbol is. Always respond with valid JSON only.',
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.2,
      max_tokens: 1200,
      response_format: {
        type: 'json_object',
      },
    });

    const raw = completion.choices[0].message.content?.trim() || '{}';
    let parsed: any;
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = JSON.parse(jsonrepair(raw));
    }

    const articleImpactsRaw: any[] = Array.isArray(parsed.articleImpacts) ? parsed.articleImpacts : [];

    const articleImpacts: ArticleImpact[] = articleImpactsRaw
      .map((item: any): ArticleImpact | null => {
        const index = Number.isFinite(Number(item.index)) ? Number(item.index) : 0;
        if (!index || index < 1 || index > selectedNews.length) {
          return null;
        }

        const marketScore = Math.max(-10, Math.min(10, Number(item.marketScore) || 0));

        const symbolImpacts: ArticleSymbolImpact[] = Array.isArray(item.symbolImpacts)
          ? item.symbolImpacts
              .map((s: any): ArticleSymbolImpact | null => {
                const symbol = (s.symbol || '').toString().toUpperCase().trim();
                if (!symbol) {
                  return null;
                }

                const assetClassRaw = (s.assetClass || '').toString().toLowerCase();
                const assetClass: AssetClass =
                  assetClassRaw === 'stock' ||
                  assetClassRaw === 'crypto' ||
                  assetClassRaw === 'future' ||
                  assetClassRaw === 'index' ||
                  assetClassRaw === 'etf' ||
                  assetClassRaw === 'fx'
                    ? (assetClassRaw as AssetClass)
                    : 'other';

                const score = Math.max(-10, Math.min(10, Number(s.score) || 0));
                const reasoning = (s.reasoning || '').toString().trim() || 'No reasoning provided.';

                return {
                  symbol,
                  assetClass,
                  score,
                  reasoning,
                };
              })
              .filter(Boolean) as ArticleSymbolImpact[]
          : [];

        return {
          index,
          marketScore,
          symbolImpacts,
        };
      })
      .filter(Boolean) as ArticleImpact[];

    const symbolMap = new Map<
      string,
      {
        assetClass: AssetClass;
        scores: number[];
        headlines: Set<number>;
        reasons: string[];
      }
    >();

    const marketScores: number[] = [];

    articleImpacts.forEach(impact => {
      const clampedMarketScore = Math.max(-10, Math.min(10, impact.marketScore));
      marketScores.push(clampedMarketScore);

      impact.symbolImpacts.forEach(sym => {
        const key = sym.symbol;
        const existing =
          symbolMap.get(key) ||
          {
            assetClass: sym.assetClass,
            scores: [],
            headlines: new Set<number>(),
            reasons: [],
          };

        existing.assetClass = sym.assetClass;
        existing.scores.push(sym.score);
        existing.headlines.add(impact.index);
        existing.reasons.push(sym.reasoning);

        symbolMap.set(key, existing);
      });
    });

    let aggregateMarketScore = 0;
    if (marketScores.length) {
      const weightedSum = marketScores.reduce((sum, score) => sum + score * Math.abs(score), 0);
      const weight = marketScores.reduce((sum, score) => sum + Math.abs(score), 0) || 1;
      aggregateMarketScore = Math.max(-10, Math.min(10, weightedSum / weight));
    }

    let marketSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
    if (aggregateMarketScore > 1) {
      marketSentiment = 'BULLISH';
    } else if (aggregateMarketScore < -1) {
      marketSentiment = 'BEARISH';
    }

    const symbolRatings: SymbolSentimentRating[] = [];

    symbolMap.forEach((value, symbol) => {
      if (!value.scores.length) {
        return;
      }

      const weightedSum = value.scores.reduce((sum, score) => sum + score * Math.abs(score), 0);
      const weight = value.scores.reduce((sum, score) => sum + Math.abs(score), 0) || 1;
      const avgScore = Math.max(-10, Math.min(10, weightedSum / weight));

      let sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
      if (avgScore > 1) {
        sentiment = 'BULLISH';
      } else if (avgScore < -1) {
        sentiment = 'BEARISH';
      }

      const supportingHeadlines = Array.from(value.headlines).sort((a, b) => a - b);
      const reasoning = `Aggregated score ${avgScore.toFixed(
        1
      )} from ${value.scores.length} headline impact(s): ${value.reasons.slice(0, 3).join(' | ')}`;

      symbolRatings.push({
        symbol,
        assetClass: value.assetClass,
        score: avgScore,
        sentiment,
        reasoning,
        supportingHeadlines,
      });
    });

    // Derive futures sentiment from related index/ETF symbols when not explicitly provided
    const FUTURES_DERIVATIONS: Array<{ target: string; from: string[] }> = [
      { target: 'ES', from: ['SPY', 'SPX'] }, // S&P 500 -> ES futures
      { target: 'NQ', from: ['QQQ', 'NDX'] }, // Nasdaq 100 -> NQ futures
      { target: 'RTY', from: ['IWM', 'RUT'] }, // Russell 2000 -> RTY futures
      { target: 'NKD', from: ['N225', 'NI225', 'NKY'] }, // Nikkei 225 -> Nikkei futures
    ];

    const existingSymbols = new Set(symbolRatings.map(r => r.symbol));

    FUTURES_DERIVATIONS.forEach(rule => {
      if (existingSymbols.has(rule.target)) {
        return;
      }

      const contributors = symbolRatings.filter(r => rule.from.includes(r.symbol));
      if (!contributors.length) {
        return;
      }

      const scores: number[] = [];
      const headlinesSet = new Set<number>();
      const reasons: string[] = [];

      contributors.forEach(c => {
        scores.push(c.score);
        c.supportingHeadlines.forEach(h => headlinesSet.add(h));
        reasons.push(`${c.symbol}: ${c.reasoning}`);
      });

      const weightedSum = scores.reduce((sum, score) => sum + score * Math.abs(score), 0);
      const weight = scores.reduce((sum, score) => sum + Math.abs(score), 0) || 1;
      const avgScore = Math.max(-10, Math.min(10, weightedSum / weight));

      let sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
      if (avgScore > 1) {
        sentiment = 'BULLISH';
      } else if (avgScore < -1) {
        sentiment = 'BEARISH';
      }

      const supportingHeadlines = Array.from(headlinesSet).sort((a, b) => a - b);
      const reasoning = `Derived from ${contributors
        .map(c => c.symbol)
        .join(', ')} news: ${reasons.slice(0, 3).join(' | ')}`;

      symbolRatings.push({
        symbol: rule.target,
        assetClass: 'future',
        score: avgScore,
        sentiment,
        reasoning,
        supportingHeadlines,
      });
      existingSymbols.add(rule.target);
    });

    // First pass: enrich near-zero scores using keyword-based stats and reasoning text
    const bearishHints = [
      'bearish',
      'drop',
      'crater',
      'craters',
      'tumble',
      'tumbles',
      'selloff',
      'sell-off',
      'outflow',
      'outflows',
      'disappointment',
      'miss',
      'cuts',
      'cut',
      'lower',
      'trim',
      'trims',
      'reduces stake',
      'reduces',
    ];
    const bullishHints = [
      'bullish',
      'surge',
      'rally',
      'record high',
      'beats',
      'beat',
      'raises guidance',
      'raises forecast',
      'initiates position',
      'opens position',
      'adds to stake',
      'accumulat',
    ];

    symbolRatings.forEach(r => {
      if (Math.abs(r.score) > 1e-3) {
        return;
      }

      const stats = keywordStats.get(r.symbol);
      const reasoningText = r.reasoning.toLowerCase();

      const countHits = (hints: string[]) =>
        hints.reduce((acc, h) => (reasoningText.includes(h) ? acc + 1 : acc), 0);

      let bullishWeight = 0;
      let bearishWeight = 0;

      if (stats) {
        bullishWeight += stats.bullish * 2 + stats.positiveHits;
        bearishWeight += stats.bearish * 2 + stats.negativeHits;
      }

      bullishWeight += countHits(bullishHints);
      bearishWeight += countHits(bearishHints);

      if (bullishWeight === 0 && bearishWeight === 0) {
        return;
      }

      if (bullishWeight > bearishWeight) {
        const mag = Math.min(5, 1 + bullishWeight);
        r.score = mag;
        r.sentiment = 'BULLISH';
      } else if (bearishWeight > bullishWeight) {
        const mag = Math.min(5, 1 + bearishWeight);
        r.score = -mag;
        r.sentiment = 'BEARISH';
      }
    });

    // Rescale scores to use more of the -10..+10 range and avoid clustering
    const nonZero = symbolRatings.filter(r => r.score !== 0);
    let maxAbs = 0;
    nonZero.forEach(r => {
      const abs = Math.abs(r.score);
      if (abs > maxAbs) {
        maxAbs = abs;
      }
    });

    if (maxAbs > 0) {
      symbolRatings.forEach(r => {
        if (r.score === 0) {
          return;
        }

        // Normalize to [-10, 10] relative to the strongest symbol
        let scaled = (r.score / maxAbs) * 10;

        // Ensure a minimum magnitude so small but non-zero scores are visible
        if (Math.abs(scaled) < 2) {
          scaled = scaled > 0 ? 2 : -2;
        }

        // Clamp and round to one decimal for readability
        const finalScore = Math.max(-10, Math.min(10, Number(scaled.toFixed(1))));

        r.score = finalScore;

        if (finalScore > 1) {
          r.sentiment = 'BULLISH';
        } else if (finalScore < -1) {
          r.sentiment = 'BEARISH';
        } else {
          r.sentiment = 'NEUTRAL';
        }
      });
    }

    symbolRatings.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));

    const result: NewsImpactAnalysis = {
      generatedAt: new Date().toISOString(),
      newsWindowDays: windowDays,
      marketScore: aggregateMarketScore,
      marketSentiment,
      symbolRatings,
      articleImpacts,
      totalNewsAnalyzed: windowedNews.length,
    };

    console.log(
      `   ✅ Multi-asset news impact: marketScore=${result.marketScore.toFixed(
        1
      )} (${result.marketSentiment}), symbols=${result.symbolRatings.length}, articles=${result.articleImpacts.length}`
    );

    return result;
  } catch (error: any) {
    console.error('   ❌ Error generating multi-asset news impact analysis:', error.message);

    return {
      generatedAt: new Date().toISOString(),
      newsWindowDays: windowDays,
      marketScore: 0,
      marketSentiment: 'NEUTRAL',
      symbolRatings: [],
      articleImpacts: [],
      totalNewsAnalyzed: 0,
    };
  }
}
