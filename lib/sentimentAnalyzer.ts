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

export function filterDailyNews(allNews: NewsItem[], date?: string): NewsItem[] {
  const now = new Date();
  const end = date ? new Date(`${date}T23:59:59Z`) : now;
  const start = date ? new Date(`${date}T00:00:00Z`) : new Date(end.getTime() - 24 * 60 * 60 * 1000);

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
UW Sentiment: ${sentimentTag}
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
    const dailyNews = filterDailyNews(allNews, date);

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

    const prompt = `You are a market news analyst. Review the last 24 hours of Unusual Whales headlines and summarize the overall market sentiment.

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

    const insightSymbols = new Set(baseInsights.map(insight => insight.symbol));
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
