import { describe, it, expect, beforeAll, afterAll, beforeEach, vi } from 'vitest';
import { analyzeDailyMarketSentiment, filterDailyNews, formatNewsDigest } from '../lib/sentimentAnalyzer';
import type { NewsItem } from '@/types';

const openAiMocks = vi.hoisted(() => ({
  mockCreate: vi.fn(),
}));

vi.mock('openai', () => ({
  default: vi.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: openAiMocks.mockCreate,
      },
    },
  })),
}));

vi.mock('../lib/unusualwhales', () => ({
  getNewsHeadlines: vi.fn(),
}));

import { getNewsHeadlines } from '../lib/unusualwhales';

const mockedGetNewsHeadlines = vi.mocked(getNewsHeadlines);

const BASE_TIME = new Date('2025-01-02T12:00:00Z');

function buildNews(overrides: Partial<NewsItem>): NewsItem {
  return {
    title: 'Sample Headline',
    summary: 'Sample summary text.',
    url: 'https://example.com/article',
    source: 'Example',
    symbols: ['SAMP'],
    sentiment: 'bullish',
    importance: 0.5,
    timestamp: BASE_TIME,
    ...overrides,
  };
}

beforeAll(() => {
  vi.useFakeTimers();
  vi.setSystemTime(BASE_TIME);
});

afterAll(() => {
  vi.useRealTimers();
});

beforeEach(() => {
  openAiMocks.mockCreate.mockReset();
  mockedGetNewsHeadlines.mockReset();
  process.env.OPENAI_API_KEY = 'test-key';
});

describe('filterDailyNews', () => {
  it('filters out news older than 24 hours', () => {
    const recent = buildNews({ title: 'Recent news', timestamp: new Date('2025-01-02T08:00:00Z') });
    const stale = buildNews({ title: 'Old news', timestamp: new Date('2024-12-31T08:00:00Z') });

    const filtered = filterDailyNews([recent, stale]);

    expect(filtered).toHaveLength(1);
    expect(filtered[0].title).toBe('Recent news');
  });
});

describe('formatNewsDigest', () => {
  it('formats a digest with headline metadata', () => {
    const digest = formatNewsDigest([
      buildNews({
        title: 'Apple announces record earnings',
        summary: 'Apple reports best quarter ever.',
        symbols: ['AAPL'],
        sentiment: 'bullish',
      }),
    ]);

    expect(digest).toContain('1. Headline: Apple announces record earnings');
    expect(digest).toContain('Summary: Apple reports best quarter ever.');
    expect(digest).toContain('Tickers Mentioned: AAPL');
    expect(digest).toContain('UW Sentiment: BULLISH');
  });
});

describe('analyzeDailyMarketSentiment', () => {
  it('returns structured market sentiment insights', async () => {
    const newsItems: NewsItem[] = [
      buildNews({
        title: 'Apple delivers strong iPhone sales',
        summary: 'Holiday sales beat expectations.',
        symbols: ['AAPL'],
        sentiment: 'bullish',
        importance: 0.9,
        timestamp: new Date('2025-01-02T09:00:00Z'),
      }),
      buildNews({
        title: 'Microsoft expands cloud business',
        summary: 'Azure growth accelerates.',
        symbols: ['MSFT'],
        sentiment: 'bullish',
        importance: 0.7,
        timestamp: new Date('2025-01-02T07:30:00Z'),
      }),
    ];

    mockedGetNewsHeadlines.mockResolvedValue(newsItems);

    openAiMocks.mockCreate.mockResolvedValue({
      choices: [
        {
          message: {
            content: JSON.stringify({
              marketSentiment: 'bullish',
              summary: 'Tech sector news remains broadly positive.',
              insights: [
                {
                  symbol: 'AAPL',
                  sentiment: 'bullish',
                  score: 8,
                  confidence: 75,
                  reasoning: 'Strong product sales and demand outperformance.',
                  supportingHeadlines: [1],
                },
                {
                  symbol: 'MSFT',
                  sentiment: 'bullish',
                  score: 7,
                  confidence: 70,
                  reasoning: 'Cloud business expansion.',
                  supportingHeadlines: [2],
                },
              ],
            }),
          },
        },
      ],
    });

    const result = await analyzeDailyMarketSentiment();

    expect(mockedGetNewsHeadlines).toHaveBeenCalledTimes(1);
    expect(openAiMocks.mockCreate).toHaveBeenCalledTimes(1);

    expect(result.marketSentiment).toBe('BULLISH');
    expect(result.summary).toBe('Tech sector news remains broadly positive.');
    expect(result.totalNewsAnalyzed).toBe(2);
    expect(result.insights).toHaveLength(2);

    const appleInsight = result.insights[0];
    expect(appleInsight.symbol).toBe('AAPL');
    expect(appleInsight.sentiment).toBe('BULLISH');
    expect(appleInsight.supportingHeadlines[0]).toContain('Apple delivers strong iPhone sales');
  });
});
