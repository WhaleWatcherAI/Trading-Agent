import { NextRequest, NextResponse } from 'next/server';
import {
  getUnusualOptionsActivity,
  getInstitutionalActivity,
  getNewsHeadlines,
} from '@/lib/unusualwhales';

/**
 * Market Scanner - Find stocks with unusual activity
 *
 * Scans for:
 * - High unusual options volume
 * - Large institutional trades
 * - Recent news catalysts
 *
 * Filters by market cap (optional)
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const minVolume = parseInt(searchParams.get('minVolume') || '1000');
    const minPremium = parseInt(searchParams.get('minPremium') || '100000'); // $100K
    const limit = parseInt(searchParams.get('limit') || '20');
    const marketCap = searchParams.get('marketCap') || 'all'; // small, mid, large, all

    console.log(`🔍 Scanning market for unusual activity (min volume: ${minVolume}, min premium: $${minPremium})`);

    // Fetch all unusual activity
    const [optionsActivity, institutionalActivity, news] = await Promise.all([
      getUnusualOptionsActivity().catch(() => []),
      getInstitutionalActivity().catch(() => []),
      getNewsHeadlines().catch(() => []),
    ]);

    // Score each symbol by activity
    const symbolScores = new Map<string, {
      symbol: string;
      optionsScore: number;
      institutionalScore: number;
      newsScore: number;
      totalScore: number;
      details: {
        optionsFlow: number; // Number of unusual trades
        totalPremium: number; // Total options premium
        institutionalValue: number; // Total institutional value
        newsCount: number; // Number of recent news items
        latestActivity: Date;
      };
    }>();

    // Score options activity
    optionsActivity.forEach(trade => {
      if (trade.volume < minVolume || trade.premium < minPremium) return;

      const existing = symbolScores.get(trade.underlying) || {
        symbol: trade.underlying,
        optionsScore: 0,
        institutionalScore: 0,
        newsScore: 0,
        totalScore: 0,
        details: {
          optionsFlow: 0,
          totalPremium: 0,
          institutionalValue: 0,
          newsCount: 0,
          latestActivity: trade.timestamp,
        },
      };

      // Weight by premium and unusual flag
      const weight = trade.unusual ? 3 : 1;
      existing.optionsScore += weight;
      existing.details.optionsFlow += 1;
      existing.details.totalPremium += trade.premium;

      if (trade.timestamp > existing.details.latestActivity) {
        existing.details.latestActivity = trade.timestamp;
      }

      symbolScores.set(trade.underlying, existing);
    });

    // Score institutional activity
    institutionalActivity.forEach(trade => {
      const existing = symbolScores.get(trade.symbol) || {
        symbol: trade.symbol,
        optionsScore: 0,
        institutionalScore: 0,
        newsScore: 0,
        totalScore: 0,
        details: {
          optionsFlow: 0,
          totalPremium: 0,
          institutionalValue: 0,
          newsCount: 0,
          latestActivity: trade.timestamp,
        },
      };

      // Weight by trade size (>$10M = 3x, >$1M = 2x)
      const weight = trade.value > 10_000_000 ? 3 : trade.value > 1_000_000 ? 2 : 1;
      existing.institutionalScore += weight;
      existing.details.institutionalValue += trade.value;

      if (trade.timestamp > existing.details.latestActivity) {
        existing.details.latestActivity = trade.timestamp;
      }

      symbolScores.set(trade.symbol, existing);
    });

    // Score news
    news.forEach(item => {
      item.symbols.forEach(symbol => {
        const existing = symbolScores.get(symbol) || {
          symbol,
          optionsScore: 0,
          institutionalScore: 0,
          newsScore: 0,
          totalScore: 0,
          details: {
            optionsFlow: 0,
            totalPremium: 0,
            institutionalValue: 0,
            newsCount: 0,
            latestActivity: item.timestamp,
          },
        };

        existing.newsScore += 1;
        existing.details.newsCount += 1;

        if (item.timestamp > existing.details.latestActivity) {
          existing.details.latestActivity = item.timestamp;
        }

        symbolScores.set(symbol, existing);
      });
    });

    // Calculate total scores
    symbolScores.forEach((data, symbol) => {
      data.totalScore = (
        data.optionsScore * 3 +    // Options flow is most important
        data.institutionalScore * 2 + // Institutional is second
        data.newsScore * 1           // News is supporting
      );
    });

    // Filter and sort
    const results = Array.from(symbolScores.values())
      .filter(data => data.totalScore > 0)
      .sort((a, b) => b.totalScore - a.totalScore)
      .slice(0, limit);

    console.log(`✅ Scanner found ${results.length} symbols with unusual activity`);

    return NextResponse.json({
      symbols: results.map(r => r.symbol),
      details: results,
      scannedAt: new Date().toISOString(),
      filters: {
        minVolume,
        minPremium,
        limit,
        marketCap,
      },
    });

  } catch (error: any) {
    console.error('Scanner error:', error);
    return NextResponse.json(
      { error: 'Failed to scan market', details: error.message },
      { status: 500 }
    );
  }
}
