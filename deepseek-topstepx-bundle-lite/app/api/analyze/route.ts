import { NextRequest, NextResponse } from 'next/server';
import { analyzeMultiStrategy, analyzeSingleStrategy } from '@/lib/advancedAgent';
import { Strategy } from '@/lib/scoringEngine';
import { AnalysisRequest } from '@/types';

export async function POST(request: NextRequest) {
  try {
    const body: AnalysisRequest = await request.json();

    const symbols = body.symbols || ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA'];
    const threshold = body.threshold || 3; // Lowered for testing with limited data

    // Multi-strategy analysis
    if (!body.strategy || body.strategy === 'all') {
      const strategies: Strategy[] = ['scalp', 'intraday', 'swing', 'leap'];
      const result = await analyzeMultiStrategy(symbols, strategies, threshold);
      return NextResponse.json(result);
    }

    // Single strategy analysis
    const result = await analyzeSingleStrategy(symbols, body.strategy as Strategy, threshold);
    return NextResponse.json({ recommendations: result });
  } catch (error: any) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: 'Failed to analyze market data', details: error.message },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const strategyParam = searchParams.get('strategy') as string | null;
    const threshold = parseInt(searchParams.get('threshold') || '3'); // Lowered for testing
    const symbols = searchParams.get('symbols')?.split(',').filter(Boolean) || ['SPY', 'QQQ', 'AAPL'];

    // Multi-strategy analysis
    if (!strategyParam || strategyParam === 'all') {
      const strategies: Strategy[] = ['scalp', 'intraday', 'swing', 'leap'];
      const result = await analyzeMultiStrategy(symbols, strategies, threshold);
      return NextResponse.json(result);
    }

    // Single strategy analysis
    const result = await analyzeSingleStrategy(symbols, strategyParam as Strategy, threshold);
    return NextResponse.json({ recommendations: result });
  } catch (error: any) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: 'Failed to analyze market data', details: error.message },
      { status: 500 }
    );
  }
}
