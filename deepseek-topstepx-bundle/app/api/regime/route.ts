import { NextRequest, NextResponse } from 'next/server';
import { analyzeVolatilityRegime } from '@/lib/regimeAgent';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbolsParam = searchParams.get('symbols');
  const modeParam = searchParams.get('mode');

  const symbols = symbolsParam
    ? symbolsParam
        .split(',')
        .map(symbol => symbol.trim().toUpperCase())
        .filter(Boolean)
    : undefined;

  const mode = modeParam === 'swing' || modeParam === 'leaps' ? modeParam : 'scalp';

  try {
    const result = await analyzeVolatilityRegime({
      symbols,
      mode,
    });

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Regime agent error:', error);
    return NextResponse.json(
      {
        error: 'Failed to analyze volatility regimes',
        details: error.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
