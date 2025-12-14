import { NextRequest, NextResponse } from 'next/server';
import { getIvRankHistory } from '@/lib/unusualwhales';

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const ticker = params.get('ticker') || 'SPY';
  const date = params.get('date');

  try {
    const history = await getIvRankHistory(ticker, date);

    return NextResponse.json({
      ticker,
      date: date || 'latest',
      entries: history.length,
      data: history,
    });
  } catch (error: any) {
    console.error('Test IV history error:', error);
    return NextResponse.json(
      {
        error: 'Failed to fetch IV rank history',
        details: error?.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
