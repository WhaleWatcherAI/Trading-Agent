import { NextRequest, NextResponse } from 'next/server';
import { getTodayData, getDateData, getStats } from '@/lib/storage/dataStore';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const date = searchParams.get('date');
    const statsOnly = searchParams.get('stats') === 'true';

    if (statsOnly) {
      const stats = getStats();
      return NextResponse.json(stats);
    }

    const data = date ? getDateData(date) : getTodayData();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Data fetch error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch data', details: error.message },
      { status: 500 }
    );
  }
}
