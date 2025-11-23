import { NextResponse } from 'next/server';
import { listCachedFlowDates } from '@/lib/regimeBacktester';

export async function GET() {
  try {
    const dates = await listCachedFlowDates();
    return NextResponse.json({ dates });
  } catch (error: any) {
    console.error('Failed to list cached backtest dates:', error);
    return NextResponse.json(
      {
        error: 'Unable to list cached backtest dates',
        details: error?.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
