import { NextRequest, NextResponse } from 'next/server';
import { runSmaSentimentBacktest } from '@/lib/smaSentimentBacktester';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, date } = body;

    if (!symbol || !date) {
      return NextResponse.json(
        { error: 'Symbol and date are required' },
        { status: 400 }
      );
    }

    console.log(`\n🚀 Running SMA Sentiment backtest: ${symbol} on ${date}`);

    const result = await runSmaSentimentBacktest({
      symbol,
      date,
      smaPeriod: 20,
      initialCapital: 10000,
      positionSize: 0.1,
      exitAfterMinutes: 30,
    });

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Backtest error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to run backtest' },
      { status: 500 }
    );
  }
}
