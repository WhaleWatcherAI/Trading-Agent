import { NextRequest, NextResponse } from 'next/server';
import { runSmaCrossoverStrategy } from '@/lib/smaCrossoverAgent';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbolParam = searchParams.get('symbol');
  const smaPeriodParam = searchParams.get('smaPeriod');
  const dateParam = searchParams.get('date');

  if (!symbolParam) {
    return NextResponse.json(
      {
        error: 'Missing symbol parameter',
      },
      { status: 400 },
    );
  }

  const symbol = symbolParam.toUpperCase();
  const smaPeriod = smaPeriodParam ? parseInt(smaPeriodParam, 10) : undefined;
  const date = dateParam || undefined;

  try {
    const signal = await runSmaCrossoverStrategy({
      symbol,
      smaPeriod,
      date,
    });

    if (signal) {
      return NextResponse.json(signal);
    } else {
      return NextResponse.json(
        {
          message: 'No SMA crossover signal generated for the given symbol.',
        },
        { status: 200 },
      );
    }
  } catch (error: any) {
    console.error('SMA Crossover agent error:', error);
    return NextResponse.json(
      {
        error: 'Failed to run SMA Crossover strategy',
        details: error.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
