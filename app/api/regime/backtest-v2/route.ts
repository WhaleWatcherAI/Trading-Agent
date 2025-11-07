import { NextRequest, NextResponse } from 'next/server';
import { runRegimeBacktestV2 } from '@/lib/regimeBacktesterV2';

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const date = params.get('date');

  if (!date) {
    return NextResponse.json(
      { error: 'Missing required "date" query parameter (expected YYYY-MM-DD).' },
      { status: 400 },
    );
  }

  const modeParam = params.get('mode');
  const mode = modeParam === 'swing' || modeParam === 'leaps' ? modeParam : 'scalp';

  const symbolsParam = params.get('symbols');
  const symbols = symbolsParam && symbolsParam.trim()
    ? symbolsParam
        .split(',')
        .map(symbol => symbol.trim().toUpperCase())
        .filter(Boolean)
    : [
        'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA',
        'MSFT', 'AMZN', 'META', 'GOOGL', 'AMD',
        'NFLX', 'COIN', 'PLTR', 'MARA', 'MSTR',
        'IWM', 'TLT', 'GLD', 'SLV', 'UNG'
      ]; // Major liquid tickers for testing

  const intervalParam = params.get('interval');
  const intervalMinutes = intervalParam ? parseInt(intervalParam, 10) : 1;

  try {
    const result = await runRegimeBacktestV2({
      date,
      symbols,
      mode,
      intervalMinutes,
    });

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Regime backtest V2 error:', error);
    return NextResponse.json(
      {
        error: 'Failed to execute regime backtest V2',
        details: error?.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
