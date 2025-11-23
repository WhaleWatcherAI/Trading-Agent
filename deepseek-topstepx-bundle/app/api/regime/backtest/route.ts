import { NextRequest, NextResponse } from 'next/server';
import { runRegimeBacktest } from '@/lib/regimeBacktester';

function parseMode(value: string | null): 'scalp' | 'swing' | 'leaps' | undefined {
  if (!value) return undefined;
  if (value === 'scalp' || value === 'swing' || value === 'leaps') {
    return value;
  }
  return undefined;
}

export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const date = params.get('date');

  if (!date) {
    return NextResponse.json(
      { error: 'Missing required "date" query parameter (expected YYYY-MM-DD).' },
      { status: 400 },
    );
  }

  const mode = parseMode(params.get('mode'));
  const intervalRaw = params.get('interval');
  const usePricesParam = params.get('prices');
  const whalePremiumRaw = params.get('whalePremium');
  const whaleVolumeRaw = params.get('whaleVolume');
  const flowParam = params.get('flow');
  const lookbackParam = params.get('lookback');
  const symbolsParam = params.get('symbols');

  let intervalMinutes: number | undefined;
  if (intervalRaw) {
    const parsed = parseInt(intervalRaw, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      intervalMinutes = parsed;
    }
  }

  let liveLookbackMinutes: number | undefined;
  if (lookbackParam) {
    const parsed = parseInt(lookbackParam, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      liveLookbackMinutes = parsed;
    }
  }

  const symbols = symbolsParam
    ? symbolsParam
        .split(',')
        .map(symbol => symbol.trim().toUpperCase())
        .filter(Boolean)
    : undefined;

  const config = {
    date,
    mode: mode || undefined,
    intervalMinutes,
    useTradierPrices: usePricesParam === null ? true : usePricesParam !== 'false',
    whalePremiumThreshold: whalePremiumRaw ? parseFloat(whalePremiumRaw) : undefined,
    whaleVolumeThreshold: whaleVolumeRaw ? parseFloat(whaleVolumeRaw) : undefined,
    fetchLiveFlow: flowParam === 'live' || flowParam === 'true',
    liveLookbackMinutes,
    symbols,
  };

  try {
    const result = await runRegimeBacktest(config);
    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Regime backtest error:', error);
    return NextResponse.json(
      {
        error: 'Failed to execute regime backtest',
        details: error?.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
