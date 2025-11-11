import { NextRequest, NextResponse } from 'next/server';
import { fetchTopstepXFuturesBars } from '@/lib/topstepx';
import { resolveTopstepxContractId } from '@/lib/server/topstepxResolver';
import { getTopstepxFeed } from '@/lib/server/topstepxFeed';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const secondsParam = request.nextUrl.searchParams.get('seconds');
  const seconds = Math.min(Math.max(Number(secondsParam) || 3600, 60), 6 * 60 * 60);

  try {
    const feed = await getTopstepxFeed();
    const snapshot = feed.getSnapshot();
    if (snapshot.candles?.length) {
      const recentCandles = snapshot.candles.slice(-seconds);
      return NextResponse.json({
        symbol:
          process.env.TOPSTEPX_MR_LIVE_SYMBOL ||
          process.env.TOPSTEPX_SECOND_SMA_SYMBOL ||
          process.env.TOPSTEPX_SMA_SYMBOL ||
          'MESZ5',
        contractId:
          process.env.TOPSTEPX_MR_LIVE_CONTRACT_ID ||
          process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID ||
          process.env.TOPSTEPX_CONTRACT_ID ||
          null,
        seconds,
        candles: recentCandles,
      });
    }

    const contractId = await resolveTopstepxContractId();
    const end = new Date();
    const start = new Date(end.getTime() - seconds * 1000);

    const bars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: start.toISOString(),
      endTime: end.toISOString(),
      unit: 1,
      unitNumber: 1,
      limit: seconds,
    });

    const ordered = [...bars].reverse();
    const candles = ordered.map(bar => ({
      time: bar.timestamp,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));

    return NextResponse.json({
      symbol:
        process.env.TOPSTEPX_MR_LIVE_SYMBOL ||
        process.env.TOPSTEPX_SECOND_SMA_SYMBOL ||
        process.env.TOPSTEPX_SMA_SYMBOL ||
        'MESZ5',
      contractId,
      seconds,
      candles,
    });
  } catch (error: any) {
    console.error('[topstepx mr bars] failed:', error);
    return NextResponse.json(
      { error: 'Failed to fetch TopstepX bars', details: error?.message },
      { status: 500 },
    );
  }
}
