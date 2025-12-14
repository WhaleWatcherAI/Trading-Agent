import { NextResponse } from 'next/server';
import { fetchTopstepXFuturesBars } from '@/lib/topstepx';
import { resolveTopstepxContractId } from '@/lib/server/topstepxResolver';
import { getTopstepxFeed } from '@/lib/server/topstepxFeed';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const feed = await getTopstepxFeed();
    const snapshot = feed.getSnapshot();
    if (snapshot.price && snapshot.timestamp && snapshot.candles?.length) {
      const lastCandle = snapshot.candles[snapshot.candles.length - 1];
      return NextResponse.json({
        price: snapshot.price,
        open: lastCandle.open,
        high: lastCandle.high,
        low: lastCandle.low,
        timestamp: snapshot.timestamp,
        contractId:
          process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID ||
          process.env.TOPSTEPX_CONTRACT_ID ||
          null,
        symbol:
          process.env.TOPSTEPX_SECOND_SMA_SYMBOL ||
          process.env.TOPSTEPX_SMA_SYMBOL ||
          'MESZ5',
      });
    }

    const contractId = await resolveTopstepxContractId();
    const end = new Date();
    const start = new Date(end.getTime() - 15_000);

    const bars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: start.toISOString(),
      endTime: end.toISOString(),
      unit: 1,
      unitNumber: 1,
      limit: 5,
    });

    const ordered = [...bars].reverse();
    const latest = ordered[ordered.length - 1] ?? ordered[0];
    if (!latest) {
      return NextResponse.json({ error: 'No recent bars available' }, { status: 404 });
    }

    return NextResponse.json({
      price: latest.close,
      open: latest.open,
      high: latest.high,
      low: latest.low,
      timestamp: latest.timestamp,
      contractId,
      symbol:
        process.env.TOPSTEPX_SECOND_SMA_SYMBOL ||
        process.env.TOPSTEPX_SMA_SYMBOL ||
        'MESZ5',
    });
  } catch (error: any) {
    console.error('[topstepx price] failed:', error);
    return NextResponse.json(
      { error: 'Failed to fetch latest price', details: error?.message },
      { status: 500 },
    );
  }
}
