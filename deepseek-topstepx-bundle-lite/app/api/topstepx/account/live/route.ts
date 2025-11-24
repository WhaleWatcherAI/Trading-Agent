import { NextRequest, NextResponse } from 'next/server';
import { getTopstepxAccountFeed } from '@/lib/server/topstepxAccountFeed';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const idParam = request.nextUrl.searchParams.get('id');
  if (!idParam) {
    return NextResponse.json({ error: 'Missing id' }, { status: 400 });
  }
  const accountId = Number(idParam);
  if (!Number.isFinite(accountId)) {
    return NextResponse.json({ error: 'Invalid id' }, { status: 400 });
  }

  try {
    const feed = await getTopstepxAccountFeed(accountId);
    const snapshot = feed.getSnapshot();
    return NextResponse.json(snapshot);
  } catch (error: any) {
    console.error('[topstepx account live] failed', error);
    return NextResponse.json(
      { error: 'Failed to fetch TopstepX account snapshot', details: error?.message },
      { status: 500 },
    );
  }
}
