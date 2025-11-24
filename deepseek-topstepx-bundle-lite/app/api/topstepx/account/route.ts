import { NextRequest, NextResponse } from 'next/server';
import { fetchTopstepXAccounts } from '@/lib/topstepx';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const idParam = request.nextUrl.searchParams.get('id');
  if (!idParam) {
    return NextResponse.json({ error: 'Missing id' }, { status: 400 });
  }

  const id = Number(idParam);
  if (!Number.isFinite(id)) {
    return NextResponse.json({ error: 'Invalid id' }, { status: 400 });
  }

  try {
    const accounts = await fetchTopstepXAccounts(false);
    const account = accounts.find(acc => acc.id === id);
    if (!account) {
      return NextResponse.json({ error: 'Account not found' }, { status: 404 });
    }
    return NextResponse.json(account);
  } catch (error: any) {
    console.error('[topstepx account] failed', error);
    return NextResponse.json(
      { error: 'Failed to fetch TopstepX account', details: error?.message },
      { status: 500 },
    );
  }
}
