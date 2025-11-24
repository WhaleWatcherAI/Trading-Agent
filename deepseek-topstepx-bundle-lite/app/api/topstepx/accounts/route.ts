import { NextResponse } from 'next/server';
import { fetchTopstepXAccounts } from '@/lib/topstepx';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const accounts = await fetchTopstepXAccounts(false);
    return NextResponse.json({ accounts });
  } catch (error: any) {
    console.error('[topstepx accounts] failed', error);
    return NextResponse.json(
      { error: 'Failed to fetch TopstepX accounts', details: error?.message },
      { status: 500 },
    );
  }
}
