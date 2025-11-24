import { NextRequest, NextResponse } from 'next/server';
import { getAccountBalances, getAccountPositions, getOpenOrders } from '@/lib/tradier';
import { AccountSnapshot } from '@/types';

export async function GET(request: NextRequest) {
  try {
    const [balances, positions, orders] = await Promise.all([
      getAccountBalances().catch(() => null),
      getAccountPositions().catch(() => []),
      getOpenOrders().catch(() => []),
    ]);

    const snapshot: AccountSnapshot = {
      balances,
      positions,
      orders,
      fetchedAt: new Date().toISOString(),
    };

    return NextResponse.json(snapshot);
  } catch (error: any) {
    console.error('Failed to fetch account snapshot', error);
    return NextResponse.json(
      { error: 'Failed to fetch account snapshot', details: error.message },
      { status: 500 },
    );
  }
}
