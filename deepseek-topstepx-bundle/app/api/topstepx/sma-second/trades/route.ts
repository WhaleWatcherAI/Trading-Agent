import { NextResponse } from 'next/server';
import { getStrategyTrades } from '@/lib/server/topstepxSmaSecondData';

export async function GET() {
  const trades = await getStrategyTrades();
  return NextResponse.json(trades);
}
