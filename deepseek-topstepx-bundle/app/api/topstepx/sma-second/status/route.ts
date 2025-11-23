import { NextResponse } from 'next/server';
import { getTopstepxSmaSecondStatus } from '@/lib/server/topstepxSmaSecondProcess';
import { getStrategyTrades } from '@/lib/server/topstepxSmaSecondData';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const [status, trades] = await Promise.all([
    Promise.resolve(getTopstepxSmaSecondStatus()),
    getStrategyTrades(),
  ]);

  return NextResponse.json({
    status,
    trades,
    symbol:
      process.env.TOPSTEPX_SECOND_SMA_SYMBOL ||
      process.env.TOPSTEPX_SMA_SYMBOL ||
      'MESZ5',
    contractId:
      process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID ||
      process.env.TOPSTEPX_CONTRACT_ID ||
      null,
    multiplier: Number(process.env.TOPSTEPX_SECOND_CONTRACT_MULTIPLIER || '50'),
  });
}
