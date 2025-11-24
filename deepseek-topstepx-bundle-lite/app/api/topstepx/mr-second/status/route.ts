import { NextResponse } from 'next/server';
import { getTopstepxMrStatus } from '@/lib/server/topstepxMrProcess';
import { getMrTrades } from '@/lib/server/topstepxMrData';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const [status, trades] = await Promise.all([
    Promise.resolve(getTopstepxMrStatus()),
    getMrTrades(),
  ]);

  return NextResponse.json({
    status,
    trades,
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
    multiplier: Number(process.env.TOPSTEPX_MR_CONTRACT_MULTIPLIER || '5'),
  });
}
