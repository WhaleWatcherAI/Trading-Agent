import { NextRequest, NextResponse } from 'next/server';
import {
  flattenTopstepxSmaSecondPosition,
  startTopstepxSmaSecondProcess,
  stopTopstepxSmaSecondProcess,
} from '@/lib/server/topstepxSmaSecondProcess';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => ({}));
  const action = body?.action;

  if (!action) {
    return NextResponse.json({ error: 'Missing action' }, { status: 400 });
  }

  if (action === 'start') {
    const accountId = body?.accountId;
    const parsedAccount = accountId != null ? Number(accountId) : undefined;
    if (parsedAccount != null && !Number.isFinite(parsedAccount)) {
      return NextResponse.json({ error: 'Invalid accountId' }, { status: 400 });
    }
    const result = startTopstepxSmaSecondProcess({
      accountId: parsedAccount,
    });
    return NextResponse.json(result, { status: result.success ? 200 : 400 });
  }

  if (action === 'stop') {
    const result = stopTopstepxSmaSecondProcess();
    return NextResponse.json(result, { status: result.success ? 200 : 400 });
  }

  if (action === 'flatten') {
    const result = flattenTopstepxSmaSecondPosition();
    return NextResponse.json(result, { status: result.success ? 200 : 400 });
  }

  return NextResponse.json({ error: `Unsupported action ${action}` }, { status: 400 });
}
