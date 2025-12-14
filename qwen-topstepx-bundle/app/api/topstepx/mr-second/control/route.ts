import { NextRequest, NextResponse } from 'next/server';
import {
  flattenTopstepxMrPosition,
  startTopstepxMrProcess,
  stopTopstepxMrProcess,
} from '@/lib/server/topstepxMrProcess';

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
    const result = startTopstepxMrProcess({ accountId: parsedAccount });
    return NextResponse.json(result, { status: result.success ? 200 : 400 });
  }

  if (action === 'stop') {
    const result = stopTopstepxMrProcess();
    return NextResponse.json(result, { status: result.success ? 200 : 400 });
  }

  if (action === 'flatten') {
    const result = flattenTopstepxMrPosition();
    return NextResponse.json(result, { status: result.success ? 200 : 400 });
  }

  return NextResponse.json({ error: `Unsupported action ${action}` }, { status: 400 });
}
