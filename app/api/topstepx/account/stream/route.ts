import { NextRequest, NextResponse } from 'next/server';
import { getTopstepxAccountFeed } from '@/lib/server/topstepxAccountFeed';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const encoder = new TextEncoder();

export async function GET(request: NextRequest) {
  const idParam = request.nextUrl.searchParams.get('id');
  if (!idParam) {
    return NextResponse.json({ error: 'Missing id' }, { status: 400 });
  }
  const accountId = Number(idParam);
  if (!Number.isFinite(accountId)) {
    return NextResponse.json({ error: 'Invalid id' }, { status: 400 });
  }

  const feed = await getTopstepxAccountFeed(accountId);
  let cleanup: (() => void) | null = null;

  const stream = new ReadableStream({
    start(controller) {
      const abortSignal = request.signal;
      let closed = false;
      let abortHandler: (() => void) | null = null;
      let keepAlive: NodeJS.Timeout | null = null;

      const send = (payload: any) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };

      const snapshot = feed.getSnapshot();
      try {
        send({ type: 'snapshot', ...snapshot });
      } catch (err) {
        console.error('[account-sse] failed to send snapshot', err);
      }

      const handler = (payload: any) => {
        try {
          send({ type: 'update', ...payload });
        } catch (err) {
          console.error('[account-sse] failed to send update', err);
        }
      };
      const close = () => {
        if (closed) return;
        closed = true;
        if (keepAlive) {
          clearInterval(keepAlive);
          keepAlive = null;
        }
        feed.off('update', handler);
        if (abortHandler) {
          abortSignal.removeEventListener('abort', abortHandler);
        }
        try {
          controller.close();
        } catch (err) {
          console.warn('[account-sse] controller close failed', err);
        }
      };
      feed.on('update', handler);

      keepAlive = setInterval(() => {
        try {
          controller.enqueue(encoder.encode(': keep-alive\n\n'));
        } catch (err) {
          console.warn('[account-sse] keep-alive failed', err);
          close();
        }
      }, 15000);

      abortHandler = () => close();
      abortSignal.addEventListener('abort', abortHandler);
      cleanup = close;
    },
    cancel() {
      if (cleanup) {
        cleanup();
        cleanup = null;
      }
    },
  });

  return new NextResponse(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}
