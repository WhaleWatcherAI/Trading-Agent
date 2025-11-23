import { NextRequest, NextResponse } from 'next/server';
import { getTopstepxFeed } from '@/lib/server/topstepxFeed';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const encoder = new TextEncoder();

export async function GET(request: NextRequest) {
  const feed = await getTopstepxFeed();
  let cleanup: (() => void) | null = null;

  const stream = new ReadableStream({
    start(controller) {
      const abortSignal = request.signal;
      let aborted = false;
      let abortHandler: (() => void) | null = null;
      let keepAlive: NodeJS.Timeout | null = null;

      const send = (payload: any) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };

      const snapshot = feed.getSnapshot();
      try {
        send({
          type: 'snapshot',
          ...snapshot,
        });
      } catch (err) {
        console.error('[mr-stream] failed to send bootstrap snapshot', err);
      }

      const handler = (event: any) => {
        try {
          send(event);
        } catch (err) {
          console.error('[mr-stream] failed to send event', err);
        }
      };

      const close = () => {
        if (aborted) return;
        aborted = true;
        if (keepAlive) {
          clearInterval(keepAlive);
          keepAlive = null;
        }
        feed.off('event', handler);
        if (abortHandler) {
          abortSignal.removeEventListener('abort', abortHandler);
        }
        try {
          controller.close();
        } catch (err) {
          console.warn('[mr-stream] controller close failed', err);
        }
      };

      feed.on('event', handler);

      keepAlive = setInterval(() => {
        try {
          controller.enqueue(encoder.encode(': keep-alive\n\n'));
        } catch (err) {
          console.warn('[mr-stream] keep-alive failed', err);
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
