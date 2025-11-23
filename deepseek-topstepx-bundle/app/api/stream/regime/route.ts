import { NextRequest } from 'next/server';
import { subscribeLifecycleUpdates, subscribeLifecycleBroadcast } from '@/lib/lifecycleBus';
import { getLifecycleSnapshot } from '@/lib/regimeLifecycle';

const encoder = new TextEncoder();

const allowedModes = new Set(['scalp', 'swing', 'leaps']);

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const symbolParam = searchParams.get('symbol') || '';
  const modeParam = (searchParams.get('mode') || 'scalp').toLowerCase();
  const broadcastParam = searchParams.get('broadcast') === 'true';

  const symbol = symbolParam.toUpperCase().trim();
  if (!symbol) {
    return new Response('Symbol required', { status: 400 });
  }

  const mode = allowedModes.has(modeParam) ? (modeParam as 'scalp' | 'swing' | 'leaps') : 'scalp';

  const abortSignal = request.signal;
  let cleanup: (() => void) | null = null;

  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      const send = (payload: any) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
      };

      const snapshot = getLifecycleSnapshot(mode, symbol);
      if (snapshot.stage2 && snapshot.stage3) {
        send({
          mode,
          symbol,
          trades: snapshot.trades,
          stage2: snapshot.stage2,
          stage3: snapshot.stage3,
          timestamp: new Date().toISOString(),
        });
      }

      const unsubscribe = broadcastParam
        ? subscribeLifecycleBroadcast(payload => {
            if (payload.symbol === symbol && payload.mode === mode) {
              send(payload);
            }
          })
        : subscribeLifecycleUpdates(mode, symbol, payload => {
            send(payload);
          });

      const pingInterval = setInterval(() => {
        controller.enqueue(encoder.encode('event: ping\ndata: {}\n\n'));
      }, 25_000);

      controller.enqueue(encoder.encode(': connected\n\n'));
      controller.enqueue(encoder.encode('event: ready\ndata: {}\n\n'));

      cleanup = () => {
        if (closed) return;
        closed = true;
        clearInterval(pingInterval);
        unsubscribe();
        controller.close();
      };
      abortSignal.addEventListener('abort', cleanup!, { once: true });
    },
    cancel() {
      cleanup?.();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}
