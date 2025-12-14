import WebSocket from 'ws';
import EventSource from 'eventsource';

let installed = false;

export function ensureSignalRPolyfills() {
  if (installed) {
    return;
  }
  if (typeof (globalThis as any).WebSocket === 'undefined') {
    (globalThis as any).WebSocket = WebSocket;
  }
  if (typeof (globalThis as any).EventSource === 'undefined') {
    (globalThis as any).EventSource = EventSource;
  }
  installed = true;
}
