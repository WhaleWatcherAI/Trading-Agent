import { handlePriceTick } from './regimeLifecycle';

type LifecycleMode = 'scalp' | 'swing' | 'leaps';

const STREAM_URL =
  process.env.TRADIER_STREAM_URL || 'wss://stream.tradier.com/v1/markets/events';
const API_KEY = process.env.TRADIER_API_KEY || '';
const ENABLE_STREAM = process.env.ENABLE_TRADIER_STREAM === 'true';
const SESSION_ID = process.env.TRADIER_STREAM_SESSION || 'regime-stream';

type SymbolModes = Map<string, Set<LifecycleMode>>;

let WebSocketImpl: any = null;
if (typeof globalThis !== 'undefined' && (globalThis as any).WebSocket) {
  WebSocketImpl = (globalThis as any).WebSocket;
} else {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require('ws');
    WebSocketImpl = mod.WebSocket || mod;
  } catch {
    WebSocketImpl = null;
  }
}

const WS_STATE = {
  CONNECTING: WebSocketImpl?.CONNECTING ?? 0,
  OPEN: WebSocketImpl?.OPEN ?? 1,
};

let ws: any = null;
let reconnectTimeout: NodeJS.Timeout | null = null;
const reconnectDelay = 5_000;
const symbolModes: SymbolModes = new Map();

const pendingAdditions = new Set<string>();
const pendingRemovals = new Set<string>();

const isServer = typeof window === 'undefined';

const log = (...args: unknown[]) => {
  if (process.env.NODE_ENV !== 'production') {
    console.log('[TradierStream]', ...args);
  }
};

function shouldConnect(): boolean {
  return (
  isServer &&
  ENABLE_STREAM &&
  Boolean(API_KEY) &&
  symbolModes.size > 0 &&
  WebSocketImpl
  );
}

function sendMessage(payload: Record<string, unknown>): void {
  if (!ws || ws.readyState !== WS_STATE.OPEN) return;
  ws.send(JSON.stringify(payload));
}

function subscribePayload(symbols: string[]): Record<string, unknown> {
  return {
    symbols: symbols.join(','),
    sessionid: SESSION_ID,
    filter: 'quotes',
  };
}

function sendSubscriptionSnapshot(): void {
  if (!ws || ws.readyState !== WS_STATE.OPEN) return;
  const symbols = Array.from(symbolModes.keys());
  if (symbols.length === 0) return;
  sendMessage(subscribePayload(symbols));
}

function sendIncrementalSubscription(symbol: string, action: '+' | '-'): void {
  if (!ws || ws.readyState !== WS_STATE.OPEN) return;
  sendMessage({
    symbols: `${action}${symbol}`,
    sessionid: SESSION_ID,
    filter: 'quotes',
  });
}

function scheduleReconnect(): void {
  if (reconnectTimeout || !shouldConnect()) return;
  reconnectTimeout = setTimeout(() => {
    reconnectTimeout = null;
    connect();
  }, reconnectDelay);
}

function handleQuoteEvent(event: any): void {
  if (!event || !event.symbol) return;
  const symbol = String(event.symbol).toUpperCase();
  const modes = symbolModes.get(symbol);
  if (!modes || modes.size === 0) return;

  const priceFields = [
    event.last,
    event.trade_price,
    event.price,
    event.close,
    event.bid,
    event.ask,
  ];
  const price = priceFields.find(
    value => typeof value === 'number' && Number.isFinite(value),
  );
  if (typeof price !== 'number') return;

  modes.forEach(mode => {
    try {
      handlePriceTick(mode, symbol, price);
    } catch (error) {
      console.error('[TradierStream] Failed to handle price tick', error);
    }
  });
}

function handleMessage(data: any): void {
  const text = data.toString();
  const lines = text.split(/\r?\n/).filter(Boolean);

  lines.forEach((line: string) => {
    try {
      const event = JSON.parse(line);
      if (event.type === 'quote') {
        handleQuoteEvent(event);
      }
    } catch (error) {
      log('Failed to parse stream event', line, error);
    }
  });
}

function connect(): void {
  if (!shouldConnect()) {
    return;
  }

  if (!WebSocketImpl) {
    log('WebSocket implementation unavailable; streaming disabled');
    return;
  }

  if (ws && (ws.readyState === WS_STATE.OPEN || ws.readyState === WS_STATE.CONNECTING)) {
    return;
  }

  log('Connecting to Tradier stream...');
  ws = new WebSocketImpl(STREAM_URL, {
    headers: {
      Authorization: `Bearer ${API_KEY}`,
    },
    handshakeTimeout: 15_000,
  });

  ws.on('open', () => {
    log('Tradier stream connected');
    sendSubscriptionSnapshot();

    if (pendingAdditions.size > 0) {
      pendingAdditions.forEach(symbol => sendIncrementalSubscription(symbol, '+'));
      pendingAdditions.clear();
    }
    pendingRemovals.clear();
  });

  ws.on('message', handleMessage);

  ws.on('close', (code: number, reason: { toString(): string }) => {
    const reasonText = typeof reason === 'string' ? reason : reason?.toString?.() ?? '';
    log('Tradier stream closed', code, reasonText);
    ws = null;
    scheduleReconnect();
  });

  ws.on('error', (error: Error) => {
    console.error('[TradierStream] socket error', error);
    try {
      ws?.close();
    } catch {
      // noop
    }
    ws = null;
    scheduleReconnect();
  });
}

function disconnectIfIdle(): void {
  if (symbolModes.size > 0) return;
  if (ws) {
    try {
      ws.close();
    } catch {
      // noop
    }
    ws = null;
  }
}

export function subscribeSymbolForMode(symbol: string, mode: LifecycleMode): void {
  if (!ENABLE_STREAM || !isServer || !WebSocketImpl) return;
  const normalized = symbol.toUpperCase();
  let modes = symbolModes.get(normalized);
  if (!modes) {
    modes = new Set<LifecycleMode>();
    symbolModes.set(normalized, modes);
  }
  modes.add(mode);

  if (ws && ws.readyState === WS_STATE.OPEN) {
    sendIncrementalSubscription(normalized, '+');
  } else {
    pendingAdditions.add(normalized);
  }

  connect();
}

export function unsubscribeSymbolForMode(symbol: string, mode: LifecycleMode): void {
  if (!ENABLE_STREAM || !isServer || !WebSocketImpl) return;
  const normalized = symbol.toUpperCase();
  const modes = symbolModes.get(normalized);
  if (!modes) return;

  modes.delete(mode);
  if (modes.size > 0) {
    return;
  }

  symbolModes.delete(normalized);
  pendingAdditions.delete(normalized);
  if (ws && ws.readyState === WS_STATE.OPEN) {
    sendIncrementalSubscription(normalized, '-');
  } else {
    pendingRemovals.add(normalized);
  }

  disconnectIfIdle();
}

export function getStreamStatus() {
  return {
    enabled: ENABLE_STREAM && isServer,
    connected: Boolean(ws && ws.readyState === WS_STATE.OPEN),
    subscribedSymbols: Array.from(symbolModes.keys()),
  };
}
