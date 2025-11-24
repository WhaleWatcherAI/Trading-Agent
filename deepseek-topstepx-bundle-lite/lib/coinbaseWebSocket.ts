import WebSocket from 'ws';
import { EventEmitter } from 'events';

/**
 * Coinbase Public WebSocket Client
 * Uses Coinbase Exchange WebSocket Feed (no authentication required)
 * Docs: https://docs.cloud.coinbase.com/exchange/docs/websocket-overview
 */

export interface CoinbaseBar {
  t: string; // ISO timestamp
  o: number; // open
  h: number; // high
  l: number; // low
  c: number; // close
  v: number; // volume
  partial?: boolean;
}

export interface CoinbaseTick {
  price: number;
  time: string;
  size: number;
  side: 'buy' | 'sell';
}

interface CoinbaseTickerMessage {
  type: 'ticker';
  sequence: number;
  product_id: string;
  price: string;
  open_24h: string;
  volume_24h: string;
  low_24h: string;
  high_24h: string;
  volume_30d: string;
  best_bid: string;
  best_ask: string;
  side: 'buy' | 'sell';
  time: string;
  trade_id: number;
  last_size: string;
}

interface CoinbaseMatchMessage {
  type: 'match';
  trade_id: number;
  maker_order_id: string;
  taker_order_id: string;
  side: 'buy' | 'sell';
  size: string;
  price: string;
  product_id: string;
  sequence: number;
  time: string;
}

interface CoinbaseSubscribeMessage {
  type: 'subscribe';
  product_ids: string[];
  channels: string[];
}

export interface CoinbaseWebSocketConfig {
  symbols: string[]; // e.g., ['SOL-USD', 'BTC-USD']
  wsUrl?: string; // Optional custom WebSocket URL
}

export class CoinbaseWebSocketClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private symbols: string[];
  private wsUrl: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 5000;
  private isShuttingDown = false;
  private pingInterval: NodeJS.Timeout | null = null;
  private isConnected = false;

  // Store latest data
  private latestPrices = new Map<string, number>();
  private bars = new Map<string, Map<string, CoinbaseBar[]>>(); // symbol -> interval -> bars[]

  constructor(config: CoinbaseWebSocketConfig) {
    super();
    this.symbols = config.symbols;
    this.wsUrl = config.wsUrl || 'wss://ws-feed.exchange.coinbase.com';
  }

  /**
   * Connect to Coinbase WebSocket and subscribe to ticker channel
   */
  connect(): void {
    if (this.ws) {
      console.log('[Coinbase WS] Already connected');
      return;
    }

    console.log(`[Coinbase WS] Connecting to ${this.wsUrl}...`);
    this.ws = new WebSocket(this.wsUrl);

    this.ws.on('open', () => {
      console.log('[Coinbase WS] Connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.subscribe();
      this.startPingInterval();
    });

    this.ws.on('message', (data: WebSocket.Data) => {
      try {
        const message = JSON.parse(data.toString());
        this.handleMessage(message);
      } catch (err) {
        console.error('[Coinbase WS] Failed to parse message:', err);
      }
    });

    this.ws.on('error', (error: Error) => {
      console.error('[Coinbase WS] Error:', error.message);
    });

    this.ws.on('close', () => {
      console.log('[Coinbase WS] Connection closed');
      this.isConnected = false;
      this.stopPingInterval();

      if (!this.isShuttingDown) {
        this.attemptReconnect();
      }
    });
  }

  /**
   * Subscribe to ticker and matches channels for all symbols
   */
  private subscribe(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('[Coinbase WS] Cannot subscribe - connection not open');
      return;
    }

    const subscribeMessage: CoinbaseSubscribeMessage = {
      type: 'subscribe',
      product_ids: this.symbols,
      channels: ['ticker', 'matches'],
    };

    console.log(`[Coinbase WS] Subscribing to: ${this.symbols.join(', ')}`);
    this.ws.send(JSON.stringify(subscribeMessage));
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(message: any): void {
    if (message.type === 'subscriptions') {
      console.log('[Coinbase WS] Subscribed to channels:', JSON.stringify(message.channels));
      return;
    }

    if (message.type === 'ticker') {
      this.handleTicker(message as CoinbaseTickerMessage);
    } else if (message.type === 'match') {
      this.handleMatch(message as CoinbaseMatchMessage);
    }
  }

  /**
   * Handle ticker message (best bid/ask, 24h stats)
   */
  private handleTicker(ticker: CoinbaseTickerMessage): void {
    const symbol = ticker.product_id;
    const price = parseFloat(ticker.price);

    if (!Number.isFinite(price)) {
      return;
    }

    this.latestPrices.set(symbol, price);

    // Emit tick event
    const tick: CoinbaseTick = {
      price,
      time: ticker.time,
      size: parseFloat(ticker.last_size || '0'),
      side: ticker.side,
    };

    this.emit('tick', symbol, tick);
  }

  /**
   * Handle match message (actual trade execution)
   */
  private handleMatch(match: CoinbaseMatchMessage): void {
    const symbol = match.product_id;
    const price = parseFloat(match.price);
    const size = parseFloat(match.size);

    if (!Number.isFinite(price)) {
      return;
    }

    this.latestPrices.set(symbol, price);

    const tick: CoinbaseTick = {
      price,
      time: match.time,
      size,
      side: match.side,
    };

    this.emit('tick', symbol, tick);
    this.emit('trade', symbol, tick);
  }

  /**
   * Start ping interval to keep connection alive
   */
  private startPingInterval(): void {
    this.stopPingInterval();
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.ping();
      }
    }, 30000); // Ping every 30 seconds
  }

  /**
   * Stop ping interval
   */
  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Attempt to reconnect after connection loss
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[Coinbase WS] Max reconnect attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * this.reconnectAttempts;
    console.log(`[Coinbase WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.ws = null;
      this.connect();
    }, delay);
  }

  /**
   * Get latest price for a symbol
   */
  getLatestPrice(symbol: string): number | null {
    return this.latestPrices.get(symbol) ?? null;
  }

  /**
   * Fetch historical candles from Coinbase REST API
   */
  async fetchCandles(symbol: string, granularity: number, limit: number = 300): Promise<CoinbaseBar[]> {
    const supportedGranularities = new Set([60, 300, 900, 3600, 21600, 86400]);
    let fetchGranularity = granularity;
    let aggregateFactor = 1;

    if (!supportedGranularities.has(granularity)) {
      if (granularity === 1800) {
        fetchGranularity = 900; // Fetch 15m bars and aggregate pairs
        aggregateFactor = 2;
      } else {
        throw new Error(`Unsupported Coinbase granularity: ${granularity}`);
      }
    }

    const url = `https://api.exchange.coinbase.com/products/${symbol}/candles?granularity=${fetchGranularity}`;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Coinbase API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Coinbase returns: [timestamp, low, high, open, close, volume]
      let bars: CoinbaseBar[] = data.map((candle: any[]) => ({
        t: new Date(candle[0] * 1000).toISOString(),
        o: candle[3],
        h: candle[2],
        l: candle[1],
        c: candle[4],
        v: candle[5],
        partial: false,
      }));

      // Sort by timestamp ascending
      bars.sort((a, b) => new Date(a.t).getTime() - new Date(b.t).getTime());

      if (aggregateFactor > 1) {
        const aggregated: CoinbaseBar[] = [];
        for (let i = 0; i + aggregateFactor - 1 < bars.length; i += aggregateFactor) {
          const slice = bars.slice(i, i + aggregateFactor);
          if (slice.length < aggregateFactor) {
            continue;
          }

          aggregated.push({
            t: slice[0].t,
            o: slice[0].o,
            h: Math.max(...slice.map(bar => bar.h)),
            l: Math.min(...slice.map(bar => bar.l)),
            c: slice[slice.length - 1].c,
            v: slice.reduce((sum, bar) => sum + bar.v, 0),
            partial: false,
          });
        }
        bars = aggregated;
      }

      return bars.slice(-limit);
    } catch (error) {
      console.error(`[Coinbase REST] Failed to fetch candles for ${symbol}:`, error);
      return [];
    }
  }

  /**
   * Get recent bars for a symbol and interval
   * Fetches from REST API and caches
   */
  async getRecentBars(symbol: string, interval: '1Min' | '5Min' | '15Min' | '30Min' | '1Hour', count: number): Promise<CoinbaseBar[]> {
    const granularityMap = {
      '1Min': 60,
      '5Min': 300,
      '15Min': 900,
      '30Min': 1800,
      '1Hour': 3600,
    };

    const granularity = granularityMap[interval];
    const cacheKey = `${symbol}-${interval}`;

    // Check cache
    if (!this.bars.has(symbol)) {
      this.bars.set(symbol, new Map());
    }

    const symbolBars = this.bars.get(symbol)!;
    let cached = symbolBars.get(interval) || [];

    // Fetch if cache empty or stale (older than 1 minute)
    const now = Date.now();
    const lastBar = cached[cached.length - 1];
    const lastBarTime = lastBar ? new Date(lastBar.t).getTime() : 0;
    const isStale = now - lastBarTime > 60000;

    if (cached.length === 0 || isStale) {
      cached = await this.fetchCandles(symbol, granularity, count);
      symbolBars.set(interval, cached);
    }

    return cached.slice(-count);
  }

  /**
   * Close WebSocket connection
   */
  disconnect(): void {
    console.log('[Coinbase WS] Disconnecting...');
    this.isShuttingDown = true;
    this.stopPingInterval();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isConnected = false;
  }

  /**
   * Check if connected
   */
  connected(): boolean {
    return this.isConnected;
  }
}
