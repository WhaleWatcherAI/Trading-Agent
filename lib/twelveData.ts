import { WebSocket } from 'ws';

const MINUTE_MS = 60_000;
const FIFTEEN_MIN_MS = 15 * MINUTE_MS;
const REST_BASE_URL = 'https://api.twelvedata.com/time_series';

export interface TwelveDataOptions {
  apiKey: string;
  backupApiKey?: string;
  symbols: string[];
  url?: string;
  reconnectDelayMs?: number;
  maxMinuteBars?: number;
}

export interface RealtimePriceSnapshot {
  price: number;
  timestamp: number;
}

export interface MinuteBar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v?: number;
  partial?: boolean;
}

interface ActiveMinute {
  startMs: number;
  open: number;
  high: number;
  low: number;
  close: number;
  lastUpdateMs: number;
  volume: number;
}

interface ActiveInterval {
  startMs: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  lastUpdateMs: number;
}

interface TwelveDataMessage {
  symbol?: string;
  price?: string;
  timestamp?: string;
  status?: string;
  message?: string;
}

type PriceListener = (symbol: string, snapshot: RealtimePriceSnapshot) => void;

/**
 * Lightweight Twelve Data WebSocket price feed manager.
 * Maintains latest prices and minute-level aggregates for subscribed symbols.
 */
export class TwelveDataPriceFeed {
  private readonly url: string;
  private readonly primaryApiKey: string;
  private readonly backupApiKey: string | null;
  private apiKey: string;
  private usingBackup: boolean = false;
  private readonly symbols: string[];
  private readonly reconnectDelayMs: number;
  private readonly maxMinuteBars: number;
  private readonly maxFifteenBars: number;
  private readonly syncIntervalMs = 15 * MINUTE_MS;

  private socket: WebSocket | null = null;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;

  private readonly latestPrice = new Map<string, RealtimePriceSnapshot>();
  private readonly minuteState = new Map<string, ActiveMinute>();
  private readonly minuteHistory = new Map<string, MinuteBar[]>();
  private readonly fifteenState = new Map<string, ActiveInterval>();
  private readonly fifteenHistory = new Map<string, MinuteBar[]>();
  private readonly lastRestSync = new Map<string, number>();
  private readonly listeners = new Set<PriceListener>();
  private readonly connectionListeners = new Set<(status: string, info?: any) => void>();

  constructor(options: TwelveDataOptions) {
    this.primaryApiKey = options.apiKey;
    this.backupApiKey = options.backupApiKey ?? null;
    this.apiKey = this.primaryApiKey;
    const normalized = options.symbols.map(s => s.toUpperCase());
    const uniqueSymbols: string[] = [];
    const seen = new Set<string>();
    for (let i = 0; i < normalized.length; i += 1) {
      const sym = normalized[i];
      if (!seen.has(sym)) {
        seen.add(sym);
        uniqueSymbols.push(sym);
      }
    }
    this.symbols = uniqueSymbols;
    this.url = (options.url ?? 'wss://ws.twelvedata.com/v1/quotes/price').replace(/\/$/, '') || 'wss://ws.twelvedata.com/v1/quotes/price';
    this.reconnectDelayMs = options.reconnectDelayMs ?? 5_000;
    this.maxMinuteBars = options.maxMinuteBars ?? 200;
    this.maxFifteenBars = Math.max(50, Math.floor(this.maxMinuteBars / 3));
  }

  /**
   * Begin streaming. Safe to call multiple times; noop if already connected.
   */
  start() {
    if (this.socket || !this.apiKey || this.symbols.length === 0) {
      return;
    }

    const currentKey = this.usingBackup && this.backupApiKey ? this.backupApiKey : this.apiKey;
    const url = `${this.url}?apikey=${encodeURIComponent(currentKey)}&symbol=${encodeURIComponent(this.symbols.join(','))}`;
    this.socket = new WebSocket(url);

    this.socket.on('open', () => {
      this.clearReconnect();

      // Send proper subscription message according to Twelve Data docs
      const subscribePayload = {
        action: 'subscribe',
        params: {
          symbols: this.symbols.join(',')
        }
      };

      try {
        this.socket?.send(JSON.stringify(subscribePayload));
      } catch (err) {
        console.error('[twelve-data] Failed to send subscription:', err);
      }

      // Heartbeat ping every 30 seconds to keep connection alive.
      this.heartbeatTimer = setInterval(() => {
        try {
          this.socket?.ping?.();
        } catch {
          /* swallow */
        }
      }, 30_000);

      this.emitConnectionEvent('open');
    });

    this.socket.on('message', data => {
      this.handleMessage(data.toString());
    });

    this.socket.on('error', err => {
      console.error('[twelve-data] WebSocket error', err);
      this.emitConnectionEvent('error', err instanceof Error ? err.message : String(err));
    });

    this.socket.on('close', () => {
      this.socket = null;
      this.stopHeartbeat();
      this.scheduleReconnect();
      this.emitConnectionEvent('close');
    });
  }

  /**
   * Stop streaming and clear timers.
   */
  stop() {
    this.stopHeartbeat();
    if (this.socket) {
      try {
        this.socket.close();
      } catch {
        /* swallow */
      }
    }
    this.socket = null;
    this.clearReconnect();
  }

  onPrice(listener: PriceListener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onConnection(listener: (status: string, info?: any) => void) {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  }

  private emitConnectionEvent(status: string, info?: any) {
    this.connectionListeners.forEach(listener => {
      try {
        listener(status, info);
      } catch (err) {
        console.error('[twelve-data] connection listener error', err);
      }
    });
  }

  getSnapshot(symbol: string): RealtimePriceSnapshot | undefined {
    return this.latestPrice.get(symbol.toUpperCase());
  }

  async bootstrap(): Promise<void> {
    if (!this.apiKey || this.symbols.length === 0) {
      return;
    }
    for (const symbol of this.symbols) {
      try {
        await this.syncSymbol(symbol, true);
      } catch (err) {
        console.error(`[twelve-data] bootstrap failed for ${symbol}`, err);
      }
    }
  }

  shouldSync(symbol: string): boolean {
    const sym = symbol.toUpperCase();
    const last = this.lastRestSync.get(sym) ?? 0;
    return Date.now() - last >= this.syncIntervalMs;
  }

  async syncSymbol(symbol: string, force = false): Promise<void> {
    if (!this.apiKey) {
      return;
    }
    const sym = symbol.toUpperCase();
    if (!force && !this.shouldSync(sym)) {
      return;
    }
    try {
      const minuteBars = await this.fetchTimeSeries(sym, '1min', this.maxMinuteBars);
      this.ingestHistoricalMinuteBars(sym, minuteBars);
      this.lastRestSync.set(sym, Date.now());
    } catch (err) {
      console.error(`[twelve-data] sync failed for ${symbol}`, err);
    }
  }

  getRecentMinuteBars(symbol: string, count: number): MinuteBar[] {
    const sym = symbol.toUpperCase();
    const history = this.minuteHistory.get(sym) ?? [];
    const active = this.minuteState.get(sym);

    const combined = [...history];
    if (active) {
      combined.push(this.minuteToBar(sym, active, true));
    }
    if (count >= combined.length) {
      return combined;
    }
    return combined.slice(-count);
  }

  getRecentBars(symbol: string, interval: '1Min' | '15Min', count: number): MinuteBar[] {
    if (interval === '1Min') {
      return this.getRecentMinuteBars(symbol, count);
    }
    const sym = symbol.toUpperCase();
    const history = this.fifteenHistory.get(sym) ?? [];
    const active = this.fifteenState.get(sym);

    const combined = [...history];
    if (active) {
      combined.push(this.intervalToBar(active, true));
    }
    if (count >= combined.length) {
      return combined;
    }
    return combined.slice(-count);
  }

  private async fetchTimeSeries(symbol: string, interval: '1min', limit: number): Promise<MinuteBar[]> {
    const params = new URLSearchParams({
      symbol,
      interval,
      outputsize: String(limit),
      apikey: this.apiKey,
      order: 'ASC',
    });

    const response = await fetch(`${REST_BASE_URL}?${params.toString()}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`REST ${response.status}: ${text}`);
    }
    const data = await response.json();
    if (data?.status === 'error') {
      const errorMsg = data?.message || 'Unknown Twelve Data error';
      // Check if we ran out of credits and have a backup key
      if (errorMsg.includes('run out of API credits') && this.backupApiKey && !this.usingBackup) {
        console.log('[twelve-data] Primary API key out of credits, switching to backup key');
        this.apiKey = this.backupApiKey;
        this.usingBackup = true;
        // Retry with backup key
        const retryParams = new URLSearchParams({
          symbol,
          interval,
          outputsize: String(limit),
          apikey: this.apiKey,
          order: 'ASC',
        });
        const retryResponse = await fetch(`${REST_BASE_URL}?${retryParams.toString()}`);
        if (!retryResponse.ok) {
          const retryText = await retryResponse.text();
          throw new Error(`REST ${retryResponse.status}: ${retryText}`);
        }
        const retryData = await retryResponse.json();
        if (retryData?.status === 'error') {
          throw new Error(retryData?.message || 'Unknown Twelve Data error');
        }
        return this.parseTimeSeriesData(retryData, limit);
      }
      throw new Error(errorMsg);
    }
    return this.parseTimeSeriesData(data, limit);
  }

  private parseTimeSeriesData(data: any, limit: number): MinuteBar[] {
    const values: any[] = Array.isArray(data?.values) ? data.values : [];
    const mapped = values
      .map<MinuteBar | null>(item => {
        const ts = this.parseTimestamp(item?.datetime);
        if (!Number.isFinite(ts)) {
          return null;
        }
        const open = Number(item?.open);
        const high = Number(item?.high);
        const low = Number(item?.low);
        const close = Number(item?.close);
        const volume = item?.volume !== undefined ? Number(item.volume) : undefined;
        if (![open, high, low, close].every(v => Number.isFinite(v))) {
          return null;
        }
        const bar: MinuteBar = {
          t: new Date(ts).toISOString(),
          o: open,
          h: high,
          l: low,
          c: close,
          v: Number.isFinite(volume) ? volume : undefined,
        };
        return bar;
      })
      .filter((bar): bar is MinuteBar => bar !== null);

    mapped.sort((a, b) => new Date(a.t).getTime() - new Date(b.t).getTime());
    return mapped.slice(-limit);
  }

  private ingestHistoricalMinuteBars(symbol: string, bars: MinuteBar[]) {
    const sym = symbol.toUpperCase();
    const dedup = new Map<string, MinuteBar>();
    for (const bar of bars) {
      dedup.set(bar.t, bar);
    }
    const sorted = Array.from(dedup.values()).sort(
      (a, b) => new Date(a.t).getTime() - new Date(b.t).getTime(),
    );
    const trimmed = sorted.slice(-this.maxMinuteBars);
    console.log(`[twelve-data] Ingested ${bars.length} bars for ${sym}, after dedup: ${sorted.length}, stored: ${trimmed.length}`);
    if (trimmed.length > 0) {
      console.log(`[twelve-data] ${sym} bar range: ${trimmed[0].t} to ${trimmed[trimmed.length - 1].t}`);
    }
    this.minuteHistory.set(sym, trimmed);
    this.minuteState.delete(sym);
    this.rebuildFifteenFromMinuteBars(sym);
  }

  private rebuildFifteenFromMinuteBars(symbol: string) {
    const sym = symbol.toUpperCase();
    this.fifteenHistory.set(sym, []);
    this.fifteenState.delete(sym);
    const minutes = this.minuteHistory.get(sym) ?? [];
    for (const bar of minutes) {
      this.updateFifteenFromMinuteBar(sym, bar, false);
    }
    // No open interval after replay; next realtime minute will start fresh.
    this.fifteenState.delete(sym);
  }

  private updateFifteenFromMinuteBar(symbol: string, minuteBar: MinuteBar, isPartial: boolean) {
    const sym = symbol.toUpperCase();
    const barTime = Date.parse(minuteBar.t);
    if (!Number.isFinite(barTime)) {
      return;
    }
    const intervalStart = Math.floor(barTime / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS;
    const current = this.fifteenState.get(sym);

    if (!current || current.startMs !== intervalStart) {
      if (current) {
        this.pushFifteenHistory(sym, current);
      }
      this.fifteenState.set(sym, {
        startMs: intervalStart,
        open: minuteBar.o,
        high: minuteBar.h,
        low: minuteBar.l,
        close: minuteBar.c,
        volume: minuteBar.v ?? 0,
        lastUpdateMs: barTime,
      });
    } else {
      current.high = Math.max(current.high, minuteBar.h);
      current.low = Math.min(current.low, minuteBar.l);
      current.close = minuteBar.c;
      current.volume += minuteBar.v ?? 0;
      current.lastUpdateMs = barTime;
    }

    if (!isPartial) {
      const state = this.fifteenState.get(sym);
      if (state && barTime >= state.startMs + FIFTEEN_MIN_MS - MINUTE_MS) {
        this.pushFifteenHistory(sym, state);
        this.fifteenState.delete(sym);
      }
    }
  }

  private pushFifteenHistory(symbol: string, interval: ActiveInterval) {
    const sym = symbol.toUpperCase();
    const history = this.fifteenHistory.get(sym) ?? [];
    history.push(this.intervalToBar(interval, false));
    while (history.length > this.maxFifteenBars) {
      history.shift();
    }
    this.fifteenHistory.set(sym, history);
  }

  private intervalToBar(interval: ActiveInterval, partial: boolean): MinuteBar {
    return {
      t: new Date(interval.startMs).toISOString(),
      o: interval.open,
      h: interval.high,
      l: interval.low,
      c: interval.close,
      v: interval.volume,
      partial,
    };
  }

  private handleMessage(raw: string) {
    let parsed: TwelveDataMessage | undefined;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return;
    }
    if (!parsed || !parsed.symbol || !parsed.price) {
      if (parsed?.status && parsed.message) {
        console.warn(`[twelve-data] ${parsed.status}: ${parsed.message}`);
      }
      return;
    }

    const symbol = parsed.symbol.toUpperCase();
    const price = Number(parsed.price);
    const timestamp = this.parseTimestamp(parsed.timestamp);

    if (!Number.isFinite(price) || !Number.isFinite(timestamp)) {
      return;
    }

    const snapshot: RealtimePriceSnapshot = { price, timestamp };
    this.latestPrice.set(symbol, snapshot);
    this.updateMinuteState(symbol, snapshot);

    this.listeners.forEach(listener => {
      try {
        listener(symbol, snapshot);
      } catch (err) {
        console.error('[twelve-data] listener error', err);
      }
    });
  }

  private updateMinuteState(symbol: string, snapshot: RealtimePriceSnapshot) {
    const minuteStart = Math.floor(snapshot.timestamp / 60_000) * 60_000;
    const current = this.minuteState.get(symbol);

    if (!current || current.startMs !== minuteStart) {
      if (current) {
        this.pushMinuteHistory(symbol, current);
      }
      const fresh: ActiveMinute = {
        startMs: minuteStart,
        open: snapshot.price,
        high: snapshot.price,
        low: snapshot.price,
        close: snapshot.price,
        lastUpdateMs: snapshot.timestamp,
        volume: 0,
      };
      this.minuteState.set(symbol, fresh);
      return;
    }

    current.high = Math.max(current.high, snapshot.price);
    current.low = Math.min(current.low, snapshot.price);
    current.close = snapshot.price;
    current.lastUpdateMs = snapshot.timestamp;

    // Handle extended gaps by rolling the state when current minute has elapsed.
    const minuteEnd = current.startMs + 60_000;
    if (snapshot.timestamp >= minuteEnd) {
      this.pushMinuteHistory(symbol, current);
      this.minuteState.set(symbol, {
        startMs: minuteStart,
        open: snapshot.price,
        high: snapshot.price,
        low: snapshot.price,
        close: snapshot.price,
        lastUpdateMs: snapshot.timestamp,
        volume: 0,
      });
    }
  }

  private pushMinuteHistory(symbol: string, minute: ActiveMinute) {
    const history = this.minuteHistory.get(symbol) ?? [];
    const bar = this.minuteToBar(symbol, minute, false);
    if (history.length > 0 && history[history.length - 1].t === bar.t) {
      history[history.length - 1] = bar;
    } else {
      history.push(bar);
    }
    while (history.length > this.maxMinuteBars) {
      history.shift();
    }
    this.minuteHistory.set(symbol, history);
    this.updateFifteenFromMinuteBar(symbol, bar, false);
  }

  private minuteToBar(symbol: string, minute: ActiveMinute, partial: boolean): MinuteBar {
    const startIso = new Date(minute.startMs).toISOString();
    return {
      t: startIso,
      o: minute.open,
      h: minute.high,
      l: minute.low,
      c: minute.close,
      v: minute.volume,
      partial,
    };
  }

  private parseTimestamp(ts?: string | number): number {
    if (!ts) {
      return Date.now();
    }

    // Handle Unix timestamp (number) - Twelve Data WebSocket sends seconds
    if (typeof ts === 'number') {
      return ts * 1000; // Convert seconds to milliseconds
    }

    const trimmed = ts.trim();
    if (!trimmed) {
      return Date.now();
    }

    const normalized = trimmed.includes(' ') ? trimmed.replace(' ', 'T') : trimmed;
    const hasExplicitZone = /[zZ]|[+-]\d\d:\d\d$/.test(normalized);
    const candidates = hasExplicitZone ? [normalized] : [`${normalized}Z`, normalized];

    for (const candidate of candidates) {
      const parsed = Date.parse(candidate);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }

    const fallback = Date.parse(trimmed);
    if (!Number.isNaN(fallback)) {
      return fallback;
    }

    console.warn(`[twelve-data] Unable to parse timestamp '${ts}'; defaulting to current time`);
    return Date.now();
  }

  private scheduleReconnect() {
    if (this.reconnectTimer || !this.apiKey || this.symbols.length === 0) {
      return;
    }
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.start();
    }, this.reconnectDelayMs);
  }

  private clearReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }
}
