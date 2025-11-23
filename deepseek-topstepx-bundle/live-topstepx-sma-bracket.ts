import 'dotenv/config';
import fetch, { RequestInit } from 'node-fetch';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import { RSI } from 'technicalindicators';
import { PositionTracker } from './fills/position-tracker';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXContract,
  TopstepXContract,
  TopstepXFuturesBar,
} from './lib/topstepx';

const DRY_RUN = process.env.DRY_RUN === '1';
const MAX_DD = Number(process.env.MAX_DAILY_DRAWDOWN ?? '0');
const MINUTES_IN_MS = 60 * 1000;
const MAX_MINUTE_HISTORY = 2000;
const MAX_FIFTEEN_HISTORY = 400;
const DEFAULT_TICK_SIZE = Number(process.env.TOPSTEPX_TICK_SIZE ?? '0.25');

type OrderSide = 'Buy' | 'Sell';
type ProtectiveOrderKind = 'takeProfit' | 'stopLoss' | 'smaFollower';

interface StrategyConfig {
  contractId: string;
  symbol: string;
  marketHubUrl: string;
  userHubUrl: string;
  apiKey: string;
  restBaseUrl: string;
  smaPeriod: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  contractMultiplier: number;
  commissionPerSide: number;
  accountId?: number;
  orderSize: number;
  bootstrapMinutes: number;
  historicalRefreshMinutes: number;
  smaFollowerOffsetTicks: number;
  tickSize: number;
  limitBreachGraceMs: number;
}

interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  start: Date;
}

interface ManagedOrder {
  id: number | string;
  kind: ProtectiveOrderKind;
  side: OrderSide;
  type: 'Limit' | 'StopLimit';
  price: number;
  stopPrice?: number;
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected';
}

interface PositionState {
  direction: 'long' | 'short';
  entryPrice: number;
  entryTime: Date;
  avgEntryPrice: number;
  target: number;
  stop: number;
  size: number;
  filledQuantity: number;
  entryNotional: number;
  status: 'pendingEntry' | 'live' | 'closing';
}

class ProjectXRest {
  constructor(private cfg: { baseUrl: string; apiKey: string }) {}

  private async request(path: string, init: RequestInit = {}, retry = true) {
    const res = await fetch(`${this.cfg.baseUrl}${path}`, {
      ...init,
      headers: {
        'Authorization': `Bearer ${this.cfg.apiKey}`,
        'Content-Type': 'application/json',
        ...(init.headers || {}),
      },
    });

    if (res.status === 401 && retry) {
      return this.request(path, init, false);
    }

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`${init.method || 'GET'} ${path} failed ${res.status}: ${text}`);
    }

    if (res.status === 204) {
      return null;
    }

    return res.json();
  }

  placeOrder(payload: any) {
    return this.request('/api/Order/place', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  cancelOrder(payload: any) {
    return this.request('/api/Order/cancel', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }
}

function alignToMinute(date: Date) {
  return new Date(
    Date.UTC(
      date.getUTCFullYear(),
      date.getUTCMonth(),
      date.getUTCDate(),
      date.getUTCHours(),
      date.getUTCMinutes(),
      0,
      0,
    ),
  );
}

class CandleBuilder {
  private currentMinuteCandle: Candle | null = null;
  private currentFifteenCandle: Candle | null = null;
  private currentFifteenKey: number | null = null;
  private readonly listeners: Array<(candle: Candle, timeframe: '1m' | '15m') => void> = [];
  private minuteCloses: number[] = [];
  private fifteenCloses: number[] = [];

  onCandle(listener: (candle: Candle, timeframe: '1m' | '15m') => void) {
    this.listeners.push(listener);
  }

  getMinuteCloses() {
    return this.minuteCloses;
  }

  getFifteenCloses() {
    return this.fifteenCloses;
  }

  seedFromHistoricalBars(bars: TopstepXFuturesBar[]) {
    this.minuteCloses = [];
    this.fifteenCloses = [];
    this.currentMinuteCandle = null;
    this.currentFifteenCandle = null;
    this.currentFifteenKey = null;

    const ordered = [...bars].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
    );

    ordered.forEach(bar => {
      const start = alignToMinute(new Date(bar.timestamp));
      const candle: Candle = {
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        start,
      };
      this.recordClosedMinute(candle, false);
    });
  }

  feed(price: number, timestamp: Date) {
    this.handleMinute(price, timestamp);
  }

  private emit(candle: Candle, timeframe: '1m' | '15m') {
    this.listeners.forEach(listener => listener(candle, timeframe));
  }

  private handleMinute(price: number, timestamp: Date) {
    if (!this.currentMinuteCandle) {
      this.currentMinuteCandle = {
        open: price,
        high: price,
        low: price,
        close: price,
        start: alignToMinute(timestamp),
      };
      return;
    }

    const candle = this.currentMinuteCandle;
    candle.high = Math.max(candle.high, price);
    candle.low = Math.min(candle.low, price);
    candle.close = price;

    const minuteChanged = timestamp.getUTCMinutes() !== candle.start.getUTCMinutes()
      || timestamp.getUTCHours() !== candle.start.getUTCHours();

    if (minuteChanged) {
      this.recordClosedMinute(candle, true);
      this.currentMinuteCandle = {
        open: price,
        high: price,
        low: price,
        close: price,
        start: alignToMinute(timestamp),
      };
    }
  }

  private recordClosedMinute(candle: Candle, emitEvents: boolean) {
    this.minuteCloses.push(candle.close);
    if (this.minuteCloses.length > MAX_MINUTE_HISTORY) {
      this.minuteCloses.shift();
    }
    if (emitEvents) {
      this.emit(candle, '1m');
    }

    this.handleFifteenAggregation(candle, emitEvents);
  }

  private handleFifteenAggregation(candle: Candle, emitEvents: boolean) {
    const minuteKey = Math.floor(candle.start.getTime() / MINUTES_IN_MS);
    const intervalKey = Math.floor(minuteKey / 15);

    if (this.currentFifteenKey === null) {
      this.startNewFifteen(candle, intervalKey);
      return;
    }

    if (intervalKey !== this.currentFifteenKey) {
      this.finalizeFifteen(emitEvents);
      this.startNewFifteen(candle, intervalKey);
      return;
    }

    const agg = this.currentFifteenCandle!;
    agg.high = Math.max(agg.high, candle.high);
    agg.low = Math.min(agg.low, candle.low);
    agg.close = candle.close;
  }

  private startNewFifteen(candle: Candle, key: number) {
    this.currentFifteenKey = key;
    this.currentFifteenCandle = { ...candle };
  }

  private finalizeFifteen(emitEvents: boolean) {
    if (!this.currentFifteenCandle) {
      this.currentFifteenKey = null;
      return;
    }

    if (emitEvents) {
      this.emit(this.currentFifteenCandle, '15m');
    }

    this.fifteenCloses.push(this.currentFifteenCandle.close);
    if (this.fifteenCloses.length > MAX_FIFTEEN_HISTORY) {
      this.fifteenCloses.shift();
    }

    this.currentFifteenCandle = null;
    this.currentFifteenKey = null;
  }
}

class LiveSmaBracketStrategy {
  private marketHub: HubConnection;
  private userHub: HubConnection;
  private readonly candles = new CandleBuilder();
  private readonly tracker: PositionTracker;
  private readonly rest: ProjectXRest;
  private position: PositionState | null = null;
  private protectiveOrders: Record<ProtectiveOrderKind, ManagedOrder | null> = {
    takeProfit: null,
    stopLoss: null,
    smaFollower: null,
  };
  private currentSma: number | null = null;
  private currentRsi: number | null = null;
  private prevRsi: number | null = null;
  private bootstrapPromise: Promise<void> | null = null;
  private historicalRefreshTimer: NodeJS.Timeout | null = null;
  private contractMeta: TopstepXContract | null = null;
  private tickSize: number = DEFAULT_TICK_SIZE;

  constructor(private readonly config: StrategyConfig) {
    this.rest = new ProjectXRest({
      baseUrl: config.restBaseUrl,
      apiKey: config.apiKey,
    });
    this.tracker = new PositionTracker(config.contractMultiplier, config.commissionPerSide);

    this.marketHub = new HubConnectionBuilder()
      .withUrl(`${config.marketHubUrl}?access_token=${encodeURIComponent(config.apiKey)}`, {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: () => config.apiKey,
      })
      .withAutomaticReconnect()
      .configureLogging(LogLevel.Information)
      .build();

    this.userHub = new HubConnectionBuilder()
      .withUrl(`${config.userHubUrl}?access_token=${encodeURIComponent(config.apiKey)}`, {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: () => config.apiKey,
      })
      .withAutomaticReconnect()
      .configureLogging(LogLevel.Information)
      .build();

    this.candles.onCandle((candle, timeframe) => {
      if (timeframe === '1m') {
        this.evaluateStops(candle);
      } else {
        this.recomputeIndicators();
        this.evaluateSignals(candle);
      }
    });
  }

  async start() {
    await this.loadContractMetadata();
    await this.bootstrapHistorical('startup');

    this.marketHub.on('GatewayQuote', (_contractId: string, quote: any) => {
      const price = this.resolvePrice(quote);
      const timestamp = new Date(quote.timestamp || quote.lastUpdated || Date.now());
      this.candles.feed(price, timestamp);
    });

    await this.marketHub.start();
    console.log('[live] Connected to market hub');

    const subscribeMarket = () => {
      this.marketHub.invoke('SubscribeContractQuotes', this.config.contractId);
      this.marketHub.invoke('SubscribeContractTrades', this.config.contractId);
      this.marketHub.invoke('SubscribeContractMarketDepth', this.config.contractId);
    };
    subscribeMarket();
    this.marketHub.onreconnected(subscribeMarket);

    await this.userHub.start();
    console.log('[live] Connected to user hub');

    const subscribeUser = () => {
      this.userHub.invoke('SubscribeAccounts');
      if (this.config.accountId) {
        this.userHub.invoke('SubscribeOrders', this.config.accountId);
        this.userHub.invoke('SubscribePositions', this.config.accountId);
        this.userHub.invoke('SubscribeTrades', this.config.accountId);
      }
    };
    subscribeUser();
    this.userHub.onreconnected(subscribeUser);

    this.userHub.on('GatewayUserTrade', (_cid: string, ev: any) => {
      const side: OrderSide = ev.side === 0 ? 'Buy' : 'Sell';
      const qty = Math.abs(ev.size ?? ev.quantity ?? ev.qty ?? 0);
      const price = Number(ev.price ?? ev.avgPrice ?? 0);
      if (!qty || !price) {
        return;
      }
      this.tracker.onFill(side, qty, price);
      this.handleTradeFill(side, qty, price, ev);
      this.riskCheck();
    });

    this.userHub.on('GatewayUserOrder', data => {
      console.log('[Order]', data);
    });

    this.startHistoricalRefreshLoop();
  }

  private async loadContractMetadata() {
    try {
      this.contractMeta = await fetchTopstepXContract(this.config.contractId);
      if (this.contractMeta?.tickSize) {
        this.tickSize = this.contractMeta.tickSize;
      } else {
        this.tickSize = this.config.tickSize || DEFAULT_TICK_SIZE;
      }
      console.log(`[meta] Contract ${this.config.contractId} tick ${this.tickSize}`);
    } catch (error: any) {
      console.warn('[meta] Failed to load contract metadata, using defaults:', error.message);
      this.tickSize = this.config.tickSize || DEFAULT_TICK_SIZE;
    }
  }

  private async bootstrapHistorical(reason: string) {
    if (this.bootstrapPromise) {
      return this.bootstrapPromise;
    }

    this.bootstrapPromise = (async () => {
      const end = new Date();
      const start = new Date(end.getTime() - this.config.bootstrapMinutes * MINUTES_IN_MS);
      console.log(`[bootstrap] Fetching minute bars (${reason}) ${start.toISOString()} -> ${end.toISOString()}`);
      const bars = await fetchTopstepXFuturesBars({
        contractId: this.config.contractId,
        startTime: start.toISOString(),
        endTime: end.toISOString(),
        unit: 2,
        unitNumber: 1,
        limit: 20000,
        live: false,
      });

      if (!bars.length) {
        console.warn('[bootstrap] No historical data loaded');
        return;
      }

      bars.reverse();
      this.candles.seedFromHistoricalBars(bars);
      this.recomputeIndicators();
      console.log(`[bootstrap] Seeded ${bars.length} minute bars`);
    })()
      .catch(err => {
        console.error('[bootstrap] Failed:', err);
      })
      .finally(() => {
        this.bootstrapPromise = null;
      });

    return this.bootstrapPromise;
  }

  private startHistoricalRefreshLoop() {
    if (this.config.historicalRefreshMinutes <= 0) {
      return;
    }
    const intervalMs = this.config.historicalRefreshMinutes * MINUTES_IN_MS;
    this.historicalRefreshTimer = setInterval(() => {
      this.bootstrapHistorical('refresh').catch(err => console.error('[bootstrap] Refresh error', err));
    }, intervalMs);
    this.historicalRefreshTimer.unref?.();
  }

  private resolvePrice(quote: any) {
    if (typeof quote.lastPrice === 'number' && quote.lastPrice > 0) {
      return quote.lastPrice;
    }
    if (typeof quote.bestBid === 'number' && typeof quote.bestAsk === 'number') {
      return (quote.bestBid + quote.bestAsk) / 2;
    }
    if (typeof quote.bestBid === 'number') return quote.bestBid;
    if (typeof quote.bestAsk === 'number') return quote.bestAsk;
    throw new Error('Quote missing usable price');
  }

  private recomputeIndicators() {
    const closes = this.candles.getFifteenCloses();
    if (closes.length < Math.max(this.config.smaPeriod, 15)) {
      return;
    }

    const slice = closes.slice(-this.config.smaPeriod);
    const sma = slice.reduce((sum, val) => sum + val, 0) / slice.length;
    const rsiSeries = RSI.calculate({ values: closes, period: 14 });

    if (rsiSeries.length >= 2) {
      this.prevRsi = rsiSeries[rsiSeries.length - 2];
      this.currentRsi = rsiSeries[rsiSeries.length - 1];
    } else {
      this.prevRsi = this.currentRsi;
      this.currentRsi = rsiSeries[rsiSeries.length - 1] ?? null;
    }

    this.currentSma = sma;

    if (this.position?.status === 'live') {
      this.updateSmaFollowerOrder().catch(err => console.error('[orders] SMA follower update failed', err));
    }
  }

  private evaluateSignals(fifteenCandle: Candle) {
    if (this.position && this.position.status !== 'closing') {
      return;
    }
    if (this.currentSma == null || this.currentRsi == null || this.prevRsi == null) {
      return;
    }

    const closes = this.candles.getFifteenCloses();
    if (closes.length < 2) {
      return;
    }
    const prevClose = closes[closes.length - 2];
    const currClose = closes[closes.length - 1];
    const crossedUp = prevClose <= this.currentSma && currClose > this.currentSma;
    const crossedDown = prevClose >= this.currentSma && currClose < this.currentSma;
    const rsiBullish = this.currentRsi > 50 && this.currentRsi > this.prevRsi;
    const rsiBearish = this.currentRsi < 50 && this.currentRsi < this.prevRsi;

    if (crossedUp && rsiBullish) {
      this.openPosition('long', currClose, fifteenCandle.start);
    } else if (crossedDown && rsiBearish) {
      this.openPosition('short', currClose, fifteenCandle.start);
    }
  }

  private evaluateStops(minuteCandle: Candle) {
    if (!this.position) {
      return;
    }

    if (!this.isTradingAllowed(minuteCandle.start) && this.position.status !== 'closing') {
      console.warn('[risk] Session closed, flattening position');
      this.forceMarketExit('session_close', minuteCandle.close, minuteCandle.start).catch(err =>
        console.error('[risk] Session close exit failed', err),
      );
      return;
    }

    (Object.keys(this.protectiveOrders) as ProtectiveOrderKind[]).forEach(kind => {
      const order = this.protectiveOrders[kind];
      if (!order || order.status === 'filled' || !this.position || this.position.status !== 'live') {
        return;
      }
      const breached = this.didPriceCrossOrder(minuteCandle, order);
      if (!breached) {
        return;
      }
      const ageMs = Date.now() - minuteCandle.start.getTime();
      if (ageMs >= this.config.limitBreachGraceMs) {
        console.warn(`[risk] ${kind} (${order.price.toFixed(2)}) traded through without fill, forcing exit.`);
        this.forceMarketExit(`${kind}_breach`, minuteCandle.close, minuteCandle.start).catch(err =>
          console.error('[risk] Forced exit failed', err),
        );
      }
    });
  }

  private didPriceCrossOrder(candle: Candle, order: ManagedOrder) {
    const epsilon = this.tickSize / 2 || 0.0001;

    if (order.kind === 'takeProfit') {
      if (order.side === 'Sell') {
        return candle.high >= order.price - epsilon;
      }
      return candle.low <= order.price + epsilon;
    }

    // stopLoss and smaFollower behave like trailing stops
    if (order.side === 'Sell') {
      return candle.low <= order.price + epsilon;
    }
    return candle.high >= order.price - epsilon;
  }

  private openPosition(direction: 'long' | 'short', price: number, timestamp: Date) {
    if (this.position) {
      console.log('[entry] Position already open, skipping signal');
      return;
    }

    this.position = {
      direction,
      entryPrice: price,
      entryTime: timestamp,
      avgEntryPrice: price,
      target: price,
      stop: price,
      size: this.config.orderSize,
      filledQuantity: 0,
      entryNotional: 0,
      status: 'pendingEntry',
    };

    console.log(`[entry] ${direction.toUpperCase()} signal @ ${price.toFixed(2)}`);

    this.sendMarketOrder(direction === 'long' ? 'Buy' : 'Sell', this.config.orderSize).catch(err => {
      console.error('[entry] Failed to submit entry order', err);
    });
  }

  private async sendMarketOrder(side: OrderSide, quantity: number) {
    if (!this.config.accountId) {
      console.warn('[orders] Account ID missing; skipping live order');
      return;
    }

    if (DRY_RUN) {
      console.log(`[DRY] Market ${side} x${quantity}`);
      this.simulateFill(side, quantity);
      return;
    }

    await this.rest.placeOrder({
      accountId: this.config.accountId,
      contractId: this.config.contractId,
      side,
      quantity,
      type: 'Market',
      timeInForce: 'DAY',
    });
  }

  private simulateFill(side: OrderSide, quantity: number) {
    const price = this.currentSma ?? this.position?.entryPrice ?? 0;
    if (!price || !this.position) return;
    console.log(`[DRY] Simulated fill ${side} @ ${price.toFixed(2)}`);
    this.handleTradeFill(side, quantity, price, { simulated: true });
  }

  private async placeProtectiveOrders() {
    if (!this.position || this.position.status !== 'live') {
      return;
    }

    const direction = this.position.direction;
    const side: OrderSide = direction === 'long' ? 'Sell' : 'Buy';
    const qty = this.position.size;
    const entryPrice = this.position.avgEntryPrice;

    const takeProfitPrice = this.roundToTick(
      direction === 'long'
        ? entryPrice * (1 + this.config.takeProfitPercent)
        : entryPrice * (1 - this.config.takeProfitPercent),
    );

    const stopPrice = this.roundToTick(
      direction === 'long'
        ? entryPrice * (1 - this.config.stopLossPercent)
        : entryPrice * (1 + this.config.stopLossPercent),
    );

    await this.replaceProtectiveOrder('takeProfit', {
      kind: 'takeProfit',
      side,
      type: 'Limit',
      price: takeProfitPrice,
      quantity: qty,
    });

    await this.replaceProtectiveOrder('stopLoss', {
      kind: 'stopLoss',
      side,
      type: 'StopLimit',
      price: stopPrice,
      stopPrice,
      quantity: qty,
    });

    await this.updateSmaFollowerOrder();
  }

  private async updateSmaFollowerOrder() {
    if (!this.position || this.position.status !== 'live' || this.currentSma == null) {
      await this.replaceProtectiveOrder('smaFollower', null);
      return;
    }

    const direction = this.position.direction;
    const side: OrderSide = direction === 'long' ? 'Sell' : 'Buy';
    const offset = this.config.smaFollowerOffsetTicks * this.tickSize;
    const price = this.roundToTick(
      direction === 'long' ? this.currentSma - offset : this.currentSma + offset,
    );

    if (price <= 0) {
      return;
    }

    await this.replaceProtectiveOrder('smaFollower', {
      kind: 'smaFollower',
      side,
      type: 'StopLimit',
      price,
      stopPrice: price,
      quantity: this.position.size,
    });
  }

  private async replaceProtectiveOrder(
    kind: ProtectiveOrderKind,
    params: {
      kind: ProtectiveOrderKind;
      side: OrderSide;
      type: 'Limit' | 'StopLimit';
      price: number;
      stopPrice?: number;
      quantity: number;
    } | null,
  ) {
    const existing = this.protectiveOrders[kind];
    if (existing) {
      await this.cancelManagedOrder(existing).catch(err =>
        console.warn(`[orders] Failed to cancel ${kind}:`, err.message),
      );
      this.protectiveOrders[kind] = null;
    }

    if (!params) {
      return;
    }

    const order = await this.placeManagedOrder(params);
    this.protectiveOrders[kind] = order;
  }

  private async placeManagedOrder(params: {
    kind: ProtectiveOrderKind;
    side: OrderSide;
    type: 'Limit' | 'StopLimit';
    price: number;
    stopPrice?: number;
    quantity: number;
  }): Promise<ManagedOrder> {
    if (!this.config.accountId) {
      console.warn('[orders] Account ID missing; unable to place protective order');
      throw new Error('Account ID missing');
    }

    if (DRY_RUN) {
      const mockId = `${params.kind}-${Date.now()}`;
      console.log(`[DRY] ${params.kind} ${params.side} limit ${params.price.toFixed(2)} (#${mockId})`);
      return {
        id: mockId,
        kind: params.kind,
        side: params.side,
        type: params.type,
        price: params.price,
        stopPrice: params.stopPrice,
        status: 'open',
      };
    }

    const payload: any = {
      accountId: this.config.accountId,
      contractId: this.config.contractId,
      side: params.side,
      quantity: params.quantity,
      type: params.type,
      timeInForce: 'DAY',
      price: params.price,
    };

    if (params.type === 'StopLimit') {
      payload.stopPrice = params.stopPrice ?? params.price;
    }

    const response = await this.rest.placeOrder(payload);
    const orderId = response?.orderId ?? response?.id ?? `${params.kind}-${Date.now()}`;

    console.log(
      `[orders] Placed ${params.kind} (${params.type}) #${orderId} @ ${params.price.toFixed(2)}`,
    );

    return {
      id: orderId,
      kind: params.kind,
      side: params.side,
      type: params.type,
      price: params.price,
      stopPrice: params.stopPrice,
      status: 'open',
    };
  }

  private async cancelManagedOrder(order: ManagedOrder) {
    if (!this.config.accountId) {
      return;
    }
    if (DRY_RUN) {
      console.log(`[DRY] Cancel order #${order.id}`);
      return;
    }
    await this.rest.cancelOrder({
      accountId: this.config.accountId,
      orderId: order.id,
    });
    order.status = 'cancelled';
  }

  private async cancelAllProtectiveOrders() {
    await Promise.all(
      (Object.values(this.protectiveOrders) as (ManagedOrder | null)[])
        .filter(Boolean)
        .map(order =>
          this.cancelManagedOrder(order!).catch(err =>
            console.warn('[orders] Cancel protective order failed', err.message),
          ),
        ),
    );
    this.protectiveOrders = {
      takeProfit: null,
      stopLoss: null,
      smaFollower: null,
    };
  }

  private handleTradeFill(side: OrderSide, qty: number, price: number, raw: any) {
    if (!this.position) {
      return;
    }

    const entrySide: OrderSide = this.position.direction === 'long' ? 'Buy' : 'Sell';
    const exitSide: OrderSide = entrySide === 'Buy' ? 'Sell' : 'Buy';

    if (side === entrySide) {
      this.position.entryNotional += price * qty;
      this.position.filledQuantity += qty;
      this.position.avgEntryPrice = this.position.entryNotional / this.position.filledQuantity;
      if (this.position.filledQuantity >= this.position.size) {
        this.position.status = 'live';
        this.position.entryPrice = this.position.avgEntryPrice;
        console.log(
          `[entry] Filled ${this.position.direction.toUpperCase()} @ ${this.position.entryPrice.toFixed(2)}`,
        );
        this.placeProtectiveOrders().catch(err =>
          console.error('[orders] Failed to place protective orders', err),
        );
      }
      return;
    }

    if (side === exitSide) {
      this.position.filledQuantity -= qty;
      if (this.position.filledQuantity <= 0) {
        console.log(
          `[exit] ${this.position.direction.toUpperCase()} flat @ ${price.toFixed(2)} (${raw.reason || 'fill'})`,
        );
        this.position = null;
        this.cancelAllProtectiveOrders().catch(err =>
          console.warn('[orders] Failed to cancel leftover protective orders', err),
        );
      }
    }
  }

  private async forceMarketExit(reason: string, price: number, timestamp: Date) {
    if (!this.position || this.position.status === 'closing') {
      return;
    }
    console.warn(`[exit] Forcing market exit due to ${reason}`);
    this.position.status = 'closing';
    await this.cancelAllProtectiveOrders();
    await this.sendMarketOrder(this.position.direction === 'long' ? 'Sell' : 'Buy', this.position.size);
    console.warn(
      `[exit] Market exit requested @ ${price.toFixed(2)} (${timestamp.toISOString()}) reason=${reason}`,
    );
  }

  private riskCheck() {
    if (MAX_DD > 0 && -this.tracker.realized >= MAX_DD) {
      console.error(`[risk] Max DD hit (${this.tracker.realized}). Exiting.`);
      this.forceMarketExit('max_drawdown', this.currentSma ?? 0, new Date()).catch(err =>
        console.error('[risk] Max DD exit failed', err),
      );
    }
  }

  private roundToTick(value: number) {
    if (!this.tickSize || this.tickSize <= 0) {
      return value;
    }
    const rounded = Math.round(value / this.tickSize) * this.tickSize;
    return Number(rounded.toFixed(6));
  }

  private isTradingAllowed(timestamp: Date) {
    const ctDate = new Date(timestamp.getTime() - 6 * MINUTES_IN_MS * 60);
    const day = ctDate.getUTCDay();
    const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

    if (day === 6) return false;
    if (day === 0 && minutes < 19 * 60) return false;
    if (day === 5 && minutes >= (15 * 60 + 10)) return false;
    if (minutes >= (15 * 60 + 10) && minutes < 18 * 60) return false;
    return true;
  }
}

async function main() {
  const config: StrategyConfig = {
    contractId: process.env.TOPSTEPX_CONTRACT_ID || 'CON.F.US.MES.Z25',
    symbol: process.env.TOPSTEPX_SMA_SYMBOL || 'MESZ5',
    marketHubUrl: process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market',
    userHubUrl: process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user',
    apiKey: process.env.PROJECTX_JWT || process.env.TOPSTEPX_API_KEY || '',
    restBaseUrl: process.env.TOPSTEPX_REST_BASE || process.env.TOPSTEPX_BASE_URL || 'https://api.topstepx.com',
    smaPeriod: Number(process.env.TOPSTEPX_SMA_PERIOD || '20'),
    stopLossPercent: Number(process.env.TOPSTEPX_STOP_LOSS_PERCENT || '0.001'),
    takeProfitPercent: Number(process.env.TOPSTEPX_TAKE_PROFIT_PERCENT || '0.011'),
    contractMultiplier: Number(process.env.TOPSTEPX_CONTRACT_MULTIPLIER || '5'),
    commissionPerSide: Number(process.env.TOPSTEPX_SMA_COMMISSION || '0.37'),
    accountId: process.env.TOPSTEPX_ACCOUNT_ID ? Number(process.env.TOPSTEPX_ACCOUNT_ID) : undefined,
    orderSize: Number(process.env.TOPSTEPX_ORDER_SIZE || '1'),
    bootstrapMinutes: Number(process.env.TOPSTEPX_BOOTSTRAP_MINUTES || '600'),
    historicalRefreshMinutes: Number(process.env.TOPSTEPX_HIST_REFRESH_MINUTES || '15'),
    smaFollowerOffsetTicks: Number(process.env.TOPSTEPX_SMA_FOLLOWER_OFFSET_TICKS || '1'),
    tickSize: Number(process.env.TOPSTEPX_TICK_SIZE || DEFAULT_TICK_SIZE.toString()),
    limitBreachGraceMs: Number(process.env.TOPSTEPX_LIMIT_BREACH_GRACE_MS || '5000'),
  };

  if (!config.apiKey) {
    throw new Error('TOPSTEPX_API_KEY missing');
  }
  if (!config.accountId && !DRY_RUN) {
    throw new Error('TOPSTEPX_ACCOUNT_ID missing');
  }

  const strategy = new LiveSmaBracketStrategy(config);
  await strategy.start();
}

main().catch(err => {
  console.error('Live strategy failed', err);
  process.exit(1);
});
