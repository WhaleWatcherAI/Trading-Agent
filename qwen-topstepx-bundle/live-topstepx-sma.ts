import 'dotenv/config';
import fetch from 'node-fetch';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import { RSI } from 'technicalindicators';
import { createProjectXRest } from './projectx-rest';
import { PositionTracker } from './fills/position-tracker';

const DRY_RUN = process.env.DRY_RUN === '1';
const MAX_DD = Number(process.env.MAX_DAILY_DRAWDOWN ?? '0');

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
}

class ProjectXRest {
  constructor(private cfg: { baseUrl: string; getJwt: () => Promise<string> }) {}

  private async request(path: string, init: RequestInit = {}, retry = true) {
    const token = await this.cfg.getJwt();
    const res = await fetch(`${this.cfg.baseUrl}${path}`, {
      ...init,
      headers: {
        'Authorization': `Bearer ${token}`,
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

  placeMarketOrder(params: { accountId: number; contractId: string; side: 'Buy' | 'Sell'; quantity: number; }) {
    return this.request('/api/Order/place', {
      method: 'POST',
      body: JSON.stringify({
        accountId: params.accountId,
        contractId: params.contractId,
        side: params.side,
        quantity: params.quantity,
        type: 'Market',
        timeInForce: 'DAY',
      }),
    });
  }

  cancelOrder(accountId: number, orderId: number) {
    return this.request('/api/Order/cancel', {
      method: 'POST',
      body: JSON.stringify({ accountId, orderId }),
    });
  }
}

interface GatewayQuote {
  symbol: string;
  lastPrice: number;
  bestBid?: number;
  bestAsk?: number;
  timestamp: string;
}

interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  start: Date;
}

async function apiRequest(path: string, options: RequestInit, apiKey: string, baseUrl: string) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Request failed ${response.status}: ${text}`);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

interface PositionState {
  direction: 'long' | 'short';
  entryPrice: number;
  entryTime: Date;
  target: number;
  stop: number;
}

enum OrderSide {
  Bid = 0,
  Ask = 1,
}

enum OrderStatus {
  Open = 1,
  Filled = 2,
  Cancelled = 3,
  Expired = 4,
  Rejected = 5,
  Pending = 6,
}

enum OrderType {
  Limit = 1,
  Market = 2,
}

class CandleBuilder {
  private currentMinuteCandle: Candle | null = null;
  private currentFifteenCandle: Candle | null = null;
  private minuteCloses: number[] = [];
  private fifteenCloses: number[] = [];
  private readonly listeners: Array<(candle: Candle, timeframe: '1m' | '15m') => void> = [];

  onCandle(listener: (candle: Candle, timeframe: '1m' | '15m') => void) {
    this.listeners.push(listener);
  }

  feed(price: number, timestamp: Date) {
    this.handleMinute(price, timestamp);
  }

  getFifteenCloses() {
    return this.fifteenCloses;
  }

  getMinuteCloses() {
    return this.minuteCloses;
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
        start: new Date(timestamp.getFullYear(), timestamp.getMonth(), timestamp.getDate(), timestamp.getHours(), timestamp.getMinutes()),
      };
      return;
    }

    const candle = this.currentMinuteCandle;
    candle.high = Math.max(candle.high, price);
    candle.low = Math.min(candle.low, price);
    candle.close = price;

    if (timestamp.getMinutes() !== candle.start.getMinutes()) {
      this.minuteCloses.push(candle.close);
      this.emit(candle, '1m');
      this.handleFifteen(candle);
      this.currentMinuteCandle = null;
      this.handleMinute(price, timestamp);
    }
  }

  private handleFifteen(minuteCandle: Candle) {
    if (!this.currentFifteenCandle) {
      this.currentFifteenCandle = { ...minuteCandle };
      return;
    }

    const fifteen = this.currentFifteenCandle;
    fifteen.high = Math.max(fifteen.high, minuteCandle.high);
    fifteen.low = Math.min(fifteen.low, minuteCandle.low);
    fifteen.close = minuteCandle.close;

    const minutesDiff = (minuteCandle.start.getTime() - fifteen.start.getTime()) / 60000;
    if (minutesDiff >= 15) {
      this.fifteenCloses.push(fifteen.close);
      this.emit(fifteen, '15m');
      this.currentFifteenCandle = null;
      this.handleFifteen(minuteCandle);
    }
  }
}

class LiveSmaStrategy {
  private config: StrategyConfig;
  private marketHub: HubConnection;
  private userHub: HubConnection;
  private candles = new CandleBuilder();
  private position: PositionState | null = null;
  private currentSma: number | null = null;
  private currentRsi: number | null = null;
  private prevRsi: number | null = null;
  private rest;
  private tracker: PositionTracker;

  constructor(config: StrategyConfig) {
    this.config = config;
    this.rest = createProjectXRest(config.restBaseUrl);
    this.tracker = new PositionTracker(config.contractMultiplier, config.commissionPerSide);
    this.rest = new ProjectXRest({
      baseUrl: config.restBaseUrl,
      getJwt: async () => this.config.apiKey,
    });

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
  private resolvePrice(quote: GatewayQuote) {
    if (typeof quote.lastPrice === 'number' && quote.lastPrice > 0) {
      return quote.lastPrice;
    }
    if (quote.bestBid && quote.bestAsk) {
      return (quote.bestBid + quote.bestAsk) / 2;
    }
    if (quote.bestBid) return quote.bestBid;
    if (quote.bestAsk) return quote.bestAsk;
    throw new Error('Quote missing usable price');
  }

  async start() {
    this.marketHub.on('GatewayQuote', (_contractId: string, quote: GatewayQuote) => {
      const price = this.resolvePrice(quote);
      const timestamp = new Date(quote.timestamp || quote.lastUpdated || Date.now());
      this.candles.feed(price, timestamp);
    });

    await this.marketHub.start();
    console.log('Connected to market hub');

    const subscribeMarket = () => {
      this.marketHub.invoke('SubscribeContractQuotes', this.config.contractId);
      this.marketHub.invoke('SubscribeContractTrades', this.config.contractId);
      this.marketHub.invoke('SubscribeContractMarketDepth', this.config.contractId);
    };

    subscribeMarket();
    this.marketHub.onreconnected(subscribeMarket);

    await this.userHub.start();
    console.log('Connected to user hub');

    const subscribeUser = () => {
      this.userHub.invoke('SubscribeAccounts');
      if (process.env.TOPSTEPX_ACCOUNT_ID) {
        const accountId = Number(process.env.TOPSTEPX_ACCOUNT_ID);
        this.userHub.invoke('SubscribeOrders', accountId);
        this.userHub.invoke('SubscribePositions', accountId);
        this.userHub.invoke('SubscribeTrades', accountId);
      }
    };

    this.userHub.on('GatewayUserAccount', data => console.log('[Account]', data));
    this.userHub.on('GatewayUserOrder', data => console.log('[Order]', data));
    this.userHub.on('GatewayUserPosition', data => console.log('[Position]', data));
    this.userHub.on('GatewayUserTrade', (_cid: string, ev: any) => {
      console.log('[Trade]', ev);
      const side = ev.side === 0 ? 'Buy' : 'Sell';
      const qty = ev.size ?? ev.quantity;
      const price = ev.price;
      if (qty && price != null) {
        this.tracker.onFill(side, qty, price);
        this.riskCheck();
      }
    });

    subscribeUser();
    this.userHub.onreconnected(subscribeUser);
  }

  private riskCheck() {
    if (MAX_DD > 0 && -this.tracker.realized >= MAX_DD) {
      console.error(`[risk] Max DD hit (${this.tracker.realized}). Exiting.`);
      process.exit(2);
    }
  }

  private recomputeIndicators() {
    const closes = this.candles.getFifteenCloses();
    if (closes.length < this.config.smaPeriod + 2) {
      return;
    }

    const slice = closes.slice(-this.config.smaPeriod);
    const sma = slice.reduce((sum, v) => sum + v, 0) / this.config.smaPeriod;
    const rsiSeries = RSI.calculate({ values: closes, period: 14 });
    this.prevRsi = rsiSeries[rsiSeries.length - 2] ?? null;
    this.currentRsi = rsiSeries[rsiSeries.length - 1] ?? null;
    this.currentSma = sma;
  }

  private evaluateSignals(fifteenCandle: Candle) {
    if (!this.currentSma || this.currentRsi == null || this.prevRsi == null) {
      return;
    }

    const closes = this.candles.getFifteenCloses();
    const prevClose = closes[closes.length - 2];
    const currClose = closes[closes.length - 1];

    const crossedUp = prevClose <= this.currentSma && currClose > this.currentSma;
    const crossedDown = prevClose >= this.currentSma && currClose < this.currentSma;
    const rsiBullish = this.currentRsi > 50 && this.currentRsi > this.prevRsi;
    const rsiBearish = this.currentRsi < 50 && this.currentRsi < this.prevRsi;

    if (crossedUp && rsiBullish && !this.position && this.isTradingAllowed(fifteenCandle.start)) {
      this.openPosition('long', currClose, new Date());
    } else if (crossedDown && rsiBearish && !this.position && this.isTradingAllowed(fifteenCandle.start)) {
      this.openPosition('short', currClose, new Date());
    }
  }

  private evaluateStops(minuteCandle: Candle) {
    if (!this.position) {
      return;
    }

    if (!this.isTradingAllowed(minuteCandle.start) && this.position) {
      this.closePosition(minuteCandle.close, minuteCandle.start);
      return;
    }

    const { direction, target, stop } = this.position;
    if (direction === 'long') {
      if (minuteCandle.high >= target) {
        this.closePosition(target, minuteCandle.start);
      } else if (minuteCandle.low <= stop) {
        this.closePosition(stop, minuteCandle.start);
      }
    } else {
      if (minuteCandle.low <= target) {
        this.closePosition(target, minuteCandle.start);
      } else if (minuteCandle.high >= stop) {
        this.closePosition(stop, minuteCandle.start);
      }
    }
  }

  private openPosition(direction: 'long' | 'short', price: number, timestamp: Date) {
    const target = direction === 'long'
      ? price * (1 + this.config.takeProfitPercent)
      : price * (1 - this.config.takeProfitPercent);
    const stop = direction === 'long'
      ? price * (1 - this.config.stopLossPercent)
      : price * (1 + this.config.stopLossPercent);

    this.position = { direction, entryPrice: price, entryTime: timestamp, target, stop };
    console.log(`[Entry] ${direction.toUpperCase()} ${this.config.symbol} @ ${price.toFixed(2)} (${timestamp.toISOString()})`);
    this.sendMarketOrder(direction === 'long' ? 'Buy' : 'Sell').catch(err => {
      console.error('Order placement failed', err);
    });
  }

  private closePosition(price: number, timestamp: Date) {
    if (!this.position) return;
    const { direction, entryPrice } = this.position;
    const rawPnL = (price - entryPrice) * (direction === 'long' ? 1 : -1) * this.config.contractMultiplier;
    const commission = this.config.commissionPerSide * 2;
    const pnl = rawPnL - commission;
    console.log(`[Exit] ${direction.toUpperCase()} ${this.config.symbol} @ ${price.toFixed(2)} | PnL ${pnl.toFixed(2)} (${timestamp.toISOString()})`);
    this.sendMarketOrder(direction === 'long' ? 'Sell' : 'Buy').catch(err => {
      console.error('Exit order failed', err);
    });
    this.position = null;
  }

  private async sendMarketOrder(side: 'Buy' | 'Sell') {
    if (!this.config.accountId) {
      console.warn('Account ID missing; skipping live order');
      return;
    }

    if (DRY_RUN) {
      console.log(`[DRY] order ${side} x${this.config.orderSize}`);
      return;
    }

    try {
      await this.rest.placeOrder({
        accountId: this.config.accountId,
        contractId: this.config.contractId,
        side,
        quantity: this.config.orderSize,
        type: 'Market',
        timeInForce: 'DAY',
      });
    } catch (err) {
      console.error('Order API error', err);
    }
  }

  private isTradingAllowed(timestamp: Date) {
    const ctDate = new Date(timestamp.getTime() - (6 * 60 * 60 * 1000));
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
    };

  if (!config.apiKey) {
    throw new Error('TOPSTEPX_API_KEY missing');
  }

  const strategy = new LiveSmaStrategy(config);
  await strategy.start();
}

main().catch(err => {
  console.error('Live strategy failed', err);
  process.exit(1);
});
