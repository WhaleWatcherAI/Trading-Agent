#!/usr/bin/env tsx

/**
 * TopstepX 1-second SMA strategy with RSI + ADX confirmation.
 *
 * Live runner features:
 * - SignalR market feed for 1s ticks (no polling).
 * - SMA cross + RSI trend + ADX strength filtering (matches backtest).
 * - Order cascade: Limit IOC, Limit IOC (+tick), Market fallback.
 * - Position sync via GatewayUserTrade/GatewayUserPosition events.
 * - Session guards + risk kill-switch.
 *
 * Required env:
 *   PROJECTX_JWT, TOPSTEPX_API_KEY, TOPSTEPX_USERNAME,
 *   TOPSTEPX_CONTRACT_ID (or TOPSTEPX_SECOND_SMA_SYMBOL) and TOPSTEPX_ACCOUNT_ID.
 */

import 'dotenv/config';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import { RSI, ADX } from 'technicalindicators';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import * as path from 'path';
import { createProjectXRest } from './projectx-rest';
import { PositionTracker } from './fills/position-tracker';
import {
  authenticate,
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  TopstepXContract,
  TopstepXFuturesBar,
} from './lib/topstepx';

const DRY_RUN = process.env.DRY_RUN === '1';
const MAX_DD = Number(process.env.MAX_DAILY_DRAWDOWN ?? '0');

interface StrategyConfig {
  contractId?: string;
  symbol: string;
  accountId?: number;
  marketHubUrl: string;
  userHubUrl: string;
  restBaseUrl?: string;
  smaPeriod: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  contractMultiplier: number;
  commissionPerSide: number;
  rsiPeriod: number;
  adxPeriod: number;
  adxThreshold: number;
  bypassAdx: boolean;
  numberOfContracts: number;
  initialBackfillSeconds: number;
  orderDecisionWindowMs: number;
  retryTickOffset: number;
  allowMarketFallback: boolean;
}

interface GatewayQuote {
  symbol: string;
  lastPrice: number;
  bestBid?: number;
  bestAsk?: number;
  timestamp?: string;
  lastUpdated?: string;
}

interface SecondBar {
  open: number;
  high: number;
  low: number;
  close: number;
  start: Date;
}

interface StrategyPosition {
  direction: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  contracts: number;
  stop: number | null;
  target: number | null;
  entryRSI: number;
  entryADX: number | null;
  tradeId: string;
}

const CONFIG: StrategyConfig = {
  contractId: process.env.TOPSTEPX_SECOND_SMA_CONTRACT_ID || process.env.TOPSTEPX_CONTRACT_ID,
  symbol: process.env.TOPSTEPX_SECOND_SMA_SYMBOL || process.env.TOPSTEPX_SMA_SYMBOL || 'MESZ5',
  accountId: process.env.TOPSTEPX_ACCOUNT_ID ? Number(process.env.TOPSTEPX_ACCOUNT_ID) : undefined,
  marketHubUrl: process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market',
  userHubUrl: process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user',
  restBaseUrl: process.env.TOPSTEPX_REST_BASE || process.env.TOPSTEPX_BASE_URL,
  smaPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_PERIOD || '200'),
  stopLossPercent: Number(process.env.TOPSTEPX_SECOND_STOP_LOSS_PERCENT || '0.0009'),
  takeProfitPercent: Number(process.env.TOPSTEPX_SECOND_TAKE_PROFIT_PERCENT || '0'),
  contractMultiplier: Number(process.env.TOPSTEPX_SECOND_CONTRACT_MULTIPLIER || '50'),
  commissionPerSide: Number(process.env.TOPSTEPX_SECOND_COMMISSION || '0.37'),
  rsiPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_RSI_PERIOD || '24'),
  adxPeriod: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_PERIOD || '24'),
  adxThreshold: Number(process.env.TOPSTEPX_SECOND_SMA_ADX_THRESHOLD || '25'),
  bypassAdx: process.env.TOPSTEPX_SECOND_BYPASS_ADX === 'true',
  numberOfContracts: Number(process.env.TOPSTEPX_SECOND_CONTRACTS || '2'),
  initialBackfillSeconds: Number(process.env.TOPSTEPX_SECOND_BACKFILL || '600'),
  orderDecisionWindowMs: Number(process.env.TOPSTEPX_SECOND_IOC_WINDOW_MS || '250'),
  retryTickOffset: Number(process.env.TOPSTEPX_SECOND_RETRY_TICKS || '1'),
  allowMarketFallback: process.env.TOPSTEPX_SECOND_ALLOW_MARKET !== 'false',
};

const TRADE_LOG_FILE =
  process.env.TOPSTEPX_SECOND_SMA_TRADE_LOG || './logs/topstepx-sma-second-live.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

class SecondBarBuilder {
  private current: SecondBar | null = null;
  private listeners: Array<(bar: SecondBar) => void> = [];

  onBar(listener: (bar: SecondBar) => void) {
    this.listeners.push(listener);
  }

  feed(price: number, timestamp: Date) {
    const secondStartMs = Math.floor(timestamp.getTime() / 1000) * 1000;
    if (!this.current || this.current.start.getTime() !== secondStartMs) {
      if (this.current) {
        this.emit(this.current);
      }
      this.current = {
        open: price,
        high: price,
        low: price,
        close: price,
        start: new Date(secondStartMs),
      };
      return;
    }

    this.current.high = Math.max(this.current.high, price);
    this.current.low = Math.min(this.current.low, price);
    this.current.close = price;
  }

  flushPending() {
    if (this.current) {
      this.emit(this.current);
      this.current = null;
    }
  }

  private emit(bar: SecondBar) {
    this.listeners.forEach(listener => listener({ ...bar }));
  }
}

interface OrderExecutionOpts {
  reason: string;
  limitAnchor?: number;
}

class OrderExecutor {
  private busy = false;

  constructor(
    private deps: {
      accountId?: number;
      contractId: string;
      rest: ReturnType<typeof createProjectXRest>;
      tracker: PositionTracker;
      getQuote: () => GatewayQuote | null;
      tickSize: number;
      decisionWindowMs: number;
      retryTickOffset: number;
      allowMarketFallback: boolean;
      dryRun: boolean;
    },
  ) {}

  async moveTo(targetNet: number, opts: OrderExecutionOpts): Promise<boolean> {
    if (!this.deps.accountId) {
      console.warn('[orders] Account ID missing; cannot place live orders');
      return false;
    }
    if (this.busy) {
      console.warn('[orders] Executor busy; skipping new request');
      return false;
    }

    this.busy = true;
    try {
      const attempts: Array<{ type: 'limit' | 'market'; tickOffset: number; tif: 'IOC' | 'DAY'; label: string }> = [
        { type: 'limit', tickOffset: 0, tif: 'IOC', label: 'limit IOC' },
        { type: 'limit', tickOffset: this.deps.retryTickOffset, tif: 'IOC', label: 'limit IOC +tick' },
      ];
      if (this.deps.allowMarketFallback) {
        attempts.push({ type: 'market', tickOffset: 0, tif: 'DAY', label: 'market fallback' });
      }

      for (const attempt of attempts) {
        const signedRemaining = targetNet - this.deps.tracker.position;
        if (signedRemaining === 0) {
          return true;
        }

        const side: 'Buy' | 'Sell' = signedRemaining > 0 ? 'Buy' : 'Sell';
        const qty = Math.abs(signedRemaining);
        if (qty === 0) {
          return true;
        }

        const limitPrice =
          attempt.type === 'limit'
            ? this.computeLimitPrice(side, attempt.tickOffset, opts.limitAnchor)
            : null;

        if (attempt.type === 'limit' && (limitPrice == null || limitPrice <= 0)) {
          console.warn('[orders] No quote for limit attempt, skipping');
          continue;
        }

        await this.submitOrder(side, qty, attempt, limitPrice, opts.reason);
        const filled = await this.waitForTarget(targetNet, this.deps.decisionWindowMs);
        if (filled) {
          return true;
        }
      }

      return targetNet === this.deps.tracker.position;
    } finally {
      this.busy = false;
    }
  }

  private computeLimitPrice(side: 'Buy' | 'Sell', tickOffset: number, anchor?: number) {
    const quote = this.deps.getQuote();
    const base = anchor ?? (side === 'Buy' ? quote?.bestAsk ?? quote?.lastPrice : quote?.bestBid ?? quote?.lastPrice);
    if (!base || !Number.isFinite(base)) {
      return null;
    }
    const direction = side === 'Buy' ? 1 : -1;
    const price = base + direction * tickOffset * this.deps.tickSize;
    return this.roundToTick(price);
  }

  private roundToTick(value: number) {
    if (!this.deps.tickSize) return value;
    return Math.round(value / this.deps.tickSize) * this.deps.tickSize;
  }

  private async submitOrder(
    side: 'Buy' | 'Sell',
    quantity: number,
    attempt: { type: 'limit' | 'market'; tickOffset: number; tif: 'IOC' | 'DAY'; label: string },
    price: number | null,
    reason: string,
  ) {
    if (this.deps.dryRun) {
      const fillPrice = price ?? this.deps.getQuote()?.lastPrice ?? 0;
      console.log(`[DRY] ${attempt.label} ${side} x${quantity} @ ${price ?? 'MKT'} (${reason})`);
      this.deps.tracker.onFill(side, quantity, fillPrice);
      return;
    }

    try {
      await this.deps.rest.placeOrder({
        accountId: this.deps.accountId,
        contractId: this.deps.contractId,
        side,
        quantity,
        type: attempt.type === 'limit' ? 'Limit' : 'Market',
        timeInForce: attempt.tif,
        ...(attempt.type === 'limit' ? { price } : {}),
      });
      console.log(
        `[orders] ${attempt.label} ${side} x${quantity} ${
          attempt.type === 'limit' ? `@ ${price?.toFixed(2)}` : '@ MKT'
        } (${reason})`,
      );
    } catch (err: any) {
      console.error('[orders] place order failed:', err.message ?? err);
    }
  }

  private waitForTarget(target: number, timeoutMs: number) {
    return new Promise<boolean>(resolve => {
      const start = Date.now();
      const poll = () => {
        const current = this.deps.tracker.position;
        if (target === 0 && current === 0) {
          return resolve(true);
        }
        if (current === target) {
          return resolve(true);
        }
        if (Date.now() - start >= timeoutMs) {
          return resolve(false);
        }
        setTimeout(poll, 20);
      };
      poll();
    });
  }
}

const rsiIndicator = new RSI({ period: CONFIG.rsiPeriod, values: [] as number[] });
const adxIndicator = new ADX({ period: CONFIG.adxPeriod, high: [], low: [], close: [] });

let prevRsiValue: number | null = null;
let lastQuote: GatewayQuote | null = null;
let tickSize = Number(process.env.TOPSTEPX_TICK_SIZE || '0.25');
let multiplier = CONFIG.contractMultiplier;
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let contractMeta: TopstepXContract | null = null;
let orderExecutor: OrderExecutor | null = null;
let shuttingDown = false;
let strategyPosition: StrategyPosition | null = null;
let tradeSequence = 0;

const tracker = new PositionTracker(CONFIG.contractMultiplier, CONFIG.commissionPerSide);
const restClient = createProjectXRest(CONFIG.restBaseUrl);
const barBuilder = new SecondBarBuilder();

let closes: number[] = [];
let highs: number[] = [];
let lows: number[] = [];

barBuilder.onBar(bar => {
  processLiveBar(bar).catch(err => console.error('[bar] processing failed', err));
});

function nowIso() {
  return new Date().toISOString();
}

function log(message: string) {
  console.log(`[${nowIso()}][${CONFIG.symbol}] ${message}`);
}

function ensureTradeLogDir() {
  if (!existsSync(TRADE_LOG_DIR)) {
    mkdirSync(TRADE_LOG_DIR, { recursive: true });
  }
}

function logTradeEvent(event: Record<string, unknown>) {
  try {
    ensureTradeLogDir();
    appendFileSync(
      TRADE_LOG_FILE,
      `${JSON.stringify({ timestamp: nowIso(), ...event })}\n`,
      'utf-8',
    );
  } catch (err) {
    console.error('[trade-log] write failed', err);
  }
}

function nextTradeId() {
  tradeSequence = (tradeSequence + 1) % 1_000_000;
  return `${Date.now()}-${tradeSequence}`;
}

function toCentralTime(date: Date) {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isTradingAllowed(timestamp: Date) {
  const ctDate = toCentralTime(timestamp);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 6) return false;
  if (day === 0 && minutes < WEEKEND_REOPEN_MINUTES) return false;
  if (day === 5 && minutes >= CUT_OFF_MINUTES) return false;
  return minutes < CUT_OFF_MINUTES || minutes >= REOPEN_MINUTES;
}

function shouldFlattenForClose(timestamp: Date) {
  const ctDate = toCentralTime(timestamp);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();
  const flattenMinutes = CUT_OFF_MINUTES - 5;
  if (minutes >= flattenMinutes) {
    if (day === 5) return true;
    return true;
  }
  return false;
}

function calculateSMA(values: number[], period: number) {
  if (values.length < period) return null;
  const slice = values.slice(-period);
  const sum = slice.reduce((acc, value) => acc + value, 0);
  return sum / slice.length;
}

function computeStop(direction: 'long' | 'short', entryPrice: number) {
  return direction === 'long'
    ? entryPrice * (1 - CONFIG.stopLossPercent)
    : entryPrice * (1 + CONFIG.stopLossPercent);
}

function computeTarget(direction: 'long' | 'short', entryPrice: number) {
  if (CONFIG.takeProfitPercent <= 0) return null;
  return direction === 'long'
    ? entryPrice * (1 + CONFIG.takeProfitPercent)
    : entryPrice * (1 - CONFIG.takeProfitPercent);
}

async function processLiveBar(bar: SecondBar) {
  closes.push(bar.close);
  highs.push(bar.high);
  lows.push(bar.low);
  const maxHistory = Math.max(CONFIG.smaPeriod + 10, CONFIG.adxPeriod * 3, 600);
  if (closes.length > maxHistory) {
    closes = closes.slice(-maxHistory);
    highs = highs.slice(-maxHistory);
    lows = lows.slice(-maxHistory);
  }

  if (!isTradingAllowed(bar.start)) {
    if (tracker.position !== 0) {
      await flattenPosition('session_gate', bar.close);
    }
    return;
  }

  if (shouldFlattenForClose(bar.start) && tracker.position !== 0) {
    await flattenPosition('session_close', bar.close);
    return;
  }

  await monitorActivePosition(bar);

  const sma = calculateSMA(closes, CONFIG.smaPeriod);
  if (sma === null) {
    return;
  }

  const rsiValue = rsiIndicator.nextValue(bar.close);
  const adxValue = adxIndicator.nextValue({ high: bar.high, low: bar.low, close: bar.close });
  if (rsiValue === undefined) {
    return;
  }

  const typedAdx =
    typeof adxValue === 'number'
      ? { adx: adxValue }
      : adxValue && typeof adxValue === 'object'
      ? adxValue
      : null;
  if (!CONFIG.bypassAdx && (!typedAdx || typedAdx.adx === undefined)) {
    prevRsiValue = rsiValue;
    return;
  }

  const prevClose = closes[closes.length - 2];
  if (prevClose === undefined || prevRsiValue === null) {
    prevRsiValue = rsiValue;
    return;
  }

  const crossedUp = prevClose <= sma && bar.close > sma;
  const crossedDown = prevClose >= sma && bar.close < sma;

  const rsiGrowingLong = rsiValue > prevRsiValue;
  const rsiGrowingShort = rsiValue < prevRsiValue;

  const adxTrending = CONFIG.bypassAdx ? true : (typedAdx?.adx ?? 0) >= CONFIG.adxThreshold;
  const adxSupportsLong =
    CONFIG.bypassAdx || (adxTrending && (typedAdx?.pdi ?? 0) > (typedAdx?.mdi ?? 0));
  const adxSupportsShort =
    CONFIG.bypassAdx || (adxTrending && (typedAdx?.mdi ?? 0) > (typedAdx?.pdi ?? 0));

  const longSignal = crossedUp && rsiValue > 50 && rsiGrowingLong && adxSupportsLong;
  const shortSignal = crossedDown && rsiValue < 50 && rsiGrowingShort && adxSupportsShort;

  // Flip out if opposite signal fires.
  if (strategyPosition) {
    if (strategyPosition.direction === 'long' && shortSignal) {
      await flattenPosition('signal_flip', bar.close);
    } else if (strategyPosition.direction === 'short' && longSignal) {
      await flattenPosition('signal_flip', bar.close);
    }
  }

  if (tracker.position !== 0 || !orderExecutor) {
    prevRsiValue = rsiValue;
    return;
  }

  if (longSignal) {
    log(
      `[signal] LONG @ ${bar.close.toFixed(2)} | SMA ${sma?.toFixed(2) ?? '—'} | RSI ${rsiValue.toFixed(2)} (prev ${
        prevRsiValue?.toFixed(2) ?? '—'
      }) | ADX ${typedAdx?.adx?.toFixed(1) ?? '—'} (+DI ${
        typedAdx?.pdi?.toFixed(1) ?? '—'
      } / -DI ${typedAdx?.mdi?.toFixed(1) ?? '—'})`,
    );
    await enterPosition('long', bar, rsiValue, typedAdx?.adx ?? null);
  } else if (shortSignal) {
    log(
      `[signal] SHORT @ ${bar.close.toFixed(2)} | SMA ${sma?.toFixed(2) ?? '—'} | RSI ${rsiValue.toFixed(2)} (prev ${
        prevRsiValue?.toFixed(2) ?? '—'
      }) | ADX ${typedAdx?.adx?.toFixed(1) ?? '—'} (+DI ${
        typedAdx?.pdi?.toFixed(1) ?? '—'
      } / -DI ${typedAdx?.mdi?.toFixed(1) ?? '—'})`,
    );
    await enterPosition('short', bar, rsiValue, typedAdx?.adx ?? null);
  } else if (crossedDown || crossedUp) {
    const reason = crossedDown
      ? `SHORT blocked (RSI<50=${rsiValue < 50}, RSI trend=${rsiGrowingShort}, ADX trend=${adxTrending}, mdi>pdi=${
          (typedAdx?.mdi ?? 0) > (typedAdx?.pdi ?? 0)
        })`
      : `LONG blocked (RSI>50=${rsiValue > 50}, RSI trend=${rsiGrowingLong}, ADX trend=${adxTrending}, pdi>mdi=${
          (typedAdx?.pdi ?? 0) > (typedAdx?.mdi ?? 0)
        })`;
    log(`[signal-check] ${reason} | price ${bar.close.toFixed(2)} | ADX ${
      typedAdx?.adx?.toFixed(1) ?? '—'
    }`);
  }

  prevRsiValue = rsiValue;
}

async function monitorActivePosition(bar: SecondBar) {
  if (!strategyPosition || tracker.position === 0) {
    return;
  }

  const direction = strategyPosition.direction === 'long' ? 1 : -1;
  const contracts = Math.abs(tracker.position);
  strategyPosition.contracts = contracts;

  const stopHit =
    strategyPosition.stop != null &&
    ((direction === 1 && bar.low <= strategyPosition.stop) ||
      (direction === -1 && bar.high >= strategyPosition.stop));

  if (stopHit) {
    await flattenPosition('stop', strategyPosition.stop ?? bar.close);
    return;
  }

  const targetHit =
    strategyPosition.target != null &&
    ((direction === 1 && bar.high >= strategyPosition.target) ||
      (direction === -1 && bar.low <= strategyPosition.target));

  if (targetHit) {
    await flattenPosition('target', strategyPosition.target ?? bar.close);
  }
}

async function enterPosition(
  direction: 'long' | 'short',
  bar: SecondBar,
  rsiValue: number,
  adxValue: number | null,
) {
  if (!orderExecutor) return;

  const targetNet =
    tracker.position +
    (direction === 'long' ? CONFIG.numberOfContracts : -CONFIG.numberOfContracts);
  const success = await orderExecutor.moveTo(targetNet, {
    reason: `entry_${direction}`,
    limitAnchor: bar.close,
  });

  if (!success || tracker.position === 0) {
    log(`Entry ${direction} failed to fill`);
    return;
  }

  const tradeId = nextTradeId();
  strategyPosition = {
    direction,
    entryPrice: tracker.avgPrice,
    entryTime: bar.start.toISOString(),
    contracts: Math.abs(tracker.position),
    stop: computeStop(direction, tracker.avgPrice),
    target: computeTarget(direction, tracker.avgPrice),
    entryRSI: rsiValue,
    entryADX: adxValue,
    tradeId,
  };

  log(
    `ENTER ${direction.toUpperCase()} @ ${strategyPosition.entryPrice.toFixed(2)} ` +
      `(stop ${strategyPosition.stop?.toFixed(2) ?? 'n/a'}, target ${
        strategyPosition.target?.toFixed(2) ?? 'n/a'
      })`,
  );

  logTradeEvent({
    type: 'entry',
    direction,
    price: strategyPosition.entryPrice,
    contracts: strategyPosition.contracts,
    rsi: rsiValue,
    adx: adxValue,
    tradeId,
  });
}

async function flattenPosition(reason: string, anchor?: number) {
  if (!orderExecutor) return;
  if (tracker.position === 0) {
    strategyPosition = null;
    return;
  }
  const exitPrice = anchor ?? lastQuote?.lastPrice ?? strategyPosition?.entryPrice ?? 0;
  const success = await orderExecutor.moveTo(0, { reason, limitAnchor: anchor });
  if (success) {
    log(`EXIT position (${reason}), realized ${tracker.realized.toFixed(2)}`);
    logTradeEvent({
      type: 'exit',
      reason,
      realized: tracker.realized,
      exitPrice,
      tradeId: strategyPosition?.tradeId,
    });
    strategyPosition = null;
  } else {
    console.warn('[exit] Failed to flatten position');
  }
}

async function bootstrapHistorical(contractId: string) {
  const end = new Date();
  const start = new Date(end.getTime() - CONFIG.initialBackfillSeconds * 1000);
  log(
    `[bootstrap] Fetching ${CONFIG.initialBackfillSeconds}s of history (${start.toISOString()} -> ${end.toISOString()})`,
  );
  const bars = await fetchTopstepXFuturesBars({
    contractId,
    startTime: start.toISOString(),
    endTime: end.toISOString(),
    unit: 1,
    unitNumber: 1,
    limit: CONFIG.initialBackfillSeconds,
  });

  if (!bars.length) {
    throw new Error('No bootstrap data retrieved');
  }

  bars.reverse();
  for (const bar of bars) {
    ingestHistoricalBar(bar);
  }
  log(`[bootstrap] Seeded ${bars.length} bars`);
}

function ingestHistoricalBar(bar: TopstepXFuturesBar) {
  closes.push(bar.close);
  highs.push(bar.high);
  lows.push(bar.low);
  rsiIndicator.nextValue(bar.close);
  adxIndicator.nextValue({ high: bar.high, low: bar.low, close: bar.close });
}

async function startHubs(contractId: string) {
  const tokenProvider = async () => authenticate();
  const initialToken = await tokenProvider();

  marketHub = new HubConnectionBuilder()
    .withUrl(`${CONFIG.marketHubUrl}?access_token=${encodeURIComponent(initialToken)}`, {
      skipNegotiation: true,
      transport: HttpTransportType.WebSockets,
      accessTokenFactory: tokenProvider,
    })
    .withAutomaticReconnect()
    .configureLogging(LogLevel.Information)
    .build();

  marketHub.on('GatewayQuote', (_cid: string, quote: GatewayQuote) => {
    lastQuote = quote;
    const price = resolveQuotePrice(quote);
    const ts = new Date(quote.timestamp || quote.lastUpdated || Date.now());
    barBuilder.feed(price, ts);
  });

  const handleMarketTrade = (_cid: string, trade: any) => {
    if (!trade) return;
    const price = Number(trade.price ?? trade.lastPrice ?? 0);
    if (!price) return;
    const ts = new Date(trade.timestamp ?? trade.lastUpdated ?? Date.now());
    barBuilder.feed(price, ts);
  };
  marketHub.on('GatewayTrade', handleMarketTrade);
  marketHub.on('gatewaytrade', handleMarketTrade);

  await marketHub.start();
  log('Connected to market hub');

  const subscribeMarket = () => {
    marketHub?.invoke('SubscribeContractQuotes', contractId).catch(err =>
      console.error('Subscribe quotes failed', err),
    );
    marketHub?.invoke('SubscribeContractTrades', contractId).catch(err =>
      console.error('Subscribe trades failed', err),
    );
  };
  subscribeMarket();
  marketHub.onreconnected(subscribeMarket);

  userHub = new HubConnectionBuilder()
    .withUrl(`${CONFIG.userHubUrl}?access_token=${encodeURIComponent(initialToken)}`, {
      skipNegotiation: true,
      transport: HttpTransportType.WebSockets,
      accessTokenFactory: tokenProvider,
    })
    .withAutomaticReconnect()
    .configureLogging(LogLevel.Information)
    .build();

  const handleUserTrade = async (_cid: string, ev: any) => {
    const side: 'Buy' | 'Sell' = ev.side === 0 ? 'Buy' : 'Sell';
    const qty = Math.abs(ev.size ?? ev.quantity ?? ev.qty ?? 0);
    const price = Number(ev.price ?? ev.avgPrice ?? ev.fillPrice ?? 0);
    if (!qty || !price) return;
    tracker.onFill(side, qty, price);
    syncStrategyPositionFromTracker();
    logTradeEvent({
      type: 'fill',
      side,
      qty,
      price,
      cumulativePnL: tracker.realized,
      orderId: ev.id ?? ev.orderId,
      tradeId: strategyPosition?.tradeId,
    });
    await riskCheck();
  };
  userHub.on('GatewayUserTrade', handleUserTrade);
  userHub.on('gatewaytrade', handleUserTrade); // keep legacy casing until SignalR parity confirmed

  const handleUserPosition = (ev: any) => {
    if (!ev) return;
    const qty = Number(ev.netQty ?? ev.position ?? ev.size ?? 0);
    const avgPrice = Number(ev.avgPrice ?? ev.price ?? 0);
    tracker.position = qty;
    tracker.avgPrice = avgPrice;
    syncStrategyPositionFromTracker();
  };
  userHub.on('GatewayUserPosition', handleUserPosition);
  userHub.on('gatewayposition', handleUserPosition);

  await userHub.start();
  log('Connected to user hub');

  const subscribeUser = () => {
    userHub?.invoke('SubscribeAccounts').catch(() => {});
    if (CONFIG.accountId) {
      userHub?.invoke('SubscribeOrders', CONFIG.accountId).catch(() => {});
      userHub?.invoke('SubscribePositions', CONFIG.accountId).catch(() => {});
      userHub?.invoke('SubscribeTrades', CONFIG.accountId).catch(() => {});
    }
  };
  subscribeUser();
  userHub.onreconnected(subscribeUser);
}

function resolveQuotePrice(quote: GatewayQuote) {
  if (quote.lastPrice && quote.lastPrice > 0) return quote.lastPrice;
  if (quote.bestBid && quote.bestAsk) return (quote.bestBid + quote.bestAsk) / 2;
  if (quote.bestBid) return quote.bestBid;
  if (quote.bestAsk) return quote.bestAsk;
  return 0;
}

function syncStrategyPositionFromTracker() {
  if (tracker.position === 0) {
    strategyPosition = null;
    return;
  }

  const direction: 'long' | 'short' = tracker.position > 0 ? 'long' : 'short';
  const currentTradeId = strategyPosition?.tradeId ?? nextTradeId();
  strategyPosition = {
    direction,
    entryPrice: tracker.avgPrice,
    entryTime: nowIso(),
    contracts: Math.abs(tracker.position),
    stop: computeStop(direction, tracker.avgPrice),
    target: computeTarget(direction, tracker.avgPrice),
    entryRSI: prevRsiValue ?? 50,
    entryADX: null,
    tradeId: currentTradeId,
  };
}

async function riskCheck() {
  if (MAX_DD > 0 && -tracker.realized >= MAX_DD) {
    console.error('[risk] Max daily drawdown hit; flattening');
    await flattenPosition('max_dd');
    await shutdown('max_dd');
  }
}

async function shutdown(reason: string) {
  if (shuttingDown) return;
  shuttingDown = true;
  log(`Shutting down (${reason})...`);
  barBuilder.flushPending();
  if (tracker.position !== 0) {
    await flattenPosition('shutdown', lastQuote?.lastPrice);
  }
  if (marketHub) {
    await marketHub.stop().catch(() => {});
  }
  if (userHub) {
    await userHub.stop().catch(() => {});
  }
  log(`Shutdown complete. Realized PnL: ${tracker.realized.toFixed(2)}`);
  process.exit(0);
}

async function main() {
  if (!CONFIG.accountId) {
    throw new Error('TOPSTEPX_ACCOUNT_ID missing');
  }

  const lookup = CONFIG.contractId || CONFIG.symbol;
  contractMeta = await fetchTopstepXFuturesMetadata(lookup);
  if (!contractMeta) {
    throw new Error(`Unable to resolve metadata for ${lookup}`);
  }

  if (!CONFIG.contractId) {
    CONFIG.contractId = contractMeta.id;
  }

  tickSize = contractMeta.tickSize || tickSize;
  multiplier =
    contractMeta.tickValue && contractMeta.tickSize
      ? contractMeta.tickValue / contractMeta.tickSize
      : contractMeta.multiplier || multiplier;

  orderExecutor = new OrderExecutor({
    accountId: CONFIG.accountId,
    contractId: contractMeta.id,
    rest: restClient,
    tracker,
    getQuote: () => lastQuote,
    tickSize,
    decisionWindowMs: CONFIG.orderDecisionWindowMs,
    retryTickOffset: CONFIG.retryTickOffset,
    allowMarketFallback: CONFIG.allowMarketFallback,
    dryRun: DRY_RUN,
  });

  log(
    `Live SMA second strategy | SMA ${CONFIG.smaPeriod} | RSI ${CONFIG.rsiPeriod} | ADX ${CONFIG.adxPeriod} (${CONFIG.adxThreshold})`,
  );
  log(
    `Contracts ${CONFIG.numberOfContracts} | Stop ${(CONFIG.stopLossPercent * 100).toFixed(
      3,
    )}% | Target ${(CONFIG.takeProfitPercent * 100).toFixed(3)}%`,
  );

  await bootstrapHistorical(contractMeta.id);
  await startHubs(contractMeta.id);

  process.once('SIGINT', () => shutdown('SIGINT'));
  process.once('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGUSR2', () => {
    if (shuttingDown) {
      return;
    }
    log('Received SIGUSR2 - flatten request');
    flattenPosition('manual_signal').catch(err =>
      console.error('[signal] failed to flatten via SIGUSR2:', err),
    );
  });
}

main().catch(err => {
  console.error('Live SMA second strategy failed:', err);
  process.exit(1);
});
