import 'dotenv/config';
import { randomUUID } from 'crypto';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';
import {
  fetchBars,
  submitOrder,
  getOrder,
  cancelOrder,
  getClock,
  getPosition,
  sleep,
  submitOptionOrder,
  getOptionOrder,
  cancelOptionOrder,
  AlpacaOptionOrder,
  AlpacaOptionContract,
} from './lib/alpaca';
import { TwelveDataPriceFeed, MinuteBar, RealtimePriceSnapshot } from './lib/twelveData';
import { fetchTradierOptionExpirations, fetchTradierOptionChain } from './lib/tradier';
import { BollingerBands, EMA, ATR } from 'technicalindicators';
import {
  getCachedExpirations,
  setCachedExpirations,
  getCachedOptionChain,
  setCachedOptionChain,
  isCacheEntryFresh,
  DEFAULT_CACHE_MAX_AGE_MS,
} from './lib/optionCache';
import { StrategyHooks, RunningStrategy } from './lib/strategyHooks';
import { submitTopstepXOrder } from './lib/topstepx';

type DirectionRelation = 'above' | 'below';

interface ActivePosition {
  symbol: string;
  qty: number;
  entryPrice: number;
  entryTime: string;
  entryOrderId: string;
  instrument: 'stock' | 'option';
  multiplier: number;
  capitalUsed?: number;
  notionalExposure?: number;
  entrySma?: number;
  squeezeOnEntry?: boolean;
  optionSymbol?: string;
  optionStrike?: number;
  optionExpiration?: string;
  optionType?: 'call' | 'put';
  entryPremium?: number;
}

interface SymbolState {
  lastRelation: DirectionRelation | null;
  lastBarTime?: string;
  processing: boolean;
  queued: boolean;
  forceFlatten: boolean;
}

interface OrderFillResult {
  filledQty: number;
  avgPrice: number;
  status: string;
}

export const SMA_STRATEGY_ID = 'sma-crossover';

export interface SmaStrategyOptions {
  feed?: TwelveDataPriceFeed | null;
  hooks?: StrategyHooks;
  manageProcessSignals?: boolean;
}

const CONFIG = {
  symbols: (process.env.SMA_SYMBOLS || 'SPY')
    .split(',')
    .map(s => s.trim().toUpperCase())
    .filter(Boolean),
  fastPeriod: Number(process.env.SMA_FAST || '12'),
  slowPeriod: process.env.SMA_SLOW ? Number(process.env.SMA_SLOW) : null,
  orderQty: Number(process.env.SMA_ORDER_QTY || '1'),
  optionContracts: Number(process.env.SMA_OPTION_CONTRACTS || '1'),
  optionMinDte: Number(process.env.SMA_MIN_DTE || '3'),
  pollIntervalMs: Number(process.env.SMA_POLL_MS || '60000'),
  fillTimeoutMs: Number(process.env.SMA_FILL_TIMEOUT_MS || '15000'),
  timeframe: process.env.SMA_TIMEFRAME || '1Min',
  flattenMinutesBeforeClose: Number(process.env.SMA_FLATTEN_BEFORE_CLOSE || '5'),
  logFile: process.env.SMA_TRADE_LOG || './logs/sma-crossover-trades.jsonl',
  priceCross: process.env.SMA_PRICE_CROSS === 'true',
  useSqueezeFilter: process.env.SMA_USE_SQUEEZE === 'false' ? false : true,
  twelveDataApiKey: process.env.TWELVE_DATA_API_KEY || '',
  twelveDataBackupApiKey: process.env.TWELVE_DATA_BACKUP_API_KEY || '',
  twelveDataUrl: process.env.TWELVE_DATA_WS_URL || '',
  useTwelveData: process.env.SMA_USE_TWELVE_DATA
    ? process.env.SMA_USE_TWELVE_DATA === 'true'
    : !!process.env.TWELVE_DATA_API_KEY,
  tradeMode: (process.env.SMA_TRADE_MODE || 'stock').toLowerCase(),
};

CONFIG.timeframe = '1Min';
CONFIG.priceCross = true;
CONFIG.slowPeriod = null;
CONFIG.tradeMode = 'option'; // Trade options (calls and puts) to match backtest

const BOLLINGER_PERIOD = 20;
const BOLLINGER_STD_DEV = 2;
const KELTNER_PERIOD = 20;
const KELTNER_MULTIPLIER = 1.5;

const STRATEGY = {
  tradeBudget: Number(process.env.SMA_TRADE_BUDGET || '200'),
  leverage: Math.max(1, Number(process.env.SMA_LEVERAGE || '100')),
  stopLossPercent: Math.max(0.0001, Number(process.env.SMA_STOP_LOSS_PCT || '1') / 100),
  slippagePercent: Math.max(0, Number(process.env.SMA_SLIPPAGE_PCT || '0.03') / 100),
};

if (CONFIG.symbols.length === 0) {
  throw new Error('No symbols configured; set SMA_SYMBOLS env variable.');
}

if (!Number.isFinite(CONFIG.fastPeriod) || CONFIG.fastPeriod <= 1) {
  throw new Error('Invalid SMA_FAST period; must be > 1.');
}

if (!CONFIG.priceCross) {
  if (CONFIG.slowPeriod == null) {
    throw new Error('SMA_SLOW must be provided when SMA_PRICE_CROSS is false.');
  }
  if (!Number.isFinite(CONFIG.slowPeriod) || (CONFIG.slowPeriod as number) <= CONFIG.fastPeriod) {
    throw new Error('Invalid SMA_SLOW period; must be greater than SMA_FAST.');
  }
} else {
  CONFIG.slowPeriod = null;
}

if (!Number.isFinite(CONFIG.orderQty) || CONFIG.orderQty <= 0) {
  throw new Error('Invalid SMA_ORDER_QTY; must be positive.');
}

const USE_STOCK_TRADING = false; // Trade options instead of stocks
const OPTION_DATA_RETRY_MS = 5 * 60 * 1000;
const OPTION_CACHE_MAX_AGE_MS =
  Number(process.env.ALPACA_OPTION_CACHE_MAX_AGE_MS || '') || DEFAULT_CACHE_MAX_AGE_MS;
const OPTION_EXPIRATION_EOD_SUFFIX = 'T21:00:00Z';

const optionDataCooldownUntil = new Map<string, number>();

type PriceBar = { t: string; o: number; h: number; l: number; c: number; partial?: boolean };

const TIMEFRAME_MINUTES = parseTimeframeMinutes(CONFIG.timeframe);
const USE_TWELVE_DATA_FEED =
  CONFIG.useTwelveData && !!CONFIG.twelveDataApiKey && TIMEFRAME_MINUTES !== null;

if (CONFIG.useTwelveData && !CONFIG.twelveDataApiKey) {
  console.warn(
    '[sma-crossover] SMA_USE_TWELVE_DATA enabled but TWELVE_DATA_API_KEY missing; falling back to Alpaca data.',
  );
}

if (CONFIG.useTwelveData && TIMEFRAME_MINUTES === null) {
  console.warn(
    `[sma-crossover] Timeframe '${CONFIG.timeframe}' is not minute-based; Twelve Data feed disabled.`,
  );
}

// For squeeze calculation we need at least 40 bars (20 Bollinger/Keltner + 20 momentum)
// Request enough for a full trading day (390 minutes) to ensure we have data even at market open
export const SMA_TWELVE_DATA_MAX_MINUTE_BARS = Math.max(
  (CONFIG.fastPeriod + (CONFIG.slowPeriod ?? CONFIG.fastPeriod)) * (TIMEFRAME_MINUTES ?? 1) * 3,
  390, // Full trading day worth of minute bars
);

let twelveDataFeed: TwelveDataPriceFeed | null = null;
let twelveDataFeedOwned = false;
let removePriceListener: (() => void) | null = null;
let removeConnectionListener: (() => void) | null = null;
let strategyHooks: StrategyHooks | undefined;
let shouldExitProcessOnShutdown = true;
let processSignalsRegistered = false;

const LOG_DIR = path.dirname(CONFIG.logFile);

const positions = new Map<string, ActivePosition>();
const symbolState = new Map<string, SymbolState>();

let realizedPnL = 0;
let shuttingDown = false;
let lastClock: { timestamp: string; is_open: boolean; next_open: string; next_close: string } | null =
  null;

function notifyPnLUpdate() {
  strategyHooks?.onPnLUpdate?.(SMA_STRATEGY_ID, realizedPnL);
}

function initializeTwelveDataFeed(providedFeed?: TwelveDataPriceFeed | null) {
  if (providedFeed) {
    twelveDataFeed = providedFeed;
    twelveDataFeedOwned = false;
    return;
  }
  if (USE_TWELVE_DATA_FEED) {
    if (!CONFIG.twelveDataApiKey) {
      twelveDataFeed = null;
      twelveDataFeedOwned = false;
      return;
    }
    twelveDataFeed = new TwelveDataPriceFeed({
      apiKey: CONFIG.twelveDataApiKey,
      backupApiKey: CONFIG.twelveDataBackupApiKey || undefined,
      symbols: CONFIG.symbols,
      url: CONFIG.twelveDataUrl || undefined,
      maxMinuteBars: SMA_TWELVE_DATA_MAX_MINUTE_BARS,
    });
    twelveDataFeedOwned = true;
    return;
  }
  twelveDataFeed = null;
  twelveDataFeedOwned = false;
}

function detachFeedListeners() {
  removePriceListener?.();
  removePriceListener = null;
  removeConnectionListener?.();
  removeConnectionListener = null;
}

function registerProcessSignalHandlers() {
  if (processSignalsRegistered) {
    return;
  }
  process.on('SIGINT', () => {
    shutdown('SIGINT').catch(err => {
      console.error('Shutdown error', err);
      process.exit(1);
    });
  });

  process.on('SIGTERM', () => {
    shutdown('SIGTERM').catch(err => {
      console.error('Shutdown error', err);
      process.exit(1);
    });
  });
  processSignalsRegistered = true;
}

function getSymbolState(symbol: string): SymbolState {
  let state = symbolState.get(symbol);
  if (!state) {
    state = {
      lastRelation: null,
      processing: false,
      queued: false,
      forceFlatten: false,
    };
    symbolState.set(symbol, state);
  }
  return state;
}

function isMarketOpen(): boolean {
  if (!lastClock) {
    return true;
  }
  return lastClock.is_open;
}

function computeFlattenFlag(): boolean {
  if (!lastClock) {
    return false;
  }
  if (!lastClock.is_open) {
    return true;
  }
  const clockTime = new Date(lastClock.timestamp);
  return shouldFlattenForClose(clockTime);
}

function scheduleSymbolEvaluation(symbol: string, options: { forceFlatten?: boolean } = {}) {
  const state = getSymbolState(symbol);
  if (options.forceFlatten) {
    state.forceFlatten = true;
  }
  if (state.processing) {
    state.queued = true;
    return;
  }
  state.processing = true;
  state.queued = false;
  setImmediate(async () => {
    try {
      const flattenNow = state.forceFlatten || computeFlattenFlag();
      state.forceFlatten = false;
      if (!isMarketOpen() && !flattenNow) {
        return;
      }
      await evaluateSymbol(symbol, flattenNow);
    } catch (err) {
      log(symbol, `Evaluation error: ${(err as Error).message}`);
    } finally {
      state.processing = false;
      if (state.queued) {
        state.queued = false;
        scheduleSymbolEvaluation(symbol);
      }
    }
  });
}

function nowIso(): string {
  return new Date().toISOString();
}

function log(symbol: string, message: string) {
  console.log(`[${new Date().toISOString()}][${symbol}] ${message}`);
}

function ensureLogDir() {
  if (!existsSync(LOG_DIR)) {
    mkdirSync(LOG_DIR, { recursive: true });
  }
}

function logTradeEvent(event: Record<string, any>) {
  try {
    ensureLogDir();
    const payload = { timestamp: nowIso(), ...event };
    appendFileSync(CONFIG.logFile, `${JSON.stringify(payload)}\n`);
    strategyHooks?.onTradeEvent?.(SMA_STRATEGY_ID, payload);
  } catch (err) {
    console.error('[trade-log] failed to write entry', err);
  }
}

function makeSeriesAccessor<T>(series: T[], offset: number) {
  return (index: number): T | null => {
    const relative = index - offset;
    if (relative < 0 || relative >= series.length) {
      return null;
    }
    return series[relative];
  };
}

interface SqueezeResult {
  isOn: boolean;
  momentum: number;
  bbMidpoint: number; // Bollinger Band midpoint for direction detection
}

function calculateSqueezeSeries(bars: PriceBar[]): Array<SqueezeResult | null> {
  const closes = bars.map(bar => bar.c);
  const highs = bars.map(bar => (bar.h ?? bar.c));
  const lows = bars.map(bar => (bar.l ?? bar.c));

  // Debug: Check if we have enough bars and valid data
  if (bars.length < 40) {
    console.log(`[squeeze] Not enough bars: ${bars.length} (need 40+)`);
  } else {
    const sampleBar = bars[bars.length - 1];
    console.log(`[squeeze] Have ${bars.length} bars. Latest bar h=${sampleBar.h}, l=${sampleBar.l}, c=${sampleBar.c}`);
  }

  const bollingerSeries = BollingerBands.calculate({
    values: closes,
    period: BOLLINGER_PERIOD,
    stdDev: BOLLINGER_STD_DEV,
  });
  const bollingerAccessor = makeSeriesAccessor(
    bollingerSeries,
    closes.length - bollingerSeries.length,
  );

  const emaSeries = EMA.calculate({ period: KELTNER_PERIOD, values: closes });
  const emaAccessor = makeSeriesAccessor(emaSeries, closes.length - emaSeries.length);

  const atrSeries = ATR.calculate({
    period: KELTNER_PERIOD,
    high: highs,
    low: lows,
    close: closes,
  });
  const atrAccessor = makeSeriesAccessor(atrSeries, closes.length - atrSeries.length);

  // Calculate squeeze momentum using linear regression
  const momentumLength = 20;
  const momentum: Array<number | null> = closes.map((close, idx) => {
    if (idx < momentumLength) return null;

    // Linear regression of close prices
    const prices = closes.slice(idx - momentumLength + 1, idx + 1);
    const n = prices.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += prices[i];
      sumXY += i * prices[i];
      sumX2 += i * i;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    const linReg = slope * (n - 1) + intercept;

    // Momentum is the difference from the midpoint of Bollinger Bands
    const bb = bollingerAccessor(idx);
    if (!bb) return null;
    const bbMid = (bb.upper + bb.lower) / 2;

    return linReg - bbMid;
  });

  return closes.map((_, idx) => {
    const bb = bollingerAccessor(idx);
    const ema = emaAccessor(idx);
    const atr = atrAccessor(idx);
    const mom = momentum[idx];

    if (!bb || ema == null || atr == null || mom == null) {
      return null;
    }

    const range = atr * KELTNER_MULTIPLIER;
    const upper = ema + range;
    const lower = ema - range;
    const isOn = bb.upper <= upper && bb.lower >= lower;
    const bbMidpoint = (bb.upper + bb.lower) / 2;

    return { isOn, momentum: mom, bbMidpoint };
  });
}

function calculateSmaSeries(values: number[], period: number): Array<number | null> {
  const result = new Array<number | null>(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i];
    if (i >= period) {
      sum -= values[i - period];
    }
    if (i >= period - 1) {
      result[i] = sum / period;
    }
  }
  return result;
}

function parseTimeframeMinutes(value: string): number | null {
  if (!value) {
    return null;
  }
  const match = /^(\d+)\s*(min)$/i.exec(value.trim());
  if (!match) {
    return null;
  }
  const minutes = Number(match[1]);
  return Number.isFinite(minutes) && minutes > 0 ? minutes : null;
}

function minuteBarToPriceBar(bar: MinuteBar): PriceBar {
  return {
    t: bar.t,
    o: bar.o,
    h: bar.h,
    l: bar.l,
    c: bar.c,
    partial: bar.partial ?? false,
  };
}

interface AggregatedBucket {
  startMs: number;
  open: number;
  high: number;
  low: number;
  close: number;
  minuteStarts: Set<number>;
  partial: boolean;
}

function aggregateMinuteBars(minuteBars: MinuteBar[], minutesPerBar: number): PriceBar[] {
  if (minutesPerBar <= 1) {
    return minuteBars.map(minuteBarToPriceBar);
  }

  const sorted = [...minuteBars].sort((a, b) => Date.parse(a.t) - Date.parse(b.t));
  const bucketMs = minutesPerBar * 60_000;
  const buckets = new Map<number, AggregatedBucket>();

  for (const bar of sorted) {
    const minuteStart = Math.floor(Date.parse(bar.t) / 60_000) * 60_000;
    if (!Number.isFinite(minuteStart)) {
      continue;
    }
    const bucketStart = Math.floor(minuteStart / bucketMs) * bucketMs;
    let bucket = buckets.get(bucketStart);
    if (!bucket) {
      bucket = {
        startMs: bucketStart,
        open: bar.o,
        high: bar.h,
        low: bar.l,
        close: bar.c,
        minuteStarts: new Set<number>(),
        partial: false,
      };
      buckets.set(bucketStart, bucket);
    }
    bucket.high = Math.max(bucket.high, bar.h);
    bucket.low = Math.min(bucket.low, bar.l);
    // Update close each time to latest bar close
    bucket.close = bar.c;
    bucket.minuteStarts.add(minuteStart);
    bucket.partial = bucket.partial || !!bar.partial;
  }

  const aggregated = Array.from(buckets.values()).sort(
    (a, b) => a.startMs - b.startMs,
  );

  return aggregated.map(bucket => ({
    t: new Date(bucket.startMs).toISOString(),
    o: bucket.open,
    h: bucket.high,
    l: bucket.low,
    c: bucket.close,
    partial: bucket.partial || bucket.minuteStarts.size < minutesPerBar,
  }));
}

function updateBarsWithSnapshot(
  bars: PriceBar[],
  snapshot: RealtimePriceSnapshot | undefined,
  maxBars: number,
): PriceBar[] {
  if (!snapshot) {
    return bars;
  }
  const snapshotMinuteStart = Math.floor(snapshot.timestamp / 60_000) * 60_000;
  if (!Number.isFinite(snapshotMinuteStart)) {
    return bars;
  }

  if (bars.length === 0) {
    return [
      {
        t: new Date(snapshotMinuteStart).toISOString(),
        o: snapshot.price,
        h: snapshot.price,
        l: snapshot.price,
        c: snapshot.price,
        partial: true,
      },
    ];
  }

  const lastIndex = bars.length - 1;
  const lastBarTime = Date.parse(bars[lastIndex].t);
  if (!Number.isFinite(lastBarTime)) {
    return bars;
  }

  if (snapshotMinuteStart < lastBarTime) {
    return bars;
  }

  if (snapshotMinuteStart === lastBarTime) {
    const updated = [...bars];
    const lastBar = { ...updated[lastIndex] };
    const price = snapshot.price;
    lastBar.partial = true;
    lastBar.c = price;
    const existingHigh = lastBar.h ?? price;
    const existingLow = lastBar.l ?? price;
    lastBar.h = Math.max(existingHigh, price);
    lastBar.l = Math.min(existingLow, price);
    updated[lastIndex] = lastBar;
    return updated;
  }

  const previousClose = bars[lastIndex].c ?? snapshot.price;
  const newBar: PriceBar = {
    t: new Date(snapshotMinuteStart).toISOString(),
    o: previousClose,
    h: snapshot.price,
    l: snapshot.price,
    c: snapshot.price,
    partial: true,
  };
  const extended = [...bars, newBar];
  if (extended.length > maxBars) {
    return extended.slice(-maxBars);
  }
  return extended;
}

async function loadRecentBars(symbol: string, minBars: number): Promise<PriceBar[]> {
  const required = Math.max(minBars, 1);

  if (twelveDataFeed) {
    const minutesPerBar = TIMEFRAME_MINUTES ?? 1;
    const buffer = Math.max(5, minutesPerBar * 2);
    const minuteBarsNeeded = (required + buffer) * minutesPerBar;
    const minuteBars = twelveDataFeed.getRecentBars(symbol, '1Min', minuteBarsNeeded);
    if (minuteBars.length === 0) {
      return [];
    }
    const aggregated = aggregateMinuteBars(minuteBars, minutesPerBar);
    if (aggregated.length <= required) {
      return aggregated;
    }
    return aggregated.slice(-required);
  }

  const alpacaBars = await fetchBars(symbol, CONFIG.timeframe, {
    limit: Math.max(required + 5, required),
  });
  return alpacaBars.map(bar => ({
    t: bar.t,
    o: bar.o,
    h: bar.h,
    l: bar.l,
    c: bar.c,
  }));
}

function easternTimeParts(date: Date): { hour: number; minute: number } {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
  });
  const [hour, minute] = formatter.format(date).split(':').map(Number);
  return {
    hour: Number.isFinite(hour) ? hour : 0,
    minute: Number.isFinite(minute) ? minute : 0,
  };
}

function shouldFlattenForClose(clockTime: Date): boolean {
  if (CONFIG.flattenMinutesBeforeClose <= 0) {
    return false;
  }
  const { hour, minute } = easternTimeParts(clockTime);
  if (hour > 15) return true;
  if (hour === 15 && minute >= 60 - CONFIG.flattenMinutesBeforeClose) return true;
  return false;
}

async function waitForOrderFill(orderId: string, timeoutMs: number): Promise<OrderFillResult> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const order = await getOrder(orderId);
    const filledQty = Number(order.filled_qty || '0');
    const avgPrice = Number(order.filled_avg_price || '0');
    if (order.status === 'filled') {
      return { filledQty, avgPrice, status: order.status };
    }
    if (
      order.status === 'canceled' ||
      order.status === 'rejected' ||
      order.status === 'stopped' ||
      order.status === 'suspended'
    ) {
      return { filledQty, avgPrice, status: order.status };
    }
    await sleep(1000);
  }

  try {
    await cancelOrder(orderId);
  } catch (err) {
    log('system', `Failed to cancel order ${orderId}: ${(err as Error).message}`);
  }

  const order = await getOrder(orderId);
  return {
    filledQty: Number(order.filled_qty || '0'),
    avgPrice: Number(order.filled_avg_price || '0'),
    status: order.status,
  };
}

async function waitForOptionFill(
  orderId: string,
  timeoutMs: number,
  cancelOnTimeout = true,
): Promise<OrderFillResult> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const order = await getOptionOrder(orderId);
    const filledQty = Number(order.filled_qty || '0');
    const avgPrice = Number(order.filled_avg_price || '0');
    if (order.status === 'filled') {
      return { filledQty, avgPrice, status: order.status };
    }
    if (
      order.status === 'canceled' ||
      order.status === 'rejected' ||
      order.status === 'stopped' ||
      order.status === 'suspended' ||
      order.status === 'expired'
    ) {
      return { filledQty, avgPrice, status: order.status };
    }
    await sleep(1000);
  }

  if (cancelOnTimeout) {
    try {
      await cancelOptionOrder(orderId);
    } catch (err) {
      log('system', `Failed to cancel option order ${orderId}: ${(err as Error).message}`);
    }
  }

  const order = await getOptionOrder(orderId);
  return {
    filledQty: Number(order.filled_qty || '0'),
    avgPrice: Number(order.filled_avg_price || '0'),
    status: order.status,
  };
}

async function submitOptionMarketOrder(
  contractSymbol: string,
  side: 'buy' | 'sell',
  quantity: number,
  positionEffect: 'open' | 'close',
): Promise<OrderFillResult | null> {
  if (quantity <= 0) {
    return null;
  }
  try {
    const order = await submitOptionOrder({
      symbol: contractSymbol,
      side,
      qty: String(quantity),
      type: 'market',
      time_in_force: 'day',
      position_effect: positionEffect,
      client_order_id: randomUUID(),
    });
    return waitForOptionFill(order.id, CONFIG.fillTimeoutMs);
  } catch (err) {
    log(contractSymbol, `Option market order failed (${side} ${quantity}): ${(err as Error).message}`);
    return null;
  }
}

function optionDataOnCooldown(symbol: string): boolean {
  const until = optionDataCooldownUntil.get(symbol);
  return typeof until === 'number' && until > Date.now();
}

function setOptionDataCooldown(symbol: string) {
  optionDataCooldownUntil.set(symbol, Date.now() + OPTION_DATA_RETRY_MS);
}

function handleOptionDataFetchError(symbol: string, context: string, err: unknown) {
  const message = err instanceof Error ? err.message : String(err);
  log(symbol, `Tradier ${context} fetch failed: ${message}; backing off for 5 minutes`);
  setOptionDataCooldown(symbol);
}

interface ExpirationFetchResult {
  expirations: string[];
  source: 'live' | 'cache';
}

interface ChainFetchResult {
  contracts: AlpacaOptionContract[];
  source: 'live' | 'cache';
  expiration: string;
}

function cacheAgeDescription(updatedAt: string): string {
  const parsed = new Date(updatedAt).getTime();
  if (!Number.isFinite(parsed)) {
    return 'unknown age';
  }
  const ageMinutes = Math.max(0, Math.round((Date.now() - parsed) / 60000));
  if (ageMinutes >= 1440) {
    const days = Math.round(ageMinutes / 1440);
    return `${days}d`;
  }
  if (ageMinutes >= 60) {
    const hours = Math.round(ageMinutes / 60);
    return `${hours}h`;
  }
  return `${ageMinutes}m`;
}

async function getOptionExpirationsWithFallback(
  symbol: string,
  skipLive: boolean,
): Promise<ExpirationFetchResult | null> {
  if (!skipLive) {
    try {
      const expirations = await fetchTradierOptionExpirations(symbol);
      if (Array.isArray(expirations) && expirations.length > 0) {
        await setCachedExpirations(symbol, expirations);
        return { expirations, source: 'live' };
      }
      log(symbol, 'Tradier returned no option expirations; checking cache');
    } catch (err) {
      handleOptionDataFetchError(symbol, 'expirations', err);
    }
  } else {
    log(symbol, 'Option data on cooldown; using cached expirations if available');
  }

  const cached = await getCachedExpirations(symbol);
  if (cached && cached.expirations.length > 0) {
    if (isCacheEntryFresh(cached.updatedAt, OPTION_CACHE_MAX_AGE_MS)) {
      log(
        symbol,
        `Using cached option expirations (${cached.expirations.length}) updated ${cacheAgeDescription(cached.updatedAt)} ago`,
      );
      return { expirations: cached.expirations, source: 'cache' };
    }
    log(symbol, `Cached option expirations are stale (updated ${cached.updatedAt}); ignoring`);
  }

  return null;
}

async function getOptionChainWithFallback(
  symbol: string,
  expiration: string,
  skipLive: boolean,
): Promise<ChainFetchResult | null> {
  if (!skipLive) {
    try {
      const chain = await fetchTradierOptionChain(symbol, expiration);
      if (Array.isArray(chain) && chain.length > 0) {
        await setCachedOptionChain(symbol, expiration, chain);
        return { contracts: chain, source: 'live', expiration };
      }
      log(symbol, `Tradier returned empty option chain for ${expiration}; checking cache`);
    } catch (err) {
      handleOptionDataFetchError(symbol, `option chain ${expiration}`, err);
    }
  } else {
    log(symbol, `Option data on cooldown; using cached chain for ${expiration} if available`);
  }

  const cached = await getCachedOptionChain(symbol, expiration);
  if (cached && cached.contracts.length > 0) {
    if (isCacheEntryFresh(cached.updatedAt, OPTION_CACHE_MAX_AGE_MS)) {
      log(
        symbol,
        `Using cached option chain for ${expiration} (${cached.contracts.length} contracts, updated ${cacheAgeDescription(cached.updatedAt)} ago)`,
      );
      return { contracts: cached.contracts, source: 'cache', expiration };
    }
    log(symbol, `Cached option chain for ${expiration} is stale (updated ${cached.updatedAt}); ignoring`);
  }

  return null;
}

function computeDte(from: Date, to: Date): number {
  const diffMs = to.getTime() - from.getTime();
  if (diffMs <= 0) return 0;
  return Math.ceil(diffMs / (24 * 60 * 60 * 1000));
}

async function selectOptionContract(
  symbol: string,
  optionType: 'call' | 'put',
  referencePrice: number,
  minDte: number,
): Promise<AlpacaOptionContract | null> {
  const skipLiveExpirations = optionDataOnCooldown(symbol);
  const expirationResult = await getOptionExpirationsWithFallback(symbol, skipLiveExpirations);
  if (!expirationResult || expirationResult.expirations.length === 0) {
    log(
      symbol,
      skipLiveExpirations
        ? 'Option data on cooldown and no cached expirations available'
        : 'No option expirations available from Tradier',
    );
    return null;
  }

  const now = new Date();
  const withDte = expirationResult.expirations
    .map(dateStr => {
      const expDate = new Date(`${dateStr}${OPTION_EXPIRATION_EOD_SUFFIX}`);
      return {
        dateStr,
        dte: computeDte(now, expDate),
      };
    })
    .filter(item => Number.isFinite(item.dte) && item.dte > 0);

  if (withDte.length === 0) {
    log(symbol, 'Cached option expirations are all in the past; waiting for new data');
    return null;
  }

  const valid = withDte.filter(item => item.dte >= minDte);
  const chosen = (valid.length > 0 ? valid : withDte).sort((a, b) => a.dte - b.dte)[0];
  if (!chosen) {
    return null;
  }

  const skipLiveChain = optionDataOnCooldown(symbol);
  const chainResult = await getOptionChainWithFallback(symbol, chosen.dateStr, skipLiveChain);
  if (!chainResult || chainResult.contracts.length === 0) {
    log(
      symbol,
      skipLiveChain
        ? `Option data on cooldown and no cached chain for ${chosen.dateStr}`
        : `No option chain available for ${chosen.dateStr}`,
    );
    return null;
  }

  const filtered = chainResult.contracts.filter(contract => contract.option_type === optionType);
  if (filtered.length === 0) {
    log(symbol, `No ${optionType.toUpperCase()} contracts available for ${chosen.dateStr}`);
    return null;
  }

  filtered.sort(
    (a, b) =>
      Math.abs(a.strike_price - referencePrice) - Math.abs(b.strike_price - referencePrice),
  );
  return filtered[0] ?? null;
}

async function syncExistingPositions() {
  for (const symbol of CONFIG.symbols) {
    try {
      const position = await getPosition(symbol);
      if (position) {
        const qty = Number(position.qty);
        if (qty > 0) {
          const existing: ActivePosition = {
            symbol,
            qty,
            entryPrice: Number(position.avg_entry_price),
            entryTime: nowIso(),
            entryOrderId: 'existing-position',
            instrument: 'stock',
            multiplier: 1,
            capitalUsed: STRATEGY.tradeBudget,
            notionalExposure: qty * Number(position.avg_entry_price),
            entrySma: Number(position.avg_entry_price),
            squeezeOnEntry: false,
          };
          positions.set(symbol, existing);
          log(symbol, `Detected existing LONG position (${qty} @ ${existing.entryPrice.toFixed(2)})`);
          logTradeEvent({
            type: 'existing_position',
            instrument: 'stock',
            symbol,
            qty,
            entryPrice: existing.entryPrice,
            detectedAt: existing.entryTime,
          });
        } else if (positions.has(symbol)) {
          positions.delete(symbol);
        }
      } else {
        positions.delete(symbol);
      }
    } catch (err) {
      log(symbol, `Failed to sync position: ${(err as Error).message}`);
    }
  }
}

async function openLongPosition(symbol: string, priceContext: number, fastSma: number, squeezeOn: boolean, optionType: 'call' | 'put' = 'call') {
  if (positions.has(symbol)) {
    return;
  }
  if (USE_STOCK_TRADING) {
    await openLongStockPosition(symbol, priceContext, fastSma, squeezeOn);
  } else {
    await openOptionPosition(symbol, priceContext, fastSma, null, optionType);
  }
}

async function openLongStockPosition(
  symbol: string,
  priceContext: number,
  fastSma: number,
  squeezeOn: boolean,
) {
  const targetPrice = priceContext * (1 + STRATEGY.slippagePercent);
  const desiredExposure = STRATEGY.tradeBudget * STRATEGY.leverage;
  const qty = Math.max(1, Math.floor(desiredExposure / targetPrice));
  if (!Number.isFinite(qty) || qty <= 0) {
    log(symbol, `Calculated quantity invalid (${qty}); skipping entry.`);
    return;
  }
  const order = await submitOrder({
    symbol,
    side: 'buy',
    qty: qty.toString(),
    type: 'market',
    time_in_force: 'day',
    client_order_id: randomUUID(),
  });
  const fill = await waitForOrderFill(order.id, CONFIG.fillTimeoutMs);
  if (fill.filledQty <= 0) {
    log(symbol, `Entry order ${order.id} did not fill (status ${fill.status})`);
    logTradeEvent({
      type: 'entry_failed',
      symbol,
      orderId: order.id,
      status: fill.status,
      requestedQty: qty,
    });
    return;
  }

  const position: ActivePosition = {
    symbol,
    qty: fill.filledQty,
    entryPrice: fill.avgPrice || priceContext,
    entryTime: nowIso(),
    entryOrderId: order.id,
    instrument: 'stock',
    multiplier: 1,
    capitalUsed: STRATEGY.tradeBudget,
    notionalExposure: (fill.avgPrice || priceContext) * fill.filledQty,
    entrySma: fastSma,
    squeezeOnEntry: squeezeOn,
  };
  positions.set(symbol, position);
  log(
    symbol,
    `Entered LONG ${fill.filledQty} @ ${position.entryPrice.toFixed(
      2,
    )} (price above SMA${CONFIG.fastPeriod} | squeeze OFF)`,
  );
  logTradeEvent({
    type: 'entry',
    instrument: 'stock',
    symbol,
    side: 'long',
    qty: fill.filledQty,
    entryPrice: position.entryPrice,
    capitalUsed: position.capitalUsed,
    notionalExposure: position.notionalExposure,
    entrySma: fastSma,
    squeezeOn,
    entryOrderId: order.id,
    priceContext,
    fastSma,
  });
}

async function openOptionPosition(
  symbol: string,
  priceContext: number,
  fastSma: number,
  slowSma: number | null,
  optionType: 'call' | 'put',
) {
  const contracts = CONFIG.optionContracts;
  if (contracts <= 0) {
    log(symbol, 'SMA option contracts set to 0; skipping entry');
    return;
  }
  const contract = await selectOptionContract(
    symbol,
    optionType,
    priceContext,
    Math.max(1, CONFIG.optionMinDte),
  );
  if (!contract) {
    log(symbol, 'No suitable option contract found for SMA entry');
    return;
  }
  log(
    symbol,
    `Attempting option entry ${contract.symbol} (${contracts} contracts) with underlying ${priceContext.toFixed(
      2,
    )}`,
  );
  const fill = await submitOptionMarketOrder(contract.symbol, 'buy', contracts, 'open');
  if (!fill || fill.filledQty === 0) {
    log(symbol, 'Option entry order did not fill');
    logTradeEvent({
      type: 'entry_failed',
      instrument: 'option',
      symbol,
      optionSymbol: contract.symbol,
      strike: contract.strike_price,
      expiration: contract.expiration_date,
      requestedQty: contracts,
    });
    return;
  }

  const position: ActivePosition = {
    symbol,
    qty: fill.filledQty,
    entryPrice: priceContext,
    entryPremium: fill.avgPrice,
    entryTime: nowIso(),
    entryOrderId: `option-${randomUUID()}`,
    instrument: 'option',
    multiplier: 100,
    optionSymbol: contract.symbol,
    optionStrike: contract.strike_price,
    optionExpiration: contract.expiration_date,
    optionType,
  };
  positions.set(symbol, position);
  const slowText = '';
  log(
    symbol,
    `Entered option ${contract.symbol} (${fill.filledQty} @ ${fill.avgPrice.toFixed(
      2,
    )}) (price above SMA${CONFIG.fastPeriod}${slowText})`,
  );
  logTradeEvent({
    type: 'entry',
    instrument: 'option',
    symbol,
    optionSymbol: contract.symbol,
    strike: contract.strike_price,
    expiration: contract.expiration_date,
    side: 'long',
    qty: fill.filledQty,
    entryPrice: fill.avgPrice,
    entryOrderId: position.entryOrderId,
    priceContext,
    fastSma,
  });
}

async function closeLongPosition(
  symbol: string,
  reason: 'bearish_cross' | 'bullish_cross' | 'flatten' | 'shutdown' | 'stop_loss',
  priceContext: number,
  fastSma: number | null,
) {
  const position = positions.get(symbol);
  if (!position) {
    return;
  }

  if (position.instrument === 'stock') {
    const order = await submitOrder({
      symbol,
      side: 'sell',
      qty: position.qty.toString(),
      type: 'market',
      time_in_force: 'day',
      client_order_id: randomUUID(),
    });
    const fill = await waitForOrderFill(order.id, CONFIG.fillTimeoutMs);
    if (fill.filledQty <= 0) {
      log(symbol, `Exit order ${order.id} did not fill (status ${fill.status})`);
      logTradeEvent({
        type: 'exit_failed',
        instrument: 'stock',
        symbol,
        reason,
        orderId: order.id,
        status: fill.status,
        remainingQty: position.qty,
      });
      return;
    }

    const exitPrice = fill.avgPrice || priceContext;
    const pnl = (exitPrice - position.entryPrice) * fill.filledQty;
    realizedPnL += pnl;
    notifyPnLUpdate();

    log(
      symbol,
      `Exited ${fill.filledQty} @ ${exitPrice.toFixed(2)} (reason: ${reason}, PnL: ${pnl.toFixed(2)})`,
    );
    logTradeEvent({
      type: 'exit',
      instrument: 'stock',
      symbol,
      reason,
      qty: fill.filledQty,
      exitPrice,
      exitOrderId: order.id,
      entryPrice: position.entryPrice,
      entryTime: position.entryTime,
      exitTime: nowIso(),
      pnl,
      pnlPct: position.capitalUsed ? pnl / position.capitalUsed : 0,
      fastSma,
      priceContext,
      capitalUsed: position.capitalUsed,
      notionalExposure: position.notionalExposure,
      entrySma: position.entrySma,
      squeezeOnEntry: position.squeezeOnEntry,
      realizedPnLTally: realizedPnL,
    });

    if (fill.filledQty >= position.qty) {
      positions.delete(symbol);
    } else {
      position.qty -= fill.filledQty;
      if (position.qty <= 0) {
        positions.delete(symbol);
      } else {
        log(symbol, `Partial exit completed; ${position.qty} shares remain open.`);
        positions.set(symbol, position);
      }
    }
    return;
  }

  const fill = await submitOptionMarketOrder(position.optionSymbol!, 'sell', position.qty, 'close');
  if (!fill || fill.filledQty === 0) {
    log(symbol, `Option exit order did not fill (reason ${reason})`);
    logTradeEvent({
      type: 'exit_failed',
      instrument: 'option',
      symbol,
      optionSymbol: position.optionSymbol,
      reason,
      remainingQty: position.qty,
    });
    return;
  }

  const exitPremium = fill.avgPrice ?? 0;
  const entryPremium = position.entryPremium ?? 0;
  const pnl = (exitPremium - entryPremium) * fill.filledQty * position.multiplier;
  realizedPnL += pnl;
  notifyPnLUpdate();

  log(
    symbol,
    `Exited option ${position.optionSymbol} (${fill.filledQty} @ ${exitPremium.toFixed(
      2,
    )}) reason ${reason}, PnL ${pnl.toFixed(2)}`,
  );
  logTradeEvent({
    type: 'exit',
    instrument: 'option',
    symbol,
    reason,
    optionSymbol: position.optionSymbol,
    strike: position.optionStrike,
    expiration: position.optionExpiration,
    qty: fill.filledQty,
    exitPrice: exitPremium,
    entryPrice: entryPremium,
    exitTime: nowIso(),
    entryTime: position.entryTime,
    pnl,
    fastSma,
    realizedPnLTally: realizedPnL,
  });

  if (fill.filledQty >= position.qty) {
    positions.delete(symbol);
  } else {
    position.qty -= fill.filledQty;
    if (position.qty <= 0) {
      positions.delete(symbol);
    } else {
      positions.set(symbol, position);
      log(symbol, `Partial option exit completed; ${position.qty} contracts remain open.`);
    }
  }
}

async function evaluateSymbol(symbol: string, flattenNow: boolean) {
  const state = getSymbolState(symbol);

  if (!flattenNow && !isMarketOpen()) {
    return;
  }

  const requiredBars = Math.max(CONFIG.fastPeriod, CONFIG.slowPeriod ?? CONFIG.fastPeriod);
  // Squeeze needs 40+ bars (20 Bollinger/Keltner + 20 momentum), request more than minimum
  const fetchCount = Math.max(requiredBars + 2, 50);

  let bars: PriceBar[];
  try {
    bars = await loadRecentBars(symbol, fetchCount);
  } catch (err) {
    log(symbol, `Failed to load bars: ${(err as Error).message}`);
    return;
  }

  if (!bars || bars.length < requiredBars + 1) {
    log(
      symbol,
      `Insufficient bars (${bars?.length ?? 0}) for SMA calculation (need ${requiredBars + 1})`,
    );
    return;
  }

  let snapshot: RealtimePriceSnapshot | undefined;
  const snapshotRecencyMs = Math.max(CONFIG.pollIntervalMs * 2, 30_000);
  if (twelveDataFeed) {
    const latestSnapshot = twelveDataFeed.getSnapshot(symbol);
    if (latestSnapshot && Date.now() - latestSnapshot.timestamp <= snapshotRecencyMs) {
      bars = updateBarsWithSnapshot(bars, latestSnapshot, fetchCount);
      snapshot = latestSnapshot;
    }
  }

  const closes = bars.map(bar => bar.c);
  const lastIdx = closes.length - 1;
  const prevIdx = closes.length - 2;

  const barClose = closes[lastIdx];

  let currentPrice = barClose;
  let priceSource: 'bar' | 'snapshot' = 'bar';
  if (snapshot) {
    currentPrice = snapshot.price;
    priceSource = 'snapshot';
  }

  closes[lastIdx] = currentPrice;

  const updatedLastBar = { ...bars[lastIdx] };
  const high = updatedLastBar.h ?? currentPrice;
  const low = updatedLastBar.l ?? currentPrice;
  updatedLastBar.c = currentPrice;
  updatedLastBar.h = Math.max(high, currentPrice);
  updatedLastBar.l = Math.min(low, currentPrice);
  if (priceSource === 'snapshot') {
    updatedLastBar.partial = true;
  }
  bars[lastIdx] = updatedLastBar;

  const fastSeries = calculateSmaSeries(closes, CONFIG.fastPeriod);
  const fastCurr = fastSeries[lastIdx];
  const fastPrev = fastSeries[prevIdx];

  if (fastCurr == null || fastPrev == null) {
    log(symbol, `Waiting for enough bars to compute SMA(${CONFIG.fastPeriod})`);
    return;
  }

  const squeezeSeries = calculateSqueezeSeries(bars);
  const squeezeCurr = squeezeSeries[lastIdx];

  // Debug: Check why squeeze is N/A
  if (!squeezeCurr && bars.length >= 40) {
    const testBar = bars[0];
    log(symbol, `DEBUG: squeeze N/A with ${bars.length} bars. Sample bar has h=${testBar.h}, l=${testBar.l}`);
  }

  const priorPrice = closes[prevIdx];

  const currentRelation: DirectionRelation = currentPrice > fastCurr ? 'above' : 'below';
  const priorRelation: DirectionRelation =
    state.lastRelation ?? (priorPrice > fastPrev ? 'above' : 'below');

  const lastBarTime = bars[lastIdx].t;
  state.lastRelation = currentRelation;
  state.lastBarTime = lastBarTime;

  const partialText = bars[lastIdx]?.partial ? ' (partial)' : '';
  const priceSourceText = priceSource === 'snapshot' ? 'snapshot' : 'bar';
  const squeezeStatus = squeezeCurr
    ? `${squeezeCurr.isOn ? 'ON' : 'OFF'} | direction ${currentPrice >= squeezeCurr.bbMidpoint ? 'bullish' : 'bearish'} (price ${currentPrice.toFixed(2)} vs BB mid ${squeezeCurr.bbMidpoint.toFixed(2)})`
    : 'N/A';

  log(
    symbol,
    `Bar ${lastBarTime}${partialText} close ${barClose.toFixed(2)} | SMA(${CONFIG.fastPeriod}) ${fastCurr.toFixed(
      2,
    )} | latest ${currentPrice.toFixed(2)} (${priceSourceText}) | squeeze ${squeezeStatus}`,
  );

  if (flattenNow && positions.has(symbol)) {
    await closeLongPosition(symbol, 'flatten', currentPrice, fastCurr);
    return;
  }

  if (flattenNow) {
    return;
  }

  let activePosition = positions.get(symbol);
  let exitedDueToStop = false;

  if (activePosition) {
    const drawdown = (currentPrice - activePosition.entryPrice) / activePosition.entryPrice;
    if (drawdown <= -STRATEGY.stopLossPercent) {
      await closeLongPosition(symbol, 'stop_loss', currentPrice, fastCurr);
      exitedDueToStop = true;
    }
    if (!exitedDueToStop) {
      const crossedUp = priorRelation === 'below' && currentRelation === 'above';
      const crossedDown = priorRelation === 'above' && currentRelation === 'below';
      const isPutPosition = activePosition.optionType === 'put';
      const isCallOrStock = !isPutPosition;
      const shouldExitOnCross =
        (isPutPosition && crossedUp) || (isCallOrStock && crossedDown);
      if (shouldExitOnCross) {
        const crossReason: 'bearish_cross' | 'bullish_cross' = isPutPosition ? 'bullish_cross' : 'bearish_cross';
        await closeLongPosition(symbol, crossReason, currentPrice, fastCurr);
      }
    }

    activePosition = positions.get(symbol);
    if (activePosition) {
      if (exitedDueToStop) {
        return;
      }
      const unrealized = (currentPrice - activePosition.entryPrice) * activePosition.qty;
      log(
        symbol,
        `Holding ${activePosition.qty} @ ${activePosition.entryPrice.toFixed(
          2,
        )} | Last ${currentPrice.toFixed(2)} | Unrealized ${unrealized.toFixed(2)}`,
      );
      return;
    }
    if (exitedDueToStop) {
      return;
    }
  }

  // Bullish cross - enter CALL position
  if (priorRelation === 'below' && currentRelation === 'above') {
    if (CONFIG.useSqueezeFilter) {
      if (!squeezeCurr || squeezeCurr.isOn) {
        log(symbol, 'Bullish cross detected but squeeze not expanded (OFF); skipping entry.');
        return;
      }
      const squeezeDirection = currentPrice >= squeezeCurr.bbMidpoint ? 'bullish' : 'bearish';
      if (squeezeDirection !== 'bullish') {
        log(symbol, `Bullish cross detected but squeeze direction is ${squeezeDirection} (price ${currentPrice.toFixed(2)} vs BB mid ${squeezeCurr.bbMidpoint.toFixed(2)}); skipping entry.`);
        return;
      }
    }
    await openLongPosition(symbol, fastCurr, fastCurr, squeezeCurr?.isOn ?? false, 'call');
  }

  // Bearish cross - enter PUT position
  if (priorRelation === 'above' && currentRelation === 'below') {
    if (CONFIG.useSqueezeFilter) {
      if (!squeezeCurr || squeezeCurr.isOn) {
        log(symbol, 'Bearish cross detected but squeeze not expanded (OFF); skipping entry.');
        return;
      }
      const squeezeDirection = currentPrice >= squeezeCurr.bbMidpoint ? 'bullish' : 'bearish';
      if (squeezeDirection !== 'bearish') {
        log(symbol, `Bearish cross detected but squeeze direction is ${squeezeDirection} (price ${currentPrice.toFixed(2)} vs BB mid ${squeezeCurr.bbMidpoint.toFixed(2)}); skipping entry.`);
        return;
      }
    }
    await openLongPosition(symbol, fastCurr, fastCurr, squeezeCurr?.isOn ?? false, 'put');
  }
}

async function shutdown(reason: 'SIGINT' | 'SIGTERM' | 'external', exitProcess = true) {
  if (shuttingDown) return;
  shuttingDown = true;
  log('system', `Shutting down (${reason}); attempting to flatten open positions...`);
  detachFeedListeners();
  if (twelveDataFeed && twelveDataFeedOwned) {
    twelveDataFeed.stop();
  }
  for (const symbol of CONFIG.symbols) {
    if (positions.has(symbol)) {
      try {
        await closeLongPosition(symbol, 'shutdown', positions.get(symbol)!.entryPrice, null);
      } catch (err) {
        log(symbol, `Failed to flatten during shutdown: ${(err as Error).message}`);
      }
    }
  }
  log('system', `Shutdown complete. Realized PnL: ${realizedPnL.toFixed(2)}`);
  if (exitProcess && shouldExitProcessOnShutdown) {
    process.exit(0);
  }
}

export function startSmaCrossoverStrategy(options: SmaStrategyOptions = {}): RunningStrategy {
  strategyHooks = options.hooks;
  shouldExitProcessOnShutdown = options.manageProcessSignals !== false;
  realizedPnL = 0;
  shuttingDown = false;
  notifyPnLUpdate();
  positions.clear();
  symbolState.clear();
  lastClock = null;
  initializeTwelveDataFeed(options.feed ?? null);
  if (options.manageProcessSignals !== false) {
    registerProcessSignalHandlers();
  }
  const task = main();
  return {
    task,
    shutdown: (reason: 'SIGINT' | 'SIGTERM' | 'external' = 'external') =>
      shutdown(reason, false),
  };
}

async function main() {
  console.log('Starting SMA crossover live trader with config:', CONFIG);
  if (twelveDataFeed) {
    if (twelveDataFeedOwned) {
      try {
        log('system', 'Bootstrapping Twelve Data price feed...');
        await twelveDataFeed.bootstrap();
      } catch (err) {
        log('system', `Twelve Data bootstrap failed: ${(err as Error).message}`);
      }
      twelveDataFeed.start();
      log('system', 'Streaming prices via Twelve Data WebSocket feed');
    } else {
      log('system', 'Using shared Twelve Data price feed instance');
    }
    removePriceListener = twelveDataFeed.onPrice(symbol => {
      if (!CONFIG.symbols.includes(symbol)) {
        return;
      }
      scheduleSymbolEvaluation(symbol);
    });
    removeConnectionListener = twelveDataFeed.onConnection((status, info) => {
      if (status === 'open') {
        log('system', '[twelve-data] connection open');
      } else if (status === 'close') {
        log('system', '[twelve-data] connection closed');
      } else if (status === 'error') {
        log('system', `[twelve-data] connection error: ${info}`);
      }
    });
  } else if (USE_TWELVE_DATA_FEED) {
    log('system', 'Twelve Data feed requested but not available; falling back to Alpaca bars');
  }
  await syncExistingPositions();
  for (const symbol of CONFIG.symbols) {
    scheduleSymbolEvaluation(symbol);
  }

  while (!shuttingDown) {
    let clock;
    try {
      clock = await getClock();
    } catch (err) {
      log('system', `Failed to fetch clock: ${(err as Error).message}`);
      await sleep(5000);
      continue;
    }

    lastClock = clock;
    const now = new Date(clock.timestamp);
    if (!clock.is_open) {
      log('system', `Market closed. Next open: ${clock.next_open}. Sleeping...`);
      await sleep(60000);
      continue;
    }

    const flattenNow = shouldFlattenForClose(now);
    for (const symbol of CONFIG.symbols) {
      try {
        scheduleSymbolEvaluation(symbol, { forceFlatten: flattenNow });
      } catch (err) {
        log(symbol, `Error scheduling evaluation: ${(err as Error).message}`);
      }
    }

    await sleep(CONFIG.pollIntervalMs);
  }
}

let invokedAsScript = false;
if (typeof process !== 'undefined' && Array.isArray(process.argv) && process.argv[1]) {
  try {
    invokedAsScript = import.meta.url === pathToFileURL(process.argv[1]).href;
  } catch {
    invokedAsScript = false;
  }
}

if (invokedAsScript) {
  const runner = startSmaCrossoverStrategy();
  runner.task.catch(err => {
    console.error('Fatal error in SMA crossover runner:', err);
    process.exit(1);
  });
}
