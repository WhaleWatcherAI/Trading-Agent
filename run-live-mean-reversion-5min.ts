import 'dotenv/config';
import { randomUUID } from 'crypto';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';
import {
  submitOptionOrder,
  getOptionOrder,
  cancelOptionOrder,
  submitOrder,
  getOrder,
  cancelOrder,
  getClock,
  sleep,
  AlpacaOptionOrder,
  AlpacaOptionContract,
} from './lib/alpaca';
import { fetchTradierOptionExpirations, fetchTradierOptionChain } from './lib/tradier';
import {
  getCachedExpirations,
  setCachedExpirations,
  getCachedOptionChain,
  setCachedOptionChain,
  isCacheEntryFresh,
  DEFAULT_CACHE_MAX_AGE_MS,
} from './lib/optionCache';
import {
  generateMeanReversionSignalFromTechnicals,
  MeanReversionSignal,
} from './lib/meanReversionAgent';
import {
  TwelveDataPriceFeed,
  RealtimePriceSnapshot,
  MinuteBar,
} from './lib/twelveData';
import { StrategyHooks, RunningStrategy } from './lib/strategyHooks';

process.env.BYPASS_GEX = 'true';

type Direction = 'long' | 'short';

interface TradeFill {
  type: 'entry' | 'scale' | 'exit';
  qty: number;
  price: number;
  timestamp: string;
  realizedPnL?: number;
}

interface ActiveTrade {
  symbol: string;
  instrument: 'option' | 'stock';
  positionSide: 'long' | 'short';
  direction: Direction;
  signal: MeanReversionSignal;
  entryPrice: number;
  entryTime: string;
  stopLoss: number | null;
  target: number | null;
  scaled: boolean;
  totalQty: number;
  entryFillPrice: number;
  multiplier: number;
  optionSymbol?: string;
  optionType?: 'call' | 'put';
  optionStrike?: number;
  optionExpiration?: string;
  entryQty: number;
  costBasis: number;
  realizedPnL: number;
  fills: TradeFill[];
  remainingQty: number;
  lastFiveMinuteBar?: string | null;
  processing?: boolean;
}

interface OrderFillResult {
  filledQty: number;
  avgPrice: number;
  status: AlpacaOptionOrder['status'];
}

interface MeanReversionRunnerOptions {
  feed?: TwelveDataPriceFeed | null;
  hooks?: StrategyHooks;
  manageProcessSignals?: boolean;
}

const FIVE_MIN_MS = 5 * 60_000;
const STOP_LOSS_PERCENT = 0.001; // 0.1%

const CONFIG = {
  symbols: (process.env.MR5_SYMBOLS || 'SPY,GLD,TSLA,NVDA')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean),
  optionContracts: Number(process.env.MR5_OPTION_CONTRACTS || '2'),
  stockShares: Number(process.env.MR5_STOCK_SHARES || '0'),
  pollIntervalMs: Number(process.env.MR5_POLL_MS || '15000'),
  minuteBackfill: Number(process.env.MR5_MINUTE_BACKFILL || '600'),
  twelveDataApiKey: process.env.TWELVE_DATA_API_KEY || '',
  twelveDataBackupApiKey: process.env.TWELVE_DATA_BACKUP_API_KEY || '',
  twelveDataUrl: process.env.TWELVE_DATA_WS_URL || '',
};

const TRADE_LOG_FILE =
  process.env.MR5_TRADE_LOG || './logs/mean-reversion-5min-trades.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);

const trades = new Map<string, ActiveTrade>();
const lastProcessed5mBar = new Map<string, string>();
const optionDataCooldownUntil = new Map<string, number>();

const OPTION_DATA_RETRY_MS = 5 * 60 * 1000;
const OPTION_CACHE_MAX_AGE_MS =
  Number(process.env.ALPACA_OPTION_CACHE_MAX_AGE_MS || '') || DEFAULT_CACHE_MAX_AGE_MS;
const OPTION_EXPIRATION_EOD_SUFFIX = 'T21:00:00Z';
const TRADE_MODE = (process.env.MR5_TRADE_MODE || 'option').toLowerCase();
const USE_STOCK_TRADING = TRADE_MODE === 'stock';
const STOCK_SHARE_DEFAULT = CONFIG.optionContracts > 0 ? CONFIG.optionContracts * 100 : 100;
const STOCK_SHARE_QTY =
  Number.isFinite(CONFIG.stockShares) && CONFIG.stockShares > 0
    ? CONFIG.stockShares
    : STOCK_SHARE_DEFAULT;

export const MEAN_REVERSION_5M_STRATEGY_ID = 'mean-reversion-5m';

let twelveDataFeed: TwelveDataPriceFeed | null = null;
let twelveDataFeedOwned = false;
let removePriceListener: (() => void) | null = null;
let processSignalsRegistered = false;
let shouldExitProcessOnShutdown = true;
let shuttingDown = false;
let strategyHooks: StrategyHooks | undefined;
let realizedPnLTotal = 0;

function notifyPnLUpdate() {
  strategyHooks?.onPnLUpdate?.(MEAN_REVERSION_5M_STRATEGY_ID, realizedPnLTotal);
}

function recordRealizedPnL(delta: number) {
  if (!Number.isFinite(delta) || delta === 0) {
    return;
  }
  realizedPnLTotal += delta;
  notifyPnLUpdate();
}

function initializeTwelveDataFeed(sharedFeed?: TwelveDataPriceFeed | null) {
  if (sharedFeed) {
    twelveDataFeed = sharedFeed;
    twelveDataFeedOwned = false;
    return;
  }
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
    maxMinuteBars: CONFIG.minuteBackfill,
  });
  twelveDataFeedOwned = true;
}

function detachFeedListeners() {
  if (removePriceListener) {
    try {
      removePriceListener();
    } catch {
      /* noop */
    }
    removePriceListener = null;
  }
}

function registerProcessSignalHandlers() {
  if (processSignalsRegistered) {
    return;
  }
  const handle = (reason: 'SIGINT' | 'SIGTERM') => {
    shutdown(reason).catch(err => {
      console.error('[mean-reversion-5m] Failed to shutdown cleanly:', err);
      process.exit(1);
    });
  };
  process.once('SIGINT', () => handle('SIGINT'));
  process.once('SIGTERM', () => handle('SIGTERM'));
  processSignalsRegistered = true;
}

function nowIso() {
  return new Date().toISOString();
}

function ensureTradeLogDir() {
  if (!existsSync(TRADE_LOG_DIR)) {
    mkdirSync(TRADE_LOG_DIR, { recursive: true });
  }
}

function logTradeEvent(event: Record<string, any>) {
  try {
    ensureTradeLogDir();
    appendFileSync(
      TRADE_LOG_FILE,
      `${JSON.stringify({ timestamp: nowIso(), ...event })}\n`,
    );
  } catch (err) {
    console.error('[trade-log] failed to write entry', err);
  }
}

function easternTimeParts(date: Date): { hour: number; minute: number } {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
  });
  const parts = formatter.format(date).split(':');
  return { hour: Number(parts[0] || 0), minute: Number(parts[1] || 0) };
}

function shouldFlattenForClose(date: Date): boolean {
  const { hour, minute } = easternTimeParts(date);
  if (hour > 15) return true;
  if (hour === 15 && minute >= 55) return true;
  return false;
}

function log(symbol: string, message: string) {
  console.log(`[${new Date().toISOString()}][${symbol}] ${message}`);
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
      log('system', `Cancel failed for option order ${orderId}: ${(err as Error).message}`);
    }
    const order = await getOptionOrder(orderId);
    return {
      filledQty: Number(order.filled_qty || '0'),
      avgPrice: Number(order.filled_avg_price || '0'),
      status: order.status,
    };
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
    return waitForOptionFill(order.id, 15_000);
  } catch (err) {
    log(contractSymbol, `Option market order failed (${side} ${quantity}): ${(err as Error).message}`);
    return null;
  }
}

async function waitForStockFill(
  orderId: string,
  timeoutMs: number,
  cancelOnTimeout = true,
): Promise<OrderFillResult> {
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
      order.status === 'suspended' ||
      order.status === 'expired'
    ) {
      return { filledQty, avgPrice, status: order.status };
    }
    await sleep(1000);
  }

  if (cancelOnTimeout) {
    try {
      await cancelOrder(orderId);
    } catch (err) {
      log('system', `Cancel failed for stock order ${orderId}: ${(err as Error).message}`);
    }
    const order = await getOrder(orderId);
    return {
      filledQty: Number(order.filled_qty || '0'),
      avgPrice: Number(order.filled_avg_price || '0'),
      status: order.status,
    };
  }

  const order = await getOrder(orderId);
  return {
    filledQty: Number(order.filled_qty || '0'),
    avgPrice: Number(order.filled_avg_price || '0'),
    status: order.status,
  };
}

async function submitStockMarketOrder(
  symbol: string,
  side: 'buy' | 'sell',
  quantity: number,
): Promise<OrderFillResult | null> {
  if (quantity <= 0) {
    return null;
  }
  try {
    const order = await submitOrder({
      symbol,
      side,
      qty: String(quantity),
      type: 'market',
      time_in_force: 'day',
      client_order_id: randomUUID(),
    });
    return waitForStockFill(order.id, 15_000);
  } catch (err) {
    log(symbol, `Stock market order failed (${side} ${quantity}): ${(err as Error).message}`);
    return null;
  }
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

export function calculatePnL(trade: ActiveTrade, exitPrice: number, qty: number): number {
  const priceDiff = exitPrice - trade.entryFillPrice;
  const signedDiff = trade.positionSide === 'short' ? -priceDiff : priceDiff;
  return signedDiff * trade.multiplier * qty;
}

function setRunnerLevelsFromSignal(trade: ActiveTrade, signal: MeanReversionSignal) {
  const middleBand = signal.bbMiddle;
  if (typeof middleBand === 'number') {
    const multiplier = trade.direction === 'long' ? 1 - STOP_LOSS_PERCENT : 1 + STOP_LOSS_PERCENT;
    trade.stopLoss = middleBand * multiplier;
  }

  const outerBand =
    trade.direction === 'long'
      ? signal.bbUpper ?? signal.bbMiddle ?? null
      : signal.bbLower ?? signal.bbMiddle ?? null;

  if (typeof outerBand === 'number') {
    trade.target = outerBand;
  }
}


async function attemptEntry(symbol: string, signal: MeanReversionSignal, referencePrice: number) {
  const direction: Direction = signal.direction === 'long' ? 'long' : 'short';

  if (USE_STOCK_TRADING) {
    const shares = STOCK_SHARE_QTY;
    if (shares <= 0) {
      log(symbol, 'Stock share quantity is zero; skipping entry');
      return;
    }

    const entrySide = direction === 'long' ? 'buy' : 'sell';
    log(
      symbol,
      `Attempting ${direction.toUpperCase()} stock entry @ ${referencePrice.toFixed(
        2,
      )} for ${shares} shares`,
    );

    const fill = await submitStockMarketOrder(symbol, entrySide, shares);
    if (!fill || fill.filledQty === 0) {
      log(symbol, 'Stock entry order not filled');
      return;
    }

    const entryFillPrice = fill.avgPrice;
    const trade: ActiveTrade = {
      symbol,
      instrument: 'stock',
      positionSide: direction === 'long' ? 'long' : 'short',
      direction,
      signal,
      entryPrice: referencePrice,
      entryTime: nowIso(),
      stopLoss: signal.stopLoss,
      target: signal.target,
      scaled: false,
      totalQty: fill.filledQty,
      entryFillPrice,
      multiplier: 1,
      entryQty: fill.filledQty,
      costBasis: entryFillPrice * fill.filledQty,
      realizedPnL: 0,
      fills: [
        {
          type: 'entry',
          qty: fill.filledQty,
          price: entryFillPrice,
          timestamp: nowIso(),
        },
      ],
      remainingQty: fill.filledQty,
    };

    trades.set(symbol, trade);
    log(
      symbol,
      `Entered ${direction.toUpperCase()} stock position ${symbol} @ ${entryFillPrice.toFixed(
        2,
      )} (${fill.filledQty} shares)`,
    );

    logTradeEvent({
      type: 'entry',
      instrument: 'stock',
      symbol,
      direction: direction.toUpperCase(),
      qty: fill.filledQty,
      entryPrice: entryFillPrice,
      referencePrice,
      rationale: trade.signal.rationale,
    });

    return;
  }

  const contracts = CONFIG.optionContracts;
  if (contracts <= 0) {
    return;
  }

  const optionType: 'call' | 'put' = direction === 'long' ? 'call' : 'put';

  const contract = await selectOptionContract(symbol, optionType, referencePrice, 7);
  if (!contract) {
    log(symbol, `No suitable ${optionType.toUpperCase()} contract found for entry`);
    return;
  }

  log(
    symbol,
    `Attempting ${direction.toUpperCase()} option entry @ ${referencePrice.toFixed(
      2,
    )} using ${contract.symbol} (${contracts} contracts)`,
  );

  const fill = await submitOptionMarketOrder(contract.symbol, 'buy', contracts, 'open');
  if (!fill || fill.filledQty === 0) {
    log(symbol, 'Option entry order not filled');
    return;
  }

  const entryPremium = fill.avgPrice;
  const trade: ActiveTrade = {
    symbol,
    instrument: 'option',
    positionSide: 'long',
    direction,
    signal,
    entryPrice: referencePrice,
    entryTime: nowIso(),
    stopLoss: signal.stopLoss,
    target: signal.target,
    scaled: false,
    totalQty: fill.filledQty,
    entryFillPrice: entryPremium,
    multiplier: 100,
    optionSymbol: contract.symbol,
    optionType,
    optionStrike: contract.strike_price,
    optionExpiration: contract.expiration_date,
    entryQty: fill.filledQty,
    costBasis: entryPremium * fill.filledQty * 100,
    realizedPnL: 0,
    fills: [
      {
        type: 'entry',
        qty: fill.filledQty,
        price: entryPremium,
        timestamp: nowIso(),
      },
    ],
    remainingQty: fill.filledQty,
  };

  trades.set(symbol, trade);
  log(
    symbol,
    `Entered ${direction.toUpperCase()} option position ${contract.symbol} @ ${fill.avgPrice.toFixed(
      2,
    )} (${fill.filledQty} contracts)`,
  );

  logTradeEvent({
    type: 'entry',
    instrument: 'option',
    symbol,
    direction: direction.toUpperCase(),
    optionSymbol: contract.symbol,
    strike: contract.strike_price,
    expiration: contract.expiration_date,
    qty: fill.filledQty,
    entryPrice: entryPremium,
    referencePrice,
    rationale: trade.signal.rationale,
  });
}

async function scaleTrade(trade: ActiveTrade) {
  if (trade.scaled || trade.totalQty < 2 || trade.remainingQty <= 1) {
    return;
  }

  let scaleQty = Math.max(1, Math.floor(trade.totalQty / 2));
  if (scaleQty >= trade.remainingQty) {
    scaleQty = Math.max(1, trade.remainingQty - 1);
  }
  if (scaleQty <= 0) {
    return;
  }

  if (trade.instrument === 'stock') {
    const side = trade.direction === 'long' ? 'sell' : 'buy';
    const action = side === 'sell' ? 'selling' : 'buying to cover';
    log(
      trade.symbol,
      `Scaling stock position ${trade.symbol}: ${action} ${scaleQty} shares at initial target`,
    );

    const fill = await submitStockMarketOrder(trade.symbol, side, scaleQty);
    if (!fill || fill.filledQty === 0) {
      log(trade.symbol, 'Scale order not filled; retaining full position');
      return;
    }

    const pnl = calculatePnL(trade, fill.avgPrice, fill.filledQty);
    trade.realizedPnL += pnl;
    recordRealizedPnL(pnl);
    recordRealizedPnL(pnl);
    trade.costBasis = Math.max(
      0,
      trade.costBasis - trade.entryFillPrice * fill.filledQty * trade.multiplier,
    );
    trade.remainingQty = Math.max(0, trade.remainingQty - fill.filledQty);
    trade.scaled = true;

    trade.fills.push({
      type: 'scale',
      qty: fill.filledQty,
      price: fill.avgPrice,
      timestamp: nowIso(),
      realizedPnL: pnl,
    });

    setRunnerLevelsFromSignal(trade, trade.signal);

    log(
      trade.symbol,
      `Scaled position; remaining ${trade.remainingQty} shares, new stop ${trade.stopLoss?.toFixed(
        2,
      ) ?? 'n/a'}, new target ${trade.target?.toFixed(2) ?? 'n/a'}`,
    );

    logTradeEvent({
      type: 'scale',
      instrument: 'stock',
      symbol: trade.symbol,
      qty: fill.filledQty,
      scalePrice: fill.avgPrice,
      entryPrice: trade.entryFillPrice,
      realizedPnL: pnl,
      cumulativePnL: trade.realizedPnL,
      remainingQty: trade.remainingQty,
    });

    return;
  }

  log(
    trade.symbol,
    `Scaling option position ${trade.optionSymbol}: selling ${scaleQty} contracts at initial target`,
  );

  const fill = await submitOptionMarketOrder(trade.optionSymbol!, 'sell', scaleQty, 'close');
  if (!fill || fill.filledQty === 0) {
    log(trade.symbol, 'Scale order not filled; retaining full position');
    return;
  }

  const pnl = calculatePnL(trade, fill.avgPrice, fill.filledQty);
  trade.realizedPnL += pnl;
  recordRealizedPnL(pnl);
  recordRealizedPnL(pnl);
  trade.costBasis = Math.max(
    0,
    trade.costBasis - trade.entryFillPrice * fill.filledQty * trade.multiplier,
  );
  trade.remainingQty = Math.max(0, trade.remainingQty - fill.filledQty);
  trade.scaled = true;

  trade.fills.push({
    type: 'scale',
    qty: fill.filledQty,
    price: fill.avgPrice,
    timestamp: nowIso(),
    realizedPnL: pnl,
  });

  setRunnerLevelsFromSignal(trade, trade.signal);

  log(
    trade.symbol,
    `Scaled position; remaining ${trade.remainingQty} contracts, new stop ${trade.stopLoss?.toFixed(
      2,
    ) ?? 'n/a'}, new target ${trade.target?.toFixed(2) ?? 'n/a'}`,
  );

  logTradeEvent({
    type: 'scale',
    instrument: 'option',
    symbol: trade.symbol,
    optionSymbol: trade.optionSymbol,
    qty: fill.filledQty,
    scalePrice: fill.avgPrice,
    entryPrice: trade.entryFillPrice,
    realizedPnL: pnl,
    cumulativePnL: trade.realizedPnL,
    remainingQty: trade.remainingQty,
  });
}

async function exitTrade(trade: ActiveTrade, reason: 'stop' | 'target' | 'end_of_day') {
  if (trade.remainingQty <= 0) {
    trades.delete(trade.symbol);
    return;
  }

  if (trade.instrument === 'stock') {
    const side = trade.direction === 'long' ? 'sell' : 'buy';
    const action = side === 'sell' ? 'selling' : 'buying to cover';
    log(
      trade.symbol,
      `Closing stock position ${trade.symbol} (${reason}) for ${trade.remainingQty} shares`,
    );

    const fill = await submitStockMarketOrder(trade.symbol, side, trade.remainingQty);
    if (!fill || fill.filledQty === 0) {
      log(trade.symbol, 'Exit order failed; leaving trade open');
      return;
    }

    const pnl = calculatePnL(trade, fill.avgPrice, fill.filledQty);
    trade.realizedPnL += pnl;
    trade.costBasis = Math.max(
      0,
      trade.costBasis - trade.entryFillPrice * fill.filledQty * trade.multiplier,
    );
    trade.remainingQty = Math.max(0, trade.remainingQty - fill.filledQty);

    trade.fills.push({
      type: 'exit',
      qty: fill.filledQty,
      price: fill.avgPrice,
      timestamp: nowIso(),
      realizedPnL: pnl,
    });

    const exitTimestamp = nowIso();
    const durationMinutes =
      (new Date(exitTimestamp).getTime() - new Date(trade.entryTime).getTime()) / 60000;

    log(
      trade.symbol,
      `Exit filled (${reason}) avg ${fill.avgPrice.toFixed(2)} for ${fill.filledQty} shares`,
    );

    logTradeEvent({
      type: 'close',
      instrument: 'stock',
      symbol: trade.symbol,
      direction: trade.direction.toUpperCase(),
      entryPrice: trade.entryFillPrice,
      exitPrice: fill.avgPrice,
      totalQty: trade.totalQty,
      realizedPnL: trade.realizedPnL,
      durationMinutes: Number(durationMinutes.toFixed(2)),
      exitReason: reason,
      fills: trade.fills,
      rationale: trade.signal.rationale,
    });

    trades.delete(trade.symbol);
    return;
  }

  log(
    trade.symbol,
    `Closing option position ${trade.optionSymbol} (${reason}) for ${trade.remainingQty} contracts`,
  );

  const fill = await submitOptionMarketOrder(
    trade.optionSymbol!,
    'sell',
    trade.remainingQty,
    'close',
  );

  if (!fill || fill.filledQty === 0) {
    log(trade.symbol, 'Exit order failed; leaving trade open');
    return;
  }

  const pnl = calculatePnL(trade, fill.avgPrice, fill.filledQty);
  trade.realizedPnL += pnl;
  trade.costBasis = Math.max(
    0,
    trade.costBasis - trade.entryFillPrice * fill.filledQty * trade.multiplier,
  );
  trade.remainingQty = Math.max(0, trade.remainingQty - fill.filledQty);

  trade.fills.push({
    type: 'exit',
    qty: fill.filledQty,
    price: fill.avgPrice,
    timestamp: nowIso(),
    realizedPnL: pnl,
  });

  const exitTimestamp = nowIso();
  const durationMinutes =
    (new Date(exitTimestamp).getTime() - new Date(trade.entryTime).getTime()) / 60000;

  log(
    trade.symbol,
    `Exit filled (${reason}) avg ${fill.avgPrice.toFixed(2)} for ${fill.filledQty} contracts`,
  );

  logTradeEvent({
    type: 'close',
    instrument: 'option',
    symbol: trade.symbol,
    direction: trade.direction.toUpperCase(),
    optionSymbol: trade.optionSymbol,
    strike: trade.optionStrike,
    expiration: trade.optionExpiration,
    entryPrice: trade.entryFillPrice,
    exitPrice: fill.avgPrice,
    totalQty: trade.totalQty,
    realizedPnL: trade.realizedPnL,
    durationMinutes: Number(durationMinutes.toFixed(2)),
    exitReason: reason,
    fills: trade.fills,
    rationale: trade.signal.rationale,
  });

  trades.delete(trade.symbol);
}

async function handleRealtimeTick(symbol: string, snapshot: RealtimePriceSnapshot) {
  const trade = trades.get(symbol);
  if (!trade) {
    log(symbol, `Realtime check price ${snapshot.price.toFixed(2)} stop n/a target n/a`);
    return;
  }
  if (trade.processing) {
    return;
  }

  trade.processing = true;
  try {
    const price = snapshot.price;
    const barTime = new Date(snapshot.timestamp);

    if (shouldFlattenForClose(barTime)) {
      await exitTrade(trade, 'end_of_day');
      return;
    }

    log(
      trade.symbol,
      `Realtime check price ${price.toFixed(2)} stop ${trade.stopLoss?.toFixed(2) ?? 'n/a'} target ${trade.target?.toFixed(2) ?? 'n/a'}`
    );

    const minuteMid = trade.signal.bbMiddle;
    if (
      !trade.scaled &&
      typeof minuteMid === 'number' &&
      typeof trade.target === 'number' &&
      minuteMid !== trade.target
    ) {
      trade.target = minuteMid;
    }

    if (!trade.scaled && typeof trade.target === 'number') {
      const hitTarget =
        (trade.direction === 'long' && price >= trade.target) ||
        (trade.direction === 'short' && price <= trade.target);
      if (hitTarget) {
        await scaleTrade(trade);
        return;
      }
    }

    if (trade.stopLoss !== null) {
      const hitStop =
        (trade.direction === 'long' && price <= trade.stopLoss) ||
        (trade.direction === 'short' && price >= trade.stopLoss);
      if (hitStop) {
        await exitTrade(trade, 'stop');
        return;
      }
    }

    if (typeof trade.target === 'number') {
      const finalTarget =
        (trade.direction === 'long' && price >= trade.target) ||
        (trade.direction === 'short' && price <= trade.target);
      if (finalTarget) {
        await exitTrade(trade, 'target');
      }
    }
  } catch (err) {
    log(symbol, `Realtime handler error: ${(err as Error).message}`);
  } finally {
    const stillActive = trades.get(symbol);
    if (stillActive) {
      stillActive.processing = false;
    }
  }
}

function buildFiveMinuteBars(minuteBars: MinuteBar[], count: number): MinuteBar[] {
  const buckets = new Map<number, {
    startMs: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>();

  for (const bar of minuteBars) {
    const barTime = Date.parse(bar.t);
    if (!Number.isFinite(barTime)) {
      continue;
    }
    const intervalStart = Math.floor(barTime / FIVE_MIN_MS) * FIVE_MIN_MS;
    const bucket = buckets.get(intervalStart);
    const close = bar.c;
    const high = bar.h;
    const low = bar.l;
    const volume = bar.v ?? 0;
    if (!bucket) {
      buckets.set(intervalStart, {
        startMs: intervalStart,
        open: bar.o,
        high,
        low,
        close,
        volume,
      });
    } else {
      bucket.high = Math.max(bucket.high, high);
      bucket.low = Math.min(bucket.low, low);
      bucket.close = close;
      bucket.volume += volume;
    }
  }

  const aggregated = Array.from(buckets.values()).sort(
    (a, b) => a.startMs - b.startMs,
  );

  return aggregated.slice(-count).map(bucket => ({
    t: new Date(bucket.startMs).toISOString(),
    o: bucket.open,
    h: bucket.high,
    l: bucket.low,
    c: bucket.close,
    v: bucket.volume,
  }));
}

async function processActiveTrade(trade: ActiveTrade) {
  if (trade.processing) {
    return;
  }

  if (!twelveDataFeed) {
    log(trade.symbol, 'Twelve Data feed unavailable; cannot monitor active trade');
    return;
  }

  trade.processing = true;

  try {
    let minuteBars = twelveDataFeed.getRecentBars(trade.symbol, '1Min', 5);

    if (minuteBars.length === 0) {
      await twelveDataFeed.syncSymbol(trade.symbol, true);
      minuteBars = twelveDataFeed.getRecentBars(trade.symbol, '1Min', 5);
    }

    if (minuteBars.length === 0) {
      log(trade.symbol, 'No minute bars available from Twelve Data; skipping trade processing');
      return;
    }

    const latest = minuteBars[minuteBars.length - 1];
    if (trade.lastFiveMinuteBar === latest.t) {
      return;
    }
    trade.lastFiveMinuteBar = latest.t;

    const high = latest.h;
    const low = latest.l;
    const barTime = new Date(latest.t);

    if (shouldFlattenForClose(barTime)) {
      await exitTrade(trade, 'end_of_day');
      return;
    }

    if (!trade.scaled && typeof trade.target === 'number') {
      const hitTarget =
        (trade.direction === 'long' && high >= trade.target) ||
        (trade.direction === 'short' && low <= trade.target);

      if (hitTarget) {
        await scaleTrade(trade);
        return;
      }
    }

    if (trade.stopLoss !== null) {
      const hitStop =
        (trade.direction === 'long' && low <= trade.stopLoss) ||
        (trade.direction === 'short' && high >= trade.stopLoss);
      if (hitStop) {
        await exitTrade(trade, 'stop');
        return;
      }
    }
  } finally {
    const stillActive = trades.get(trade.symbol);
    if (stillActive) {
      stillActive.processing = false;
    }
  }
}

async function processSymbol(symbol: string) {
  if (!twelveDataFeed) {
    log(symbol, 'Twelve Data feed unavailable; skipping symbol processing');
    return;
  }

  if (twelveDataFeed.shouldSync(symbol)) {
    await twelveDataFeed.syncSymbol(symbol);
  }

  const minuteBars = twelveDataFeed.getRecentBars(symbol, '1Min', 200);
  if (minuteBars.length < 20) {
    log(symbol, 'Insufficient minute bars for aggregation');
    return;
  }

  let fiveMinuteBars = buildFiveMinuteBars(minuteBars, 40);
  if (fiveMinuteBars.length < 20) {
    log(symbol, 'Insufficient 5m bars for signal');
    return;
  }

  let latestBar = fiveMinuteBars[fiveMinuteBars.length - 1];
  const latestBarTime = Date.parse(latestBar.t);
  if (Number.isFinite(latestBarTime)) {
    const ageMs = Date.now() - latestBarTime;
    if (ageMs > 7 * 60_000) {
      log(
        symbol,
        `Latest 5m bar stale (${(ageMs / 60000).toFixed(
          1,
        )} min old); forcing Twelve Data resync`,
      );
      await twelveDataFeed.syncSymbol(symbol, true);
      const refreshedMinuteBars = twelveDataFeed.getRecentBars(symbol, '1Min', 200);
      const refreshedFiveMinuteBars = buildFiveMinuteBars(refreshedMinuteBars, 40);
      if (refreshedFiveMinuteBars.length >= 20) {
        fiveMinuteBars = refreshedFiveMinuteBars;
        latestBar = fiveMinuteBars[fiveMinuteBars.length - 1];
      }
    }
  }

  const barId = latestBar.t;
  const prevBar = lastProcessed5mBar.get(symbol);

  if (prevBar === barId) {
    return;
  }

  lastProcessed5mBar.set(symbol, barId);

  const closes = fiveMinuteBars.map(b => Number(b.c));
  const price = Number(latestBar.c);
  const signal = generateMeanReversionSignalFromTechnicals(symbol, price, closes, 1, {
    rsiPeriod: 14,
    rsiOversold: 30,
    rsiOverbought: 70,
    bbPeriod: 20,
    bbStdDev: 2,
    bbThreshold: 0.005,
    stopLossPercent: STOP_LOSS_PERCENT,
  });

  if (signal.direction === 'none') {
    log(symbol, 'No mean reversion signal on latest 5m bar');
    return;
  }

  const activeTrade = trades.get(symbol);
  if (activeTrade) {
    if (activeTrade.scaled) {
      setRunnerLevelsFromSignal(activeTrade, signal);
    } else {
      const updatedStop = signal.stopLoss ?? null;
      const updatedTarget = signal.target ?? null;
      if (typeof updatedStop === 'number') {
        activeTrade.stopLoss = updatedStop;
      }
      if (typeof updatedTarget === 'number') {
        activeTrade.target = updatedTarget;
      }
    }
    activeTrade.signal = signal;
    log(
      symbol,
      `Trade updated from latest 5m bar: stop ${activeTrade.stopLoss?.toFixed(2) ?? 'n/a'}, target ${activeTrade.target?.toFixed(2) ?? 'n/a'}`,
    );
    return;
  }

  await attemptEntry(symbol, signal, price);
}

async function shutdown(reason: 'SIGINT' | 'SIGTERM' | 'external' = 'external', exitProcess = true) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  detachFeedListeners();
  if (twelveDataFeed && twelveDataFeedOwned) {
    try {
      twelveDataFeed.stop();
    } catch {
      /* noop */
    }
  }
  const activeTrades = Array.from(trades.values());
  for (const trade of activeTrades) {
    try {
      await exitTrade(trade, 'end_of_day');
    } catch (err) {
      log(trade.symbol, `Failed to exit during shutdown: ${(err as Error).message}`);
    }
  }
  console.log(
    `[mean-reversion-5m] Shutdown complete (${reason}). Realized PnL: ${realizedPnLTotal.toFixed(
      2,
    )}`,
  );
  if (exitProcess && shouldExitProcessOnShutdown) {
    process.exit(0);
  }
}

async function main() {
  console.log('🚀 Starting 5m mean reversion manager');
  console.log(`Symbols: ${CONFIG.symbols.join(', ')}`);
  if (USE_STOCK_TRADING) {
    console.log(`Stock shares: ${STOCK_SHARE_QTY}`);
  } else {
    console.log(`Option contracts: ${CONFIG.optionContracts}`);
  }

  if (!twelveDataFeed) {
    console.error('TWELVE_DATA_API_KEY not configured; unable to start live strategy without data feed');
    shuttingDown = true;
    return;
  }

  console.log(
    twelveDataFeedOwned
      ? 'Using dedicated Twelve Data WebSocket feed for realtime price ticks'
      : 'Using shared Twelve Data WebSocket feed for realtime price ticks',
  );

  if (twelveDataFeedOwned) {
    await twelveDataFeed.bootstrap();
    twelveDataFeed.start();
  }

  removePriceListener = twelveDataFeed.onPrice((symbol, snapshot) => {
    handleRealtimeTick(symbol, snapshot).catch(err => {
      log(symbol, `Realtime tick error: ${(err as Error).message}`);
    });
  });

  while (!shuttingDown) {
    try {
      const clock = await getClock();
      if (!clock.is_open) {
        console.log(`[${nowIso()}] Market closed. Sleeping until open...`);
        await sleep(CONFIG.pollIntervalMs);
        continue;
      }

      for (const symbol of CONFIG.symbols) {
        try {
          await processSymbol(symbol);
          const trade = trades.get(symbol);
          if (trade) {
            await processActiveTrade(trade);
          }
        } catch (err) {
          log(symbol, `Error in processing: ${(err as Error).message}`);
        }
      }
    } catch (err) {
      console.error(`[${nowIso()}] Fatal loop error`, err);
    }

    await sleep(CONFIG.pollIntervalMs);
  }
}

export function startMeanReversion5mStrategy(
  options: MeanReversionRunnerOptions = {},
): RunningStrategy {
  strategyHooks = options.hooks;
  shouldExitProcessOnShutdown = options.manageProcessSignals !== false;
  shuttingDown = false;
  realizedPnLTotal = 0;
  notifyPnLUpdate();
  trades.clear();
  lastProcessed5mBar.clear();
  optionDataCooldownUntil.clear();
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

let invokedAsScript = false;
if (typeof process !== 'undefined' && Array.isArray(process.argv) && process.argv[1]) {
  try {
    invokedAsScript = import.meta.url === pathToFileURL(process.argv[1]).href;
  } catch {
    invokedAsScript = false;
  }
}

if (invokedAsScript) {
  const runner = startMeanReversion5mStrategy();
  runner.task.catch(err => {
    console.error('5m strategy crashed:', err);
    process.exit(1);
  });
}
