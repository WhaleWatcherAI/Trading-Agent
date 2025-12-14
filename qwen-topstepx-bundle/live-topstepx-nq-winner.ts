#!/usr/bin/env tsx
/**
 * TopstepX Live NQ Winner Strategy - 1-Minute Bars
 *
 * Strategy (from backtest-topstepx-mean-reversion-nq-winner.ts):
 * - Bollinger Bands: 20-period SMA with 3 standard deviations
 * - RSI confirmation: RSI(24) with 30/70 levels (oversold/overbought)
 * - Two-stage entry:
 *   Stage 1: Price touches BB outer band + RSI extreme creates setup
 *   Stage 2: TTM Squeeze ON triggers entry with BRACKET ORDER
 * - Bracket: 0-tick limit IOC for both TP and SL
 * - Stop Limit Monitoring: Convert to market if limit doesn't fill
 * - Target: 0.05% (0.0005) from entry
 * - Stop Loss: 0.01% (0.0001) from entry
 */

import 'dotenv/config';
import { RSI } from 'technicalindicators';
import { calculateTtmSqueeze } from './lib/ttmSqueeze';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  fetchTopstepXAccounts,
  TopstepXFuturesBar,
  authenticate,
} from './lib/topstepx';
import { appendFileSync, existsSync, mkdirSync } from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { createProjectXRest } from './projectx-rest';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';

function getBaseSymbol(fullSymbol: string): string {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
}

interface ActivePosition {
  tradeId: string;
  symbol: string;
  contractId: string;
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  stopLoss: number;
  target: number;
  totalQty: number;
  entryRSI: number;
  entryOrderId: string | number;
  stopOrderId?: string | number;
  targetOrderId?: string | number;
  stopFilled: boolean;
  targetFilled: boolean;
  stopLimitPending: boolean; // Track if stop limit is waiting to be monitored
  monitoringStop: boolean; // True when actively monitoring unfilled stop limit
}

interface PendingSetup {
  side: 'long' | 'short';
  setupTime: string;
  setupPrice: number;
  rsi: number;
  bb: { upper: number; middle: number; lower: number };
}

interface StrategyConfig {
  symbol: string;
  contractId?: string;
  bbPeriod: number;
  bbStdDev: number;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  numberOfContracts: number;
  pollIntervalMs: number;
  initialBackfillBars: number;
}

type OrderSide = 'Buy' | 'Sell';

interface OrderPayload {
  accountId: number;
  contractId: string;
  side: 0 | 1; // 0 = Buy, 1 = Sell
  size: number;
  type: 1 | 2; // 1 = Limit, 2 = Market
  timeInForce: 0; // 0 = IOC
  limitPrice?: number;
  stopPrice?: number;
}

class TopstepOrderManager {
  constructor(
    private rest: ReturnType<typeof createProjectXRest>,
    private accountId: number,
    private contractId: string,
    private tickSize: number,
  ) {}

  private roundToTick(price: number): number {
    return Math.round(price / this.tickSize) * this.tickSize;
  }

  async placeLimitIOC(side: OrderSide, qty: number, price: number) {
    const payload: OrderPayload = {
      accountId: this.accountId,
      contractId: this.contractId,
      side: side === 'Buy' ? 0 : 1,
      size: qty,
      type: 1, // Limit
      timeInForce: 0, // IOC
      limitPrice: this.roundToTick(price),
    };
    log(`[ORDER] Placing ${side} limit IOC @ ${payload.limitPrice?.toFixed(2)} qty=${qty}`);
    return this.rest.placeOrder({ request: payload });
  }

  async placeMarketIOC(side: OrderSide, qty: number) {
    const payload: OrderPayload = {
      accountId: this.accountId,
      contractId: this.contractId,
      side: side === 'Buy' ? 0 : 1,
      size: qty,
      type: 2, // Market
      timeInForce: 0, // IOC
    };
    log(`[ORDER] Placing ${side} market IOC qty=${qty}`);
    return this.rest.placeOrder({ request: payload });
  }

  async cancelOrder(orderId: string | number) {
    log(`[ORDER] Canceling order ${orderId}`);
    return this.rest.cancelOrder({ accountId: this.accountId, orderId: String(orderId) });
  }

  /**
   * Place bracket order: MARKET entry + 0-tick limit IOC for both TP and SL
   * Returns { entryOrderId, stopOrderId, targetOrderId }
   */
  async placeBracketEntry(
    side: OrderSide,
    stopPrice: number,
    targetPrice: number,
    qty: number,
  ) {
    log(`[BRACKET] Entry ${side} MARKET, Stop @ ${stopPrice.toFixed(2)}, Target @ ${targetPrice.toFixed(2)}`);

    // Place entry as MARKET IOC
    const entryResponse = await this.placeMarketIOC(side, qty);
    const entryOrderId = this.resolveOrderId(entryResponse);

    log(`[BRACKET] Entry market order placed: ${entryOrderId}`);

    // Immediately place both bracket legs as 0-tick limit IOC
    const stopSide: OrderSide = side === 'Buy' ? 'Sell' : 'Buy';
    const targetSide: OrderSide = side === 'Buy' ? 'Sell' : 'Buy';

    const [stopResponse, targetResponse] = await Promise.all([
      this.placeLimitIOC(stopSide, qty, stopPrice),
      this.placeLimitIOC(targetSide, qty, targetPrice),
    ]);

    const stopOrderId = this.resolveOrderId(stopResponse);
    const targetOrderId = this.resolveOrderId(targetResponse);

    log(`[BRACKET] Stop order placed: ${stopOrderId}, Target order placed: ${targetOrderId}`);

    return {
      entryOrderId,
      stopOrderId,
      targetOrderId,
      entryFilled: this.isFilledResponse(entryResponse, qty),
      stopFilled: this.isFilledResponse(stopResponse, qty),
      targetFilled: this.isFilledResponse(targetResponse, qty),
    };
  }

  private resolveOrderId(response: any): string | number {
    return response?.orderId ?? response?.id ?? `topstep-${Date.now()}`;
  }

  private isFilledResponse(response: any, qty: number): boolean {
    const filled = Number(response?.filledQuantity ?? response?.filledQty ?? response?.filled ?? 0);
    return filled >= qty || response?.status === 'Filled';
  }
}

const CONFIG: StrategyConfig = {
  symbol: process.env.TOPSTEPX_NQ_LIVE_SYMBOL || 'NQZ5',
  contractId: process.env.TOPSTEPX_NQ_LIVE_CONTRACT_ID,
  bbPeriod: Number(process.env.TOPSTEPX_NQ_BB_PERIOD || '20'),
  bbStdDev: Number(process.env.TOPSTEPX_NQ_BB_STDDEV || '3'),
  rsiPeriod: Number(process.env.TOPSTEPX_NQ_RSI_PERIOD || '24'),
  rsiOversold: Number(process.env.TOPSTEPX_NQ_RSI_OVERSOLD || '30'),
  rsiOverbought: Number(process.env.TOPSTEPX_NQ_RSI_OVERBOUGHT || '70'),
  stopLossPercent: Number(process.env.TOPSTEPX_NQ_STOP_PERCENT || '0.0001'), // 0.01%
  takeProfitPercent: Number(process.env.TOPSTEPX_NQ_TP_PERCENT || '0.0005'), // 0.05%
  numberOfContracts: Number(process.env.TOPSTEPX_NQ_CONTRACTS || '3'),
  pollIntervalMs: Number(process.env.TOPSTEPX_NQ_POLL_MS || '60000'), // 1 minute
  initialBackfillBars: Number(process.env.TOPSTEPX_NQ_BACKFILL || '100'), // 100 1-minute bars
};

const STOP_MONITOR_DELAY_MS = Number(process.env.TOPSTEPX_NQ_STOP_MONITOR_MS || '1500'); // 1.5 seconds
const TOPSTEPX_LIVE_ACCOUNT_ID =
  process.env.TOPSTEPX_ACCOUNT_ID || process.env.TOPSTEPX_NQ_LIVE_ACCOUNT_ID;
const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market';
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user';

const TRADE_LOG_FILE = process.env.TOPSTEPX_NQ_TRADE_LOG || './logs/topstepx-nq-winner-live.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

let pendingSetup: PendingSetup | null = null;
let position: ActivePosition | null = null;
let closes: number[] = [];
let bars: TopstepXFuturesBar[] = [];
let multiplier = 20;
let realizedPnL = 0;
let lastProcessedBarTime = '';
let shuttingDown = false;
let tickSize = 0.25;
let commissionPerSide = 1.40;
let resolvedContractId: string | null = null;
let topstepRest: ReturnType<typeof createProjectXRest> | null = null;
let orderManager: TopstepOrderManager | null = null;
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let lastQuotePrice = 0;
let tradeSequence = 0;
let currentBar: TopstepXFuturesBar | null = null;
let barStartTime: Date | null = null;

function nowIso(): string {
  return new Date().toISOString();
}

function log(message: string) {
  console.log(`[${nowIso()}][${CONFIG.symbol}] ${message}`);
}

function nextTradeId() {
  tradeSequence += 1;
  return `NQ-WINNER-${Date.now()}-${tradeSequence}`;
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

function toCentralTime(date: Date): Date {
  return new Date(date.getTime() - CT_OFFSET_MINUTES * 60_000);
}

function isTradingAllowed(timestamp: string | Date): boolean {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 6) return false;
  if (day === 0 && minutes < WEEKEND_REOPEN_MINUTES) return false;
  if (day === 5 && minutes >= CUT_OFF_MINUTES) return false;

  return minutes < CUT_OFF_MINUTES || minutes >= REOPEN_MINUTES;
}

function shouldFlattenForClose(date: Date): boolean {
  const ctDate = toCentralTime(date);
  const day = ctDate.getUTCDay();
  const minutes = ctDate.getUTCHours() * 60 + ctDate.getUTCMinutes();

  if (day === 5 && minutes >= CUT_OFF_MINUTES - 5) return true;
  if (day !== 5 && minutes >= CUT_OFF_MINUTES - 5) return true;

  return false;
}

function calculateBollingerBands(
  values: number[],
  period: number,
  stdDev: number,
): { upper: number; middle: number; lower: number } | null {
  if (values.length < period) return null;

  const slice = values.slice(-period);
  const sum = slice.reduce((acc, val) => acc + val, 0);
  const mean = sum / period;

  const squaredDiffs = slice.map(val => Math.pow(val - mean, 2));
  const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / period;
  const standardDeviation = Math.sqrt(variance);

  return {
    upper: mean + standardDeviation * stdDev,
    middle: mean,
    lower: mean - standardDeviation * stdDev,
  };
}

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

function calculatePnL(entryPrice: number, exitPrice: number, side: 'long' | 'short', qty: number): number {
  const direction = side === 'long' ? 1 : -1;
  return (exitPrice - entryPrice) * direction * multiplier * qty;
}

function roundToTick(price: number): number {
  return Math.round(price / tickSize) * tickSize;
}

/**
 * Monitor stop limit order - if not filled after delay, convert to market
 */
async function monitorStopLimit(stopOrderId: string | number, exitSide: OrderSide, qty: number) {
  if (!position || !orderManager || position.stopFilled || position.monitoringStop) {
    return;
  }

  position.monitoringStop = true;
  log(`[MONITOR] Monitoring stop limit ${stopOrderId} for ${STOP_MONITOR_DELAY_MS}ms`);

  await sleep(STOP_MONITOR_DELAY_MS);

  // Check if position still exists and stop not filled
  if (!position || position.stopFilled) {
    log(`[MONITOR] Stop already filled or position closed`);
    return;
  }

  // Stop limit didn't fill - convert to market
  log(`[MONITOR] Stop limit ${stopOrderId} NOT filled - converting to MARKET STOP`);

  try {
    // Cancel the unfilled limit order
    await orderManager.cancelOrder(stopOrderId);
    log(`[MONITOR] Cancelled stop limit ${stopOrderId}`);

    // Place market order
    const marketResponse = await orderManager.placeMarketIOC(exitSide, qty);
    log(`[MONITOR] Market stop placed, order ID: ${marketResponse?.orderId ?? 'unknown'}`);

    // Mark stop as filled
    if (position) {
      position.stopFilled = true;
      const exitPrice = lastQuotePrice || position.stopLoss;
      await handlePositionExit(exitPrice, nowIso(), 'stop', true);
    }
  } catch (err: any) {
    log(`[ERROR] Failed to convert stop limit to market: ${err.message}`);
  } finally {
    if (position) {
      position.monitoringStop = false;
    }
  }
}

async function enterPosition(
  side: 'long' | 'short',
  price: number,
  timestamp: string,
  rsi: number,
  bb: { upper: number; middle: number; lower: number },
) {
  if (position) {
    log('Cannot enter: position already active');
    return;
  }

  if (!orderManager) {
    log('Order manager not initialized; cannot place entry');
    return;
  }

  const tradeId = nextTradeId();
  const entrySide: OrderSide = side === 'long' ? 'Buy' : 'Sell';
  const exitSide: OrderSide = side === 'long' ? 'Sell' : 'Buy';

  // Calculate bracket prices based on current bar close
  const stopPrice = roundToTick(
    side === 'long'
      ? price * (1 - CONFIG.stopLossPercent)
      : price * (1 + CONFIG.stopLossPercent)
  );
  const targetPrice = roundToTick(
    side === 'long'
      ? price * (1 + CONFIG.takeProfitPercent)
      : price * (1 - CONFIG.takeProfitPercent)
  );

  log(`[ENTRY] Attempting ${side.toUpperCase()} MARKET, Stop @ ${stopPrice.toFixed(2)}, Target @ ${targetPrice.toFixed(2)}`);

  let bracketResult;
  try {
    bracketResult = await orderManager.placeBracketEntry(
      entrySide,
      stopPrice,
      targetPrice,
      CONFIG.numberOfContracts,
    );
  } catch (err: any) {
    log(`[ERROR] Failed to place bracket order: ${err.message}`);
    return;
  }

  // Note: Actual entry price will come from fill event
  // For now, use bar close as estimate
  const estimatedEntryPrice = price;

  // Create position
  position = {
    tradeId,
    symbol: CONFIG.symbol,
    contractId: resolvedContractId ?? '',
    side,
    entryPrice: estimatedEntryPrice,
    entryTime: timestamp,
    stopLoss: stopPrice,
    target: targetPrice,
    totalQty: CONFIG.numberOfContracts,
    entryRSI: rsi,
    entryOrderId: bracketResult.entryOrderId,
    stopOrderId: bracketResult.stopOrderId,
    targetOrderId: bracketResult.targetOrderId,
    stopFilled: bracketResult.stopFilled,
    targetFilled: bracketResult.targetFilled,
    stopLimitPending: !bracketResult.stopFilled, // Monitor if not immediately filled
    monitoringStop: false,
  };

  log(
    `ENTERED ${side.toUpperCase()} MARKET @ ~${estimatedEntryPrice.toFixed(2)} ` +
    `(RSI ${rsi.toFixed(1)}, Entry: ${bracketResult.entryOrderId}, Stop: ${bracketResult.stopOrderId}, Target: ${bracketResult.targetOrderId})`
  );

  logTradeEvent({
    type: 'entry',
    tradeId,
    side: side.toUpperCase(),
    price: estimatedEntryPrice,
    orderType: 'MARKET',
    qty: CONFIG.numberOfContracts,
    rsi,
    bbUpper: bb.upper,
    bbMiddle: bb.middle,
    bbLower: bb.lower,
    stopLoss: stopPrice,
    target: targetPrice,
    entryOrderId: bracketResult.entryOrderId,
    stopOrderId: bracketResult.stopOrderId,
    targetOrderId: bracketResult.targetOrderId,
  });

  // Start monitoring stop limit if not immediately filled
  if (position.stopLimitPending && !position.stopFilled) {
    setImmediate(() => {
      if (position && !position.stopFilled) {
        monitorStopLimit(bracketResult.stopOrderId, exitSide, CONFIG.numberOfContracts);
      }
    });
  }
}

async function handlePositionExit(
  price: number,
  timestamp: string,
  reason: 'stop' | 'target' | 'end_of_session' | 'manual',
  isMarketStop: boolean = false,
) {
  if (!position) {
    return;
  }

  const fees = commissionPerSide * 2 * position.totalQty;
  const pnl = calculatePnL(position.entryPrice, price, position.side, position.totalQty);
  realizedPnL += pnl;

  const durationSeconds = (new Date(timestamp).getTime() - new Date(position.entryTime).getTime()) / 1000;

  log(
    `EXITED ${position.side.toUpperCase()} @ ${price.toFixed(2)} (${reason}${isMarketStop ? ' - MARKET' : ''}) ` +
    `| PnL: ${formatCurrency(pnl)} | Duration: ${durationSeconds.toFixed(0)}s | Cumulative: ${formatCurrency(realizedPnL)}`
  );

  logTradeEvent({
    type: 'exit',
    tradeId: position.tradeId,
    side: position.side.toUpperCase(),
    entryPrice: position.entryPrice,
    exitPrice: price,
    reason,
    isMarketStop,
    qty: position.totalQty,
    pnl,
    durationSeconds,
    cumulativePnL: realizedPnL,
    fees,
    stopOrderId: position.stopOrderId,
    targetOrderId: position.targetOrderId,
  });

  position = null;
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function resolveAccountId(): Promise<number> {
  if (TOPSTEPX_LIVE_ACCOUNT_ID) {
    const parsed = Number(TOPSTEPX_LIVE_ACCOUNT_ID);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  const accounts = await fetchTopstepXAccounts(true);
  if (!accounts.length) {
    throw new Error('No TopstepX accounts available');
  }

  const visible = accounts.find(acc => acc.canTrade && acc.isVisible);
  const chosen = visible ?? accounts[0];
  return chosen.id;
}

function updateCurrentBar(quote: any) {
  const price = resolveQuotePrice(quote);
  if (!price) return;

  lastQuotePrice = price;
  const timestamp = new Date(quote.timestamp || quote.lastTradeTimestamp || Date.now());

  // Check if we need to start a new 1-minute bar
  const barMinute = new Date(timestamp);
  barMinute.setSeconds(0, 0);

  if (!barStartTime || barStartTime.getTime() !== barMinute.getTime()) {
    // Close previous bar if exists
    if (currentBar) {
      processBar(currentBar);
    }

    // Start new bar
    barStartTime = barMinute;
    currentBar = {
      timestamp: barMinute.toISOString(),
      open: price,
      high: price,
      low: price,
      close: price,
    };
  } else if (currentBar) {
    // Update current bar
    currentBar.high = Math.max(currentBar.high, price);
    currentBar.low = Math.min(currentBar.low, price);
    currentBar.close = price;
  }
}

function resolveQuotePrice(quote: any): number {
  if (!quote) return 0;
  const last = Number(quote.lastPrice ?? quote.lastTradePrice ?? quote.price ?? 0);
  if (Number.isFinite(last) && last > 0) {
    return last;
  }
  const bid = Number(quote.bidPrice ?? quote.bestBid ?? 0);
  const ask = Number(quote.askPrice ?? quote.bestAsk ?? 0);
  if (Number.isFinite(bid) && Number.isFinite(ask) && bid > 0 && ask > 0) {
    return (bid + ask) / 2;
  }
  return Number(quote.close ?? quote.last ?? 0) || 0;
}

async function startMarketStream(contractId: string) {
  const tokenProvider = async () => authenticate();
  const initialToken = await tokenProvider();

  marketHub = new HubConnectionBuilder()
    .withUrl(`${MARKET_HUB_URL}?access_token=${encodeURIComponent(initialToken)}`, {
      skipNegotiation: true,
      transport: HttpTransportType.WebSockets,
      accessTokenFactory: tokenProvider,
    })
    .withAutomaticReconnect()
    .configureLogging(LogLevel.Information)
    .build();

  const handleQuote = (_contractId: string, quote: any) => {
    if (quote) {
      updateCurrentBar(quote);
    }
  };

  marketHub.on('GatewayQuote', handleQuote);
  marketHub.on('GatewayTrade', handleQuote);
  marketHub.on('gatewaytrade', handleQuote);

  const subscribeMarket = () => {
    if (!marketHub) return;
    marketHub.invoke('SubscribeContractQuotes', contractId).catch(err =>
      console.error('[market] Subscribe quotes failed', err),
    );
    marketHub.invoke('SubscribeContractTrades', contractId).catch(err =>
      console.error('[market] Subscribe trades failed', err),
    );
  };

  marketHub.onreconnected(subscribeMarket);

  await marketHub.start();
  log('TopstepX market hub connected');
  subscribeMarket();
}

async function startUserStream(accountId: number) {
  const tokenProvider = async () => authenticate();
  const initialToken = await tokenProvider();

  userHub = new HubConnectionBuilder()
    .withUrl(`${USER_HUB_URL}?access_token=${encodeURIComponent(initialToken)}`, {
      skipNegotiation: true,
      transport: HttpTransportType.WebSockets,
      accessTokenFactory: tokenProvider,
    })
    .withAutomaticReconnect()
    .configureLogging(LogLevel.Information)
    .build();

  userHub.on('GatewayUserTrade', (_cid: string, ev: any) => {
    if (!ev) return;
    const side = ev.side === 0 ? 'Buy' : 'Sell';
    const qty = Math.abs(Number(ev.size ?? ev.quantity ?? ev.qty ?? 0));
    const price = Number(ev.price ?? ev.avgPrice ?? 0);
    if (qty && price) {
      log(`User trade ${side} ${qty}@${price.toFixed(2)}`);

      // Update position fill status and actual entry price
      if (position) {
        const orderId = ev.orderId ?? ev.id;

        // Update actual entry price from market fill
        if (orderId === position.entryOrderId) {
          position.entryPrice = price;
          log(`Entry filled @ ${price.toFixed(2)} (market order)`);
        } else if (orderId === position.stopOrderId) {
          position.stopFilled = true;
          handlePositionExit(price, nowIso(), 'stop', false);
        } else if (orderId === position.targetOrderId) {
          position.targetFilled = true;
          handlePositionExit(price, nowIso(), 'target', false);
        }
      }
    }
  });

  userHub.on('GatewayUserOrder', data => {
    log(`User order event: ${JSON.stringify(data)}`);
  });

  const subscribeUser = () => {
    if (!userHub) return;
    userHub.invoke('SubscribeAccounts').catch(err => console.error('[user] Subscribe accounts failed', err));
    userHub.invoke('SubscribeOrders', accountId).catch(err => console.error('[user] Subscribe orders failed', err));
    userHub.invoke('SubscribePositions', accountId).catch(err => console.error('[user] Subscribe positions failed', err));
    userHub.invoke('SubscribeTrades', accountId).catch(err => console.error('[user] Subscribe trades failed', err));
  };

  userHub.onreconnected(subscribeUser);

  await userHub.start();
  log('TopstepX user hub connected');
  subscribeUser();
}

async function processBar(bar: TopstepXFuturesBar) {
  // Avoid reprocessing the same bar
  if (bar.timestamp === lastProcessedBarTime) {
    return;
  }
  lastProcessedBarTime = bar.timestamp;

  closes.push(bar.close);
  bars.push(bar);

  // Keep only necessary history
  const maxHistory = Math.max(CONFIG.bbPeriod + 100, 200);
  if (closes.length > maxHistory) {
    closes = closes.slice(-maxHistory);
  }
  if (bars.length > maxHistory) {
    bars = bars.slice(-maxHistory);
  }

  // Check if we should flatten for end of session
  if (position && shouldFlattenForClose(new Date(bar.timestamp))) {
    log('Flattening position for end of session');
    if (orderManager) {
      const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
      try {
        await orderManager.placeMarketIOC(exitSide, position.totalQty);
      } catch (err: any) {
        log(`[ERROR] Session flatten failed: ${err.message}`);
      }
    }
    await handlePositionExit(bar.close, bar.timestamp, 'end_of_session');
    return;
  }

  // Check if trading is allowed
  if (!isTradingAllowed(bar.timestamp)) {
    if (position) {
      log('Closing position - outside trading hours');
      if (orderManager) {
        const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
        try {
          await orderManager.placeMarketIOC(exitSide, position.totalQty);
        } catch (err: any) {
          log(`[ERROR] Hours close failed: ${err.message}`);
        }
      }
      await handlePositionExit(bar.close, bar.timestamp, 'end_of_session');
    }
    return;
  }

  // Don't enter new positions if we already have one
  if (position) {
    return;
  }

  // Need enough bars for indicators
  if (closes.length < CONFIG.bbPeriod) {
    return;
  }

  const bb = calculateBollingerBands(closes, CONFIG.bbPeriod, CONFIG.bbStdDev);
  if (!bb) {
    return;
  }

  const rsiValues = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });
  const currentRSI = rsiValues[rsiValues.length - 1];
  if (currentRSI === undefined) {
    return;
  }

  // Calculate TTM Squeeze
  const ttmBars = bars.slice(Math.max(0, bars.length - 21), bars.length - 1); // Exclude current bar
  const ttmSqueeze = calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });
  if (!ttmSqueeze) {
    return;
  }

  // Entry logic - two-stage system
  const price = bar.close;

  // Stage 1: Setup detection (BB + RSI)
  const longSetupDetected = price <= bb.lower && currentRSI < CONFIG.rsiOversold;
  const shortSetupDetected = price >= bb.upper && currentRSI > CONFIG.rsiOverbought;

  // Store pending setup if detected
  if (!pendingSetup && longSetupDetected) {
    pendingSetup = {
      side: 'long',
      setupTime: bar.timestamp,
      setupPrice: bar.close,
      rsi: currentRSI,
      bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
    };
    log(
      `LONG setup detected @ ${bar.close.toFixed(2)} (RSI ${currentRSI.toFixed(1)}, awaiting TTM Squeeze trigger)`
    );
  } else if (!pendingSetup && shortSetupDetected) {
    pendingSetup = {
      side: 'short',
      setupTime: bar.timestamp,
      setupPrice: bar.close,
      rsi: currentRSI,
      bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
    };
    log(
      `SHORT setup detected @ ${bar.close.toFixed(2)} (RSI ${currentRSI.toFixed(1)}, awaiting TTM Squeeze trigger)`
    );
  }

  // Stage 2: TTM Squeeze trigger - enter if we have pending setup and squeeze fires
  if (pendingSetup && ttmSqueeze.squeezeOn) {
    const setup = { ...pendingSetup };
    log(
      `TTM Squeeze trigger fired - entering ${setup.side.toUpperCase()} @ ${bar.close.toFixed(2)} ` +
      `(setup was @ ${setup.setupPrice.toFixed(2)})`
    );
    await enterPosition(setup.side, bar.close, bar.timestamp, setup.rsi, setup.bb);
    pendingSetup = null;
  }
}

async function shutdown(reason: string) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;

  log(`Shutting down (${reason})...`);

  if (position && orderManager) {
    const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
    try {
      await orderManager.placeMarketIOC(exitSide, position.totalQty);
      const lastClose = closes[closes.length - 1] || position.entryPrice;
      await handlePositionExit(lastClose, nowIso(), 'manual');
    } catch (err: any) {
      log(`[ERROR] Shutdown flatten failed: ${err.message}`);
    }
  }

  if (marketHub) {
    await marketHub.stop();
  }
  if (userHub) {
    await userHub.stop();
  }

  log(`Shutdown complete. Total realized PnL: ${formatCurrency(realizedPnL)}`);
  process.exit(0);
}

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX LIVE NQ WINNER STRATEGY (1-MINUTE BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`BB Period: ${CONFIG.bbPeriod} bars | Std Dev: ${CONFIG.bbStdDev}`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | Oversold: ${CONFIG.rsiOversold} | Overbought: ${CONFIG.rsiOverbought}`);
  console.log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Take Profit: ${(CONFIG.takeProfitPercent * 100).toFixed(3)}%`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Bracket Orders: Market entry, 0-tick limit exits, Stop Monitor: ${STOP_MONITOR_DELAY_MS}ms`);
  console.log('='.repeat(80));

  log('Main function started.');

  // Resolve contract metadata
  log('Resolving contract metadata...');
  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey);

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  const contractId = metadata.id;
  multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 20;

  resolvedContractId = contractId;

  log(`Resolved contract: ${metadata.name} (${contractId})`);
  log(`Point multiplier: ${multiplier}`);

  tickSize = metadata.tickSize;
  if (!tickSize || !Number.isFinite(tickSize) || tickSize <= 0) {
    throw new Error(`Unable to resolve tick size`);
  }

  commissionPerSide = process.env.TOPSTEPX_NQ_LIVE_COMMISSION
    ? Number(process.env.TOPSTEPX_NQ_LIVE_COMMISSION)
    : inferFuturesCommissionPerSide([CONFIG.contractId, CONFIG.symbol, metadata.id], 1.40);

  log(`Tick size: ${tickSize}`);
  log(`Commission/side: ${commissionPerSide.toFixed(2)} USD`);

  log('Resolving account ID...');
  const accountId = await resolveAccountId();
  log('Creating ProjectX REST client...');
  topstepRest = createProjectXRest();
  orderManager = new TopstepOrderManager(topstepRest, accountId, contractId, tickSize);
  log(`Using TopstepX account ${accountId}`);

  // Initial backfill
  log(`Fetching initial ${CONFIG.initialBackfillBars} 1-minute bars...`);
  const initialBars = (await fetchTopstepXFuturesBars({
    contractId,
    startTime: new Date(Date.now() - CONFIG.initialBackfillBars * 60 * 1000).toISOString(),
    endTime: new Date().toISOString(),
    unit: 2, // Minutes
    unitNumber: 1,
    limit: CONFIG.initialBackfillBars,
  })).reverse();
  log(`Loaded ${initialBars.length} initial bars`);

  for (const bar of initialBars) {
    closes.push(bar.close);
    bars.push(bar);
  }

  log('Starting live streaming...');

  // Register shutdown handlers
  log('Registering shutdown handlers...');
  process.once('SIGINT', () => shutdown('SIGINT'));
  process.once('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGUSR2', () => {
    if (shuttingDown) return;
    if (!position) {
      log('Flatten requested but no active position.');
      return;
    }
    shutdown('SIGUSR2').catch(err =>
      console.error('[signal] manual flatten failed', err),
    );
  });

  log('Starting user stream...');
  await startUserStream(accountId);
  log('Starting market stream...');
  await startMarketStream(contractId);
  log('Live streaming started. Strategy is running...');
  await new Promise(() => {});
}

main().catch(err => {
  console.error('TopstepX NQ winner live strategy failed:', err);
  process.exit(1);
});
