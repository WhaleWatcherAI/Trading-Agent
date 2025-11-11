#!/usr/bin/env tsx
/**
 * TopstepX Live Mean Reversion Strategy on 1-Second Bars
 *
 * Strategy:
 * - Bollinger Bands: 200-period SMA with 3 standard deviations on 1s bars
 * - RSI confirmation: RSI(14) for entry filtering
 * - Two-stage entry:
 *   Stage 1: Price within 5% of BB width from outer band + RSI creates setup
 *   Stage 2: TTM Squeeze ON triggers entry (even if BB/RSI changed)
 * - Scale: Take 50% profit at middle band (200 SMA) - exact
 * - Runner target: Opposite outer BB (within 5% of BB width) - consistent with entry
 * - Stop: 0.03% from middle band after scaling
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
  stopLoss: number | null;
  target: number | null;
  scaled: boolean;
  totalQty: number;
  remainingQty: number;
  scalePnL: number;
  entryRSI: number;
  entrySlippageTicks: number;
  entryOrderId?: string | number;
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
  numberOfContracts: number;
  pollIntervalMs: number;
  initialBackfillBars: number;
}

type OrderSide = 'Buy' | 'Sell';
type OrderStage = 'limit-zero' | 'limit-one' | 'market';

interface AggressiveOrderResult {
  price: number;
  orderId: string | number;
  stage: OrderStage;
  filledQty: number;
}

class TopstepOrderManager {
  constructor(
    private rest: ReturnType<typeof createProjectXRest>,
    private accountId: number,
    private contractId: string,
    private tickSize: number,
  ) {}

  private async placeLimitOrder(side: OrderSide, qty: number, price: number) {
    const payload = {
      accountId: this.accountId,
      contractId: this.contractId,
      side: side === 'Buy' ? 0 : 1, // 0 = Bid (Buy), 1 = Ask (Sell)
      size: qty,
      type: 1, // 1 = Limit
      timeInForce: 0, // 0 = IOC
      limitPrice: Number(price.toFixed(5)),
    };
    return this.rest.placeOrder(payload);
  }

  private async placeMarketOrder(side: OrderSide, qty: number) {
    const payload = {
      accountId: this.accountId,
      contractId: this.contractId,
      side: side === 'Buy' ? 0 : 1,
      size: qty,
      type: 2, // 2 = Market
      timeInForce: 0,
    };
    return this.rest.placeOrder(payload);
  }

  private static extractFilledQty(response: any): number {
    return (
      Number(response?.filledQuantity ?? response?.filledQty ?? response?.filled ?? 0)
    );
  }

  private static resolveOrderId(response: any) {
    return response?.orderId ?? response?.id ?? `topstep-${Date.now()}`;
  }

  private isFilled(response: any, qty: number) {
    const filled = TopstepOrderManager.extractFilledQty(response);
    return filled >= qty || (response?.status === 'Filled');
  }

  private adjustPrice(base: number, offsetTicks: number, side: OrderSide) {
    const offset = offsetTicks * this.tickSize * (side === 'Buy' ? 1 : -1);
    return Math.round((base + offset) / this.tickSize) * this.tickSize;
  }

  async aggressiveEntry(side: OrderSide, basePrice: number, qty: number): Promise<AggressiveOrderResult> {
    const offsets = [0, 1];
    for (let i = 0; i < offsets.length; i += 1) {
      const stage: OrderStage = i === 0 ? 'limit-zero' : 'limit-one';
      const price = this.adjustPrice(basePrice, offsets[i], side);
      const response = await this.placeLimitOrder(side, qty, price);
      if (this.isFilled(response, qty)) {
        return {
          price,
          orderId: TopstepOrderManager.resolveOrderId(response),
          stage,
          filledQty: qty,
        };
      }
      await sleep(ORDER_RETRY_DELAY_MS);
    }

    const response = await this.placeMarketOrder(side, qty);
    return {
      price: basePrice,
      orderId: TopstepOrderManager.resolveOrderId(response),
      stage: 'market',
      filledQty: qty,
    };
  }

  async aggressiveExit(side: OrderSide, basePrice: number, qty: number): Promise<AggressiveOrderResult> {
    return this.aggressiveEntry(side, basePrice, qty);
  }
}

function stageSlippageTicks(stage: OrderStage | null): number {
  if (!stage) return 0;
  switch (stage) {
    case 'limit-zero':
      return 0;
    case 'limit-one':
      return 1;
    case 'market':
      return 2;
  }
}

const CONFIG: StrategyConfig = {
  symbol: process.env.TOPSTEPX_MR_LIVE_SYMBOL || 'MESZ5',
  contractId: process.env.TOPSTEPX_MR_LIVE_CONTRACT_ID,
  bbPeriod: Number(process.env.TOPSTEPX_MR_LIVE_BB_PERIOD || '200'),
  bbStdDev: Number(process.env.TOPSTEPX_MR_LIVE_BB_STDDEV || '3'),
  rsiPeriod: Number(process.env.TOPSTEPX_MR_LIVE_RSI_PERIOD || '14'),
  rsiOversold: Number(process.env.TOPSTEPX_MR_LIVE_RSI_OVERSOLD || '30'),
  rsiOverbought: Number(process.env.TOPSTEPX_MR_LIVE_RSI_OVERBOUGHT || '70'),
  stopLossPercent: Number(process.env.TOPSTEPX_MR_LIVE_STOP_LOSS_PERCENT || '0.0003'),
  numberOfContracts: Number(process.env.TOPSTEPX_MR_LIVE_CONTRACTS || '2'),
  pollIntervalMs: Number(process.env.TOPSTEPX_MR_LIVE_POLL_MS || '1000'),
  initialBackfillBars: Number(process.env.TOPSTEPX_MR_LIVE_BACKFILL || '300'),
};

const ORDER_RETRY_DELAY_MS = Number(process.env.TOPSTEPX_MR_ORDER_RETRY_MS || '400');
const TOPSTEPX_LIVE_ACCOUNT_ID =
  process.env.TOPSTEPX_ACCOUNT_ID || process.env.TOPSTEPX_MR_LIVE_ACCOUNT_ID;
const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market';
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user';

const TRADE_LOG_FILE = process.env.TOPSTEPX_MR_TRADE_LOG || './logs/topstepx-mean-reversion-1s.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 18 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

let pendingSetup: {
  side: 'long' | 'short';
  setupTime: string;
  setupPrice: number;
  rsi: number;
  bb: { upper: number; middle: number; lower: number };
} | null = null;
let pendingSetupLoggedSide: 'long' | 'short' | null = null;
let lastLongBlockedLog = 0;
let lastShortBlockedLog = 0;

let position: ActivePosition | null = null;
let closes: number[] = [];
let bars: { high: number; low: number; close: number }[] = [];
let multiplier = 5;
let realizedPnL = 0;
let lastProcessedBarTime = '';
let shuttingDown = false;
let slipSymbol = 'ES';
let tickSize = 0;
let commissionPerSide = 0;
let resolvedContractId: string | null = null;
let topstepRest: ReturnType<typeof createProjectXRest> | null = null;
let orderManager: TopstepOrderManager | null = null;
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let lastClose = 0;
let tradeSequence = 0;

function nowIso(): string {
  return new Date().toISOString();
}

function log(message: string) {
  console.log(`[${nowIso()}][${CONFIG.symbol}] ${message}`);
}

function nextTradeId() {
  tradeSequence += 1;
  return `MR-${Date.now()}-${tradeSequence}`;
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

function processQuote(quote: any) {
  const price = resolveQuotePrice(quote);
  if (!price) return;
  lastClose = price;
  const timestamp = new Date(quote.timestamp || quote.lastTradeTimestamp || Date.now());
  const bar: TopstepXFuturesBar = {
    timestamp: timestamp.toISOString(),
    open: price,
    high: price,
    low: price,
    close: price,
  };
  processBar(bar);
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

  const handleMarketTrade = (_contractId: string, trade: any) => {
    if (trade) {
      processQuote(trade);
    }
  };

  marketHub.on('GatewayQuote', (_contractId: string, quote: any) => {
    processQuote(quote);
  });
  marketHub.on('GatewayTrade', handleMarketTrade);
  marketHub.on('gatewaytrade', handleMarketTrade);

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

  const handleUserTrade = (_cid: string, ev: any) => {
    if (!ev) {
      log('[WARN] handleUserTrade: received undefined event');
      return;
    }
    const side = ev.side === 0 ? 'Buy' : 'Sell';
    const qty = Math.abs(Number(ev.size ?? ev.quantity ?? ev.qty ?? 0));
    const price = Number(ev.price ?? ev.avgPrice ?? 0);
    if (qty && price) {
      log(`User trade ${side} ${qty}@${price.toFixed(2)}`);
    }
  };
  userHub.on('GatewayUserTrade', handleUserTrade);
  userHub.on('gatewaytrade', handleUserTrade);
  userHub.on('gatewayusertrade', handleUserTrade);

  userHub.on('GatewayUserOrder', data => {
    log(`User order event: ${JSON.stringify(data)}`);
  });

  userHub.on('GatewayUserPosition', (_cid: string, data: any) => {
    if (data) log(`[position] ${JSON.stringify(data)}`);
  });
  userHub.on('gatewayuserposition', (_cid: string, data: any) => {
    if (data) log(`[position] ${JSON.stringify(data)}`);
  });

  userHub.on('GatewayUserAccount', (_cid: string, data: any) => {
    if (data) log(`[account] ${JSON.stringify(data)}`);
  });
  userHub.on('gatewayuseraccount', (_cid: string, data: any) => {
    if (data) log(`[account] ${JSON.stringify(data)}`);
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

function formatCurrency(value: number): string {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

function calculatePnL(entryPrice: number, exitPrice: number, side: 'long' | 'short', qty: number): number {
  const direction = side === 'long' ? 1 : -1;
  return (exitPrice - entryPrice) * direction * multiplier * qty;
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

  const entrySide: OrderSide = side === 'long' ? 'Buy' : 'Sell';
  let fillResult;
  try {
    fillResult = await orderManager.aggressiveEntry(entrySide, price, CONFIG.numberOfContracts);
  } catch (err: any) {
    log(`[ERROR] Failed to place entry order: ${err.message}`);
    if (err.message?.includes('429')) {
      log('Rate limit hit - will retry on next setup');
    }
    return;
  }
  const slippageTicks = stageSlippageTicks(fillResult.stage);
  const tradeId = nextTradeId();

  position = {
    tradeId,
    symbol: CONFIG.symbol,
    contractId: resolvedContractId ?? '',
    side,
    entryPrice: fillResult.price,
    entryTime: timestamp,
    stopLoss: side === 'long'
      ? bb.lower * (1 - CONFIG.stopLossPercent)
      : bb.upper * (1 + CONFIG.stopLossPercent),
    target: bb.middle,
    scaled: false,
    totalQty: CONFIG.numberOfContracts,
    remainingQty: CONFIG.numberOfContracts,
    scalePnL: 0,
    entryRSI: rsi,
    entrySlippageTicks: slippageTicks,
  };

  log(
    `ENTERED ${side.toUpperCase()} @ ${fillResult.price.toFixed(2)} ` +
    `(stage ${fillResult.stage}, RSI ${rsi.toFixed(1)}, BB ${bb.lower.toFixed(2)}/${bb.middle.toFixed(2)}/${bb.upper.toFixed(2)}, ` +
    `stop ${position.stopLoss?.toFixed(2)}, target ${position.target?.toFixed(2)})`
  );

  logTradeEvent({
    type: 'entry',
    tradeId,
    side: side.toUpperCase(),
    price: fillResult.price,
    marketPrice: price,
    qty: CONFIG.numberOfContracts,
    rsi,
    bbUpper: bb.upper,
    bbMiddle: bb.middle,
    bbLower: bb.lower,
    stopLoss: position.stopLoss,
    target: position.target,
    entrySlippageTicks: slippageTicks,
    slipSymbol,
    entryOrderId: fillResult.orderId,
    entryStage: fillResult.stage,
  });
}

async function scalePosition(
  price: number,
  timestamp: string,
  bb: { upper: number; middle: number; lower: number },
) {
  if (!position || position.scaled) {
    return;
  }

  const scaleQty = Math.floor(position.totalQty / 2);
  const remainingQty = position.totalQty - scaleQty;

  if (scaleQty <= 0) {
    return;
  }

  const pnl = calculatePnL(position.entryPrice, price, position.side, scaleQty);
  position.scalePnL = pnl;
  position.scaled = true;
  position.remainingQty = remainingQty;

  // Update stop and target for runner
  const direction = position.side === 'long' ? 1 : -1;
  position.stopLoss = bb.middle * (1 + direction * -CONFIG.stopLossPercent);
  position.target = position.side === 'long' ? bb.upper : bb.lower;

  realizedPnL += pnl;

  log(
    `SCALED ${position.side.toUpperCase()} @ ${price.toFixed(2)} ` +
    `(${scaleQty} contracts, PnL ${formatCurrency(pnl)}, ${remainingQty} remaining, ` +
    `new stop ${position.stopLoss.toFixed(2)}, new target ${position.target.toFixed(2)})`
  );

  logTradeEvent({
    type: 'scale',
    tradeId: position.tradeId,
    side: position.side.toUpperCase(),
    price,
    qty: scaleQty,
    pnl,
    remainingQty,
    newStop: position.stopLoss,
    newTarget: position.target,
    cumulativePnL: realizedPnL,
  });
}

async function exitPosition(
  price: number,
  timestamp: string,
  reason: 'stop' | 'target' | 'end_of_session' | 'manual',
  stage: OrderStage | null,
  orderId?: string | number,
) {
  if (!position) {
    return;
  }

  const exitSlippageTicks = stageSlippageTicks(stage);
  const exitSlippagePoints = exitSlippageTicks * tickSize;
  const entrySlippagePoints = position.entrySlippageTicks * tickSize;
  const slippageCost = (entrySlippagePoints + exitSlippagePoints) * multiplier * position.remainingQty;
  const fees = commissionPerSide * 2 * position.remainingQty;

  const pnl = calculatePnL(position.entryPrice, price, position.side, position.remainingQty);
  const totalPnL = position.scalePnL + pnl;
  realizedPnL += pnl;

  const durationSeconds = (new Date(timestamp).getTime() - new Date(position.entryTime).getTime()) / 1000;

  log(
    `EXITED ${position.side.toUpperCase()} @ ${price.toFixed(2)} (${reason}) ` +
    `| Exit PnL: ${formatCurrency(pnl)} | Total PnL: ${formatCurrency(totalPnL)} ` +
    `| Duration: ${durationSeconds.toFixed(0)}s | Cumulative: ${formatCurrency(realizedPnL)}`
  );

  logTradeEvent({
    type: 'exit',
    tradeId: position.tradeId,
    side: position.side.toUpperCase(),
    entryPrice: position.entryPrice,
    exitPrice: price,
    reason,
    qty: position.remainingQty,
    exitPnL: pnl,
    scalePnL: position.scalePnL,
    totalPnL,
    durationSeconds,
    cumulativePnL: realizedPnL,
    scaled: position.scaled,
    entrySlippageTicks: position.entrySlippageTicks,
    exitSlippageTicks,
    slippageCost,
    fees,
    slipSymbol,
    orderId,
    exitStage: stage,
  });

  position = null;
}

async function executeExit(
  reason: 'stop' | 'target' | 'end_of_session' | 'manual',
  basePrice: number,
  timestamp: string,
) {
  if (!position || !orderManager) {
    if (position) {
      await exitPosition(basePrice, timestamp, reason, null);
    }
    return;
  }

  const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
  let fill;
  try {
    fill = await orderManager.aggressiveExit(exitSide, basePrice, position.remainingQty);
  } catch (err: any) {
    log(`[ERROR] Failed to place exit order: ${err.message}`);
    if (err.message?.includes('429')) {
      log('Rate limit hit on exit - using last price for record keeping');
    }
    // Exit the position in our records even if the order failed
    await exitPosition(basePrice, timestamp, reason, null, null);
    return;
  }
  await exitPosition(fill.price, timestamp, reason, fill.stage, fill.orderId);
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

async function processBar(bar: TopstepXFuturesBar) {
  // Avoid reprocessing the same bar
  if (bar.timestamp === lastProcessedBarTime) {
    return;
  }
  lastProcessedBarTime = bar.timestamp;

  closes.push(bar.close);
  bars.push({ high: bar.high, low: bar.low, close: bar.close });

  // Keep only necessary history
  const maxHistory = Math.max(CONFIG.bbPeriod + 100, 500);
  if (closes.length > maxHistory) {
    closes = closes.slice(-maxHistory);
  }
  if (bars.length > maxHistory) {
    bars = bars.slice(-maxHistory);
  }

  // Check if we should flatten for end of session
  if (position && shouldFlattenForClose(new Date(bar.timestamp))) {
    await executeExit('end_of_session', bar.close, bar.timestamp);
    return;
  }

  // Check if trading is allowed
  if (!isTradingAllowed(bar.timestamp)) {
    if (position) {
      await executeExit('end_of_session', bar.close, bar.timestamp);
    }
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

  // Calculate BB width (needed for both position monitoring and entry logic)
  const bbWidth = bb.upper - bb.lower;

  const rsiValues = RSI.calculate({ values: closes, period: CONFIG.rsiPeriod });
  const currentRSI = rsiValues[rsiValues.length - 1];
  if (currentRSI === undefined) {
    return;
  }

  // Calculate TTM Squeeze
  const ttmSqueeze = calculateTtmSqueeze(bars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 });
  if (!ttmSqueeze) {
    return;
  }

  // Monitor active position
  if (position) {
    const direction = position.side === 'long' ? 1 : -1;

    // Check for scaling at middle band (first time only)
    if (!position.scaled && position.target !== null) {
      const hitTarget = (direction === 1 && bar.high >= position.target) ||
                       (direction === -1 && bar.low <= position.target);

      if (hitTarget) {
        await scalePosition(position.target, bar.timestamp, bb);
        return;
      }
    }

    // Check stop loss
    if (position.stopLoss !== null) {
      const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                     (direction === -1 && bar.high >= position.stopLoss);

      if (hitStop) {
        const stopPrice = position.stopLoss ?? bar.close;
        await executeExit('stop', stopPrice, bar.timestamp);
        return;
      }
    }

    // Check final target (after scaling) - use 5% BB width buffer for outer band
    if (position.scaled && position.target !== null) {
      const targetBuffer = bbWidth * 0.05;
      const hitTarget = (direction === 1 && bar.high >= position.target - targetBuffer) ||
                       (direction === -1 && bar.low <= position.target + targetBuffer);

      if (hitTarget) {
        const basePrice = direction === 1
          ? Math.min(bar.high, position.target)
          : Math.max(bar.low, position.target);
        await executeExit('target', basePrice, bar.timestamp);
        return;
      }
    }

    return;
  }

  // Entry logic - two-stage system
  const price = bar.close;
  const distanceToUpper = bb.upper - price;
  const distanceToLower = price - bb.lower;

  // Stage 1: Setup detection (BB + RSI) - must be within 5% of BB width
  const longSetupDetected = distanceToLower <= bbWidth * 0.05 && currentRSI < CONFIG.rsiOversold;
  const shortSetupDetected = distanceToUpper <= bbWidth * 0.05 && currentRSI > CONFIG.rsiOverbought;

  const nowMs = Date.now();
  if (!pendingSetup && distanceToLower <= bbWidth * 0.05 && !longSetupDetected) {
    if (currentRSI >= CONFIG.rsiOversold && nowMs - lastLongBlockedLog > 5000) {
      log(
        `[setup-check] LONG blocked: RSI ${currentRSI.toFixed(1)} >= oversold ${CONFIG.rsiOversold} (price ${price.toFixed(2)})`,
      );
      lastLongBlockedLog = nowMs;
    }
  }
  if (!pendingSetup && distanceToUpper <= bbWidth * 0.05 && !shortSetupDetected) {
    if (currentRSI <= CONFIG.rsiOverbought && nowMs - lastShortBlockedLog > 5000) {
      log(
        `[setup-check] SHORT blocked: RSI ${currentRSI.toFixed(1)} <= overbought ${CONFIG.rsiOverbought} (price ${price.toFixed(2)})`,
      );
      lastShortBlockedLog = nowMs;
    }
  }

  // Store pending setup if detected and no setup already pending
  if (!pendingSetup && longSetupDetected) {
    pendingSetup = {
      side: 'long',
      setupTime: bar.timestamp,
      setupPrice: bar.close,
      rsi: currentRSI,
      bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
    };
    pendingSetupLoggedSide = null;
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
    pendingSetupLoggedSide = null;
    log(
      `SHORT setup detected @ ${bar.close.toFixed(2)} (RSI ${currentRSI.toFixed(1)}, awaiting TTM Squeeze trigger)`
    );
  }

  // Stage 2: TTM Squeeze trigger - enter if we have pending setup and squeeze fires
  if (pendingSetup && !ttmSqueeze.squeezeOn && pendingSetupLoggedSide !== pendingSetup.side) {
    log(
      `[setup] ${pendingSetup.side.toUpperCase()} waiting for TTM squeeze ON (momentum ${ttmSqueeze.momentum.toFixed(4)}, sentiment ${ttmSqueeze.sentiment})`,
    );
    pendingSetupLoggedSide = pendingSetup.side;
  }

  if (pendingSetup && ttmSqueeze.squeezeOn) {
    // Save setup values before async call in case pendingSetup gets cleared
    const setup = { ...pendingSetup };
    await enterPosition(setup.side, bar.close, bar.timestamp, setup.rsi, setup.bb);
    log(
      `TTM Squeeze trigger fired - entering ${setup.side.toUpperCase()} @ ${bar.close.toFixed(2)} ` +
      `(setup was @ ${setup.setupPrice.toFixed(2)})`
    );
    // Clear pending setup after entry
    pendingSetup = null;
    pendingSetupLoggedSide = null;
  }
}

async function shutdown(reason: string) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;

  log(`Shutting down (${reason})...`);

  if (position) {
    const lastClose = closes[closes.length - 1] || position.entryPrice;
    await executeExit('manual', lastClose, nowIso());
  }

  if (marketHub) {
    await marketHub.stop();
  }

  log(`Shutdown complete. Total realized PnL: ${formatCurrency(realizedPnL)}`);
  process.exit(0);
}

async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('TOPSTEPX LIVE MEAN REVERSION STRATEGY (1-SECOND BARS)');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`BB Period: ${CONFIG.bbPeriod} bars (${CONFIG.bbPeriod}s) | Std Dev: ${CONFIG.bbStdDev}`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | Oversold: ${CONFIG.rsiOversold} | Overbought: ${CONFIG.rsiOverbought}`);
  console.log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Poll Interval: ${CONFIG.pollIntervalMs}ms`);
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
    : metadata.multiplier || 5;

  resolvedContractId = contractId;

  log(`Resolved contract: ${metadata.name} (${contractId})`);
  log(`Point multiplier: ${multiplier}`);

  const baseSymbol = getBaseSymbol(CONFIG.symbol);
  slipSymbol = baseSymbol;
  tickSize = metadata.tickSize;
  if (!tickSize || !Number.isFinite(tickSize) || tickSize <= 0) {
    throw new Error(`Unable to resolve tick size for ${slipSymbol}`);
  }
  commissionPerSide = process.env.TOPSTEPX_MR_LIVE_COMMISSION
    ? Number(process.env.TOPSTEPX_MR_LIVE_COMMISSION)
    : inferFuturesCommissionPerSide([CONFIG.contractId, CONFIG.symbol, metadata.id], 0.35);

  log(`Tick size for ${slipSymbol}: ${tickSize}`);
  log(`Commission/side: ${commissionPerSide.toFixed(2)} USD`);

  log('Resolving account ID...');
  const accountId = await resolveAccountId();
  log('Creating ProjectX REST client...');
  topstepRest = createProjectXRest();
  orderManager = new TopstepOrderManager(topstepRest, accountId, contractId, tickSize);
  log(`Using TopstepX account ${accountId}`);

  // Initial backfill
  log(`Fetching initial ${CONFIG.initialBackfillBars} bars...`);
  const initialBars = (await fetchTopstepXFuturesBars({
    contractId,
    startTime: new Date(Date.now() - CONFIG.initialBackfillBars * 1000).toISOString(),
    endTime: new Date().toISOString(),
    unit: 1,
    unitNumber: 1,
    limit: CONFIG.initialBackfillBars,
  })).reverse();
  log(`Loaded ${initialBars.length} initial bars`);

  for (const bar of initialBars) {
    closes.push(bar.close);
  }

  log('Starting live streaming...');

  // Register shutdown handlers
  log('Registering shutdown handlers...');
  process.once('SIGINT', () => shutdown('SIGINT'));
  process.once('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGUSR2', () => {
    if (shuttingDown) {
      return;
    }
    if (!position) {
      log('Flatten requested but no active position.');
      return;
    }
    const priceCandidate =
      lastClose ||
      (bars.length ? bars[bars.length - 1].close : 0) ||
      position.entryPrice;
    executeExit('manual', priceCandidate, nowIso()).catch(err =>
      console.error('[signal] manual flatten failed', err),
    );
  });

  log('Starting user stream...');
  await startUserStream(accountId);
  log('Starting market stream...');
  await startMarketStream(contractId);
  log('Live streaming started. Entering infinite loop...');
  await new Promise(() => {});
}

main().catch(err => {
  console.error('TopstepX mean reversion live strategy failed:', err);
  process.exit(1);
});
