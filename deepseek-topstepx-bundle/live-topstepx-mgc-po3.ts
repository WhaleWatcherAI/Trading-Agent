#!/usr/bin/env tsx
/**
 * TopstepX Live MGC Power of Three (PO3) Strategy
 *
 * STRATEGY OVERVIEW:
 * The "Power of Three" is ICT's session-based trading model:
 * 1. Accumulation (Asia range: 20:00-00:00 ET)
 * 2. Manipulation (London sweeps Asia H/L: 02:00-05:00 ET)
 * 3. Distribution (NY trades back into range: 09:30-11:30, 13:30-15:30 ET)
 *
 * ENTRY LOGIC:
 * 1. Track Asia session range (accumulation phase)
 * 2. Detect London manipulation (sweep of Asia high/low)
 * 3. Enter on Fair Value Gap (FVG) in NY session in opposite direction
 * 4. Scale out 50% at TP1 (Asia midpoint), move stop to breakeven
 * 5. Exit remaining at TP2 (75% of Asia range from midpoint)
 *
 * CRITICAL STOP LOGIC:
 * - LONG: Stop = FVG.lower - buffer (BELOW entry)
 * - SHORT: Stop = FVG.upper + buffer (ABOVE entry)
 *
 * FEATURES:
 * - Real-time WebSocket/SignalR data streaming
 * - REST API endpoints for dashboard integration
 * - Account monitoring with safety limits
 * - State persistence with position recovery
 * - One trade per day maximum
 */

import 'dotenv/config';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  fetchTopstepXAccounts,
  TopstepXFuturesBar,
  authenticate,
} from './lib/topstepx';
import { appendFileSync, existsSync, mkdirSync, writeFileSync, readFileSync } from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { createProjectXRest } from './projectx-rest';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import express from 'express';
import cors from 'cors';
import { Server } from 'socket.io';
import http from 'http';

function getBaseSymbol(fullSymbol: string): string {
  return fullSymbol.replace(/[A-Z]\d+$/, '');
}

interface ChartData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  asiaHigh?: number;
  asiaLow?: number;
  asiaMidpoint?: number;
  londonSweep?: number;
  signal?: 'long' | 'short' | null;
  entry?: boolean;
  exit?: boolean;
  manipulation?: 'bullish' | 'bearish' | null;
}

interface ActivePosition {
  tradeId: string;
  symbol: string;
  contractId: string;
  side: 'long' | 'short';
  entryPrice: number;
  entryTime: string;
  stopLoss: number;
  tp1: number;
  tp2: number;
  totalQty: number;
  remainingQty: number;
  scaledQty: number;
  entryOrderId: string | number;
  stopOrderId?: string | number;
  targetOrderId?: string | number;
  stopFilled: boolean;
  targetFilled: boolean;
  stopLimitPending: boolean;
  monitoringStop: boolean;
  unrealizedPnL?: number;
  entryCommission?: number;
  exitCommission?: number;
  asiaHigh: number;
  asiaLow: number;
  londonSweep: number;
  fvgMidpoint: number;
  tp1Hit: boolean;
}

interface AccountStatus {
  balance: number;
  buyingPower: number;
  dailyPnL: number;
  openPositions: number;
  dailyLossLimit: number;
  isAtRisk: boolean;
}

interface StrategyConfig {
  symbol: string;
  contractId?: string;
  numberOfContracts: number;
  scaleOutPercent: number;
  stopLossBuffer: number;
  minAsiaRangeATR: number;
  minFVGSizeTicks: number;
  minFVGSizeATR: number;
  minSweepTicks: number;
  minBarsAfterSweep: number;
  tp2RangePercent: number;
  pollIntervalMs: number;
  initialBackfillBars: number;
  dailyLossLimit: number;
}

type OrderSide = 'Buy' | 'Sell';

interface OrderPayload {
  accountId: number;
  contractId: string;
  side: 0 | 1;
  size: number;
  type: 1 | 2;
  timeInForce: 0;
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
      type: 1,
      timeInForce: 0,
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
      type: 2,
      timeInForce: 0,
    };
    log(`[ORDER] Placing ${side} market IOC qty=${qty}`);
    return this.rest.placeOrder({ request: payload });
  }

  async cancelOrder(orderId: string | number) {
    log(`[ORDER] Canceling order ${orderId}`);
    return this.rest.cancelOrder({ accountId: this.accountId, orderId: String(orderId) });
  }

  async placeBracketEntry(
    side: OrderSide,
    stopPrice: number,
    targetPrice: number,
    qty: number,
  ) {
    log(`[BRACKET] Entry ${side} MARKET, Stop @ ${stopPrice.toFixed(2)}, Target @ ${targetPrice.toFixed(2)}`);

    const entryResponse = await this.placeMarketIOC(side, qty);
    const entryOrderId = this.resolveOrderId(entryResponse);

    log(`[BRACKET] Entry market order placed: ${entryOrderId}`);

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
  symbol: process.env.TOPSTEPX_MGC_LIVE_SYMBOL || 'MGCZ5',
  contractId: process.env.TOPSTEPX_MGC_LIVE_CONTRACT_ID,
  numberOfContracts: Number(process.env.TOPSTEPX_MGC_CONTRACTS || '2'),
  scaleOutPercent: Number(process.env.TOPSTEPX_MGC_SCALE_PERCENT || '0.5'),
  stopLossBuffer: Number(process.env.TOPSTEPX_MGC_SL_BUFFER || '2'),
  minAsiaRangeATR: Number(process.env.TOPSTEPX_MGC_MIN_ASIA_ATR || '0.8'),
  minFVGSizeTicks: Number(process.env.TOPSTEPX_MGC_MIN_FVG_TICKS || '6'),
  minFVGSizeATR: Number(process.env.TOPSTEPX_MGC_MIN_FVG_ATR || '0.35'),
  minSweepTicks: Number(process.env.TOPSTEPX_MGC_MIN_SWEEP_TICKS || '5'),
  minBarsAfterSweep: Number(process.env.TOPSTEPX_MGC_MIN_BARS_AFTER_SWEEP || '30'),
  tp2RangePercent: Number(process.env.TOPSTEPX_MGC_TP2_RANGE_PCT || '0.75'),
  pollIntervalMs: Number(process.env.TOPSTEPX_MGC_POLL_MS || '60000'),
  initialBackfillBars: Number(process.env.TOPSTEPX_MGC_BACKFILL || '500'),
  dailyLossLimit: Number(process.env.TOPSTEPX_MGC_DAILY_LOSS_LIMIT || '2000'),
};

const STOP_MONITOR_DELAY_MS = Number(process.env.TOPSTEPX_MGC_STOP_MONITOR_MS || '1500');
const DASHBOARD_PORT = Number(process.env.TOPSTEPX_MGC_DASHBOARD_PORT || '3006');
const TOPSTEPX_LIVE_ACCOUNT_ID = process.env.TOPSTEPX_ACCOUNT_ID || process.env.TOPSTEPX_MGC_LIVE_ACCOUNT_ID;
const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market';
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user';

const TRADE_LOG_FILE = process.env.TOPSTEPX_MGC_TRADE_LOG || './logs/topstepx-mgc-po3.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);
const STATE_FILE = './logs/.mgc-po3-state.json';

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;
const REOPEN_MINUTES = 17 * 60;
const WEEKEND_REOPEN_MINUTES = 19 * 60;

// Express and Socket.IO setup
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

let position: ActivePosition | null = null;
let bars: TopstepXFuturesBar[] = [];
let chartHistory: ChartData[] = [];
let multiplier = 10;
let realizedPnL = 0;
let tradingEnabled = false;
let lastProcessedBarTime = '';
let shuttingDown = false;
let tickSize = 0.1;
let commissionPerSide = 0.86;
let accountId = 0;
let contractId: string | null = null;
let resolvedContractId: string | null = null;
let topstepRest: ReturnType<typeof createProjectXRest> | null = null;
let orderManager: TopstepOrderManager | null = null;
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let lastQuotePrice = 0;
let tradeSequence = 0;
let currentBar: TopstepXFuturesBar | null = null;
let barStartTime: Date | null = null;
let lastMarketDataTime: Date | null = null;
let accountStatus: AccountStatus = {
  balance: 0,
  buyingPower: 0,
  dailyPnL: 0,
  openPositions: 0,
  dailyLossLimit: CONFIG.dailyLossLimit,
  isAtRisk: false,
};

// PO3 session tracking
interface DayData {
  tradingDay: string;
  asiaHigh: number | null;
  asiaLow: number | null;
  londonManipulation: 'bullish' | 'bearish' | null;
  londonSweepPrice: number | null;
  londonSweepBarIndex: number | null;
  enteredToday: boolean;
}

let currentDay: DayData = {
  tradingDay: '',
  asiaHigh: null,
  asiaLow: null,
  londonManipulation: null,
  londonSweepPrice: null,
  londonSweepBarIndex: null,
  enteredToday: false,
};

let barIndex = 0;

async function updateAccountStatus() {
  try {
    const accounts = await fetchTopstepXAccounts(true);
    const account = accounts.find(a => a.id === accountId) || accounts[0];
    if (account) {
      accountStatus.balance = account.balance || 0;
      accountStatus.buyingPower = account.buyingPower || 0;
      accountStatus.dailyPnL = account.dailyProfitLoss || 0;
    }
  } catch (err: any) {
    log(`Failed to fetch account status: ${err.message}`);
  }
}

function saveState() {
  try {
    ensureTradeLogDir();
    const state = {
      tradingEnabled,
      position: position ? {
        side: position.side,
        entryPrice: position.entryPrice,
        totalQty: position.totalQty,
        remainingQty: position.remainingQty,
        scaledQty: position.scaledQty,
        stopOrderId: position.stopOrderId,
        targetOrderId: position.targetOrderId,
        entryOrderId: position.entryOrderId,
        stopLoss: position.stopLoss,
        tp1: position.tp1,
        tp2: position.tp2,
        entryTime: position.entryTime,
        tradeId: position.tradeId,
        asiaHigh: position.asiaHigh,
        asiaLow: position.asiaLow,
        londonSweep: position.londonSweep,
        fvgMidpoint: position.fvgMidpoint,
        tp1Hit: position.tp1Hit,
      } : null,
      currentDay,
      timestamp: nowIso()
    };
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (err: any) {
    log(`[STATE] Failed to save state: ${err.message}`);
  }
}

function loadState(): { tradingEnabled: boolean; position: any; currentDay: any } {
  try {
    if (!existsSync(STATE_FILE)) {
      log('[STATE] No previous state found - starting with trading DISABLED');
      return { tradingEnabled: false, position: null, currentDay: null };
    }
    const data = readFileSync(STATE_FILE, 'utf-8');
    const state = JSON.parse(data);
    const wasEnabled = state.tradingEnabled || false;
    const savedPosition = state.position || null;
    const savedDay = state.currentDay || null;

    if (wasEnabled) {
      log(`[STATE] Restored previous state: Trading was ENABLED - auto-resuming`);
    } else {
      log(`[STATE] Restored previous state: Trading was DISABLED`);
    }

    if (savedPosition) {
      log(`[STATE] Found saved position: ${savedPosition.side.toUpperCase()} ${savedPosition.totalQty} @ ${savedPosition.entryPrice}`);
    }

    if (savedDay && savedDay.enteredToday) {
      log(`[STATE] Restored day state: Already entered today (${savedDay.tradingDay})`);
    }

    return { tradingEnabled: wasEnabled, position: savedPosition, currentDay: savedDay };
  } catch (err: any) {
    log(`[STATE] Failed to load state: ${err.message} - defaulting to DISABLED`);
    return { tradingEnabled: false, position: null, currentDay: null };
  }
}

async function reconcilePosition(savedPosition: any) {
  try {
    log('🔄 Checking TopstepX for existing positions...');
    const positions = await topstepRest.getPositions(accountId);

    if (!positions || positions.length === 0) {
      log('✅ No existing positions found - starting fresh');
      position = null;
      return;
    }

    const existingPos = positions.find((p: any) => p.contractId === contractId);

    if (!existingPos) {
      log(`✅ No position found for ${CONFIG.symbol} - starting fresh`);
      position = null;
      return;
    }

    const qty = Math.abs(existingPos.quantity || existingPos.size || 0);
    if (qty === 0) {
      log(`✅ Position quantity is 0 for ${CONFIG.symbol} - starting fresh`);
      position = null;
      return;
    }

    const side = (existingPos.quantity || existingPos.size) > 0 ? 'long' : 'short';
    const avgPrice = existingPos.averagePrice || existingPos.avgPrice || 0;

    log(`⚠️ FOUND EXISTING POSITION: ${side.toUpperCase()} ${qty} @ ${avgPrice.toFixed(2)}`);

    if (savedPosition && savedPosition.side === side && savedPosition.totalQty === qty) {
      log(`✅ Saved position matches broker position - restoring with order IDs`);
      position = {
        tradeId: savedPosition.tradeId,
        symbol: CONFIG.symbol,
        contractId: resolvedContractId ?? '',
        side: savedPosition.side,
        entryPrice: savedPosition.entryPrice,
        entryTime: savedPosition.entryTime,
        stopLoss: savedPosition.stopLoss,
        tp1: savedPosition.tp1,
        tp2: savedPosition.tp2,
        totalQty: savedPosition.totalQty,
        remainingQty: savedPosition.remainingQty,
        scaledQty: savedPosition.scaledQty,
        entryOrderId: savedPosition.entryOrderId,
        stopOrderId: savedPosition.stopOrderId,
        targetOrderId: savedPosition.targetOrderId,
        stopFilled: false,
        targetFilled: false,
        stopLimitPending: savedPosition.stopOrderId ? true : false,
        monitoringStop: false,
        asiaHigh: savedPosition.asiaHigh,
        asiaLow: savedPosition.asiaLow,
        londonSweep: savedPosition.londonSweep,
        fvgMidpoint: savedPosition.fvgMidpoint,
        tp1Hit: savedPosition.tp1Hit || false,
      };
      log(`   Resume monitoring position with full order management`);
      broadcastDashboardUpdate();
      return;
    }

    log(`🔄 No saved position data or mismatch - FLATTENING for safety`);
    if (!orderManager) {
      log(`❌ Cannot flatten - order manager not initialized yet`);
      position = null;
      return;
    }

    const exitSide: OrderSide = side === 'long' ? 'Sell' : 'Buy';
    try {
      await orderManager.placeMarketIOC(exitSide, qty);
      log(`✅ Position flattened successfully @ market`);
    } catch (flattenErr: any) {
      log(`❌ Failed to flatten position: ${flattenErr.message}`);
      log(`⚠️ MANUAL INTERVENTION REQUIRED - position may still be open`);
    }

    position = null;
  } catch (err: any) {
    log(`❌ Failed to reconcile position: ${err.message}`);
    log(`⚠️ Assuming no position and starting fresh`);
    position = null;
  }
}

function nowIso(): string {
  return new Date().toISOString();
}

function log(message: string) {
  const logMessage = `[${nowIso()}][${CONFIG.symbol}] ${message}`;
  console.log(logMessage);

  let type = 'info';
  const msgLower = message.toLowerCase();

  if (msgLower.includes('error') || msgLower.includes('failed') || msgLower.includes('rejected')) {
    type = 'error';
  } else if (msgLower.includes('warning') || msgLower.includes('⚠️') || msgLower.includes('limit')) {
    type = 'warning';
  } else if (msgLower.includes('filled') || msgLower.includes('entered') || msgLower.includes('closed') ||
             msgLower.includes('win') || msgLower.includes('profit')) {
    type = 'success';
  }

  io.emit('log', { timestamp: nowIso(), message, type });
}

function nextTradeId() {
  tradeSequence += 1;
  return `MGC-PO3-${Date.now()}-${tradeSequence}`;
}

function ensureTradeLogDir() {
  if (!existsSync(TRADE_LOG_DIR)) {
    mkdirSync(TRADE_LOG_DIR, { recursive: true });
  }
}

function recalculateAccountStats(targetAccountId: number) {
  try {
    if (!existsSync(TRADE_LOG_FILE)) {
      realizedPnL = 0;
      tradeSequence = 0;
      return;
    }

    const logContent = require('fs').readFileSync(TRADE_LOG_FILE, 'utf-8');
    const lines = logContent.trim().split('\n').filter(line => line.trim());
    const accountTrades = lines
      .map(line => JSON.parse(line))
      .filter(trade => trade.accountId === targetAccountId);

    realizedPnL = accountTrades
      .filter(trade => trade.type === 'exit')
      .reduce((sum, trade) => sum + (trade.pnl || 0), 0);

    tradeSequence = accountTrades
      .filter(trade => trade.type === 'entry')
      .reduce((max, trade) => Math.max(max, trade.tradeId || 0), 0);

    log(`[ACCOUNT] Stats recalculated: ${accountTrades.length} trades, PnL: ${realizedPnL.toFixed(2)}, Last trade ID: ${tradeSequence}`);
  } catch (error) {
    log(`[ACCOUNT] Warning: Could not recalculate stats: ${error}`);
    realizedPnL = 0;
    tradeSequence = 0;
  }
}

function logTradeEvent(event: Record<string, any>) {
  try {
    ensureTradeLogDir();
    const tradeData = { timestamp: nowIso(), ...event };
    appendFileSync(
      TRADE_LOG_FILE,
      `${JSON.stringify(tradeData)}\n`,
    );

    if (event.type === 'exit') {
      const trade = {
        tradeId: event.tradeId,
        side: event.side,
        entryPrice: position?.entryPrice || event.entryPrice,
        exitPrice: event.exitPrice,
        entryTime: position?.entryTime || event.timestamp,
        exitTime: nowIso(),
        quantity: event.qty,
        pnl: event.pnl,
        exitReason: event.reason,
      };
      io.emit('trade', trade);
    }
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

function broadcastDashboardUpdate() {
  const currentPnL = position ? calculatePnL(position.entryPrice, lastQuotePrice, position.side, position.remainingQty + position.scaledQty) : 0;

  io.emit('status', {
    tradingEnabled,
    position: position ? {
      ...position,
      unrealizedPnL: currentPnL,
    } : null,
    currentDay,
    realizedPnL,
    accountStatus,
    lastQuote: lastQuotePrice,
    currentBar,
    timestamp: nowIso(),
  });

  if (position) {
    position.unrealizedPnL = currentPnL;
  }
}

async function monitorStopLimit(stopOrderId: string | number, exitSide: OrderSide, qty: number) {
  if (!position || !orderManager || position.stopFilled || position.monitoringStop) {
    return;
  }

  position.monitoringStop = true;
  log(`[MONITOR] Monitoring stop limit ${stopOrderId} for ${STOP_MONITOR_DELAY_MS}ms`);

  await sleep(STOP_MONITOR_DELAY_MS);

  if (!position || position.stopFilled) {
    log(`[MONITOR] Stop already filled or position closed`);
    return;
  }

  log(`[MONITOR] Stop limit ${stopOrderId} NOT filled - converting to MARKET STOP`);

  try {
    await orderManager.cancelOrder(stopOrderId);
    log(`[MONITOR] Cancelled stop limit ${stopOrderId}`);

    const marketResponse = await orderManager.placeMarketIOC(exitSide, qty);
    log(`[MONITOR] Market stop placed, order ID: ${marketResponse?.orderId ?? 'unknown'}`);

    if (position) {
      position.stopFilled = true;
      const exitPrice = lastQuotePrice || position.stopLoss;
      await handlePositionExit(exitPrice, nowIso(), 'stop', qty, true);
    }
  } catch (err: any) {
    log(`[ERROR] Failed to convert stop limit to market: ${err.message}`);
  } finally {
    if (position) {
      position.monitoringStop = false;
    }
  }
}

function getSessionType(timestamp: string): 'asia' | 'london' | 'ny' | 'other' {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();
  const minutes = etDate.getUTCMinutes();
  const totalMinutes = hours * 60 + minutes;

  if (totalMinutes >= 1200 || totalMinutes < 0) {
    return 'asia';
  }

  if (totalMinutes >= 120 && totalMinutes <= 300) {
    return 'london';
  }

  const nySession1 = totalMinutes >= 570 && totalMinutes <= 690;
  const nySession2 = totalMinutes >= 810 && totalMinutes <= 930;
  if (nySession1 || nySession2) {
    return 'ny';
  }

  return 'other';
}

function getTradingDay(timestamp: string): string {
  const date = new Date(timestamp);
  const etDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);
  const hours = etDate.getUTCHours();

  if (hours < 18) {
    return etDate.toISOString().split('T')[0];
  } else {
    const nextDay = new Date(etDate);
    nextDay.setUTCDate(nextDay.getUTCDate() + 1);
    return nextDay.toISOString().split('T')[0];
  }
}

function calculateATR(bars: TopstepXFuturesBar[], currentIndex: number, period: number = 14): number {
  if (currentIndex < period) return 0;

  let sumTR = 0;
  for (let i = 0; i < period; i++) {
    const idx = currentIndex - i;
    const bar = bars[idx];
    const prevBar = idx > 0 ? bars[idx - 1] : bar;

    const high_low = bar.high - bar.low;
    const high_prevClose = Math.abs(bar.high - prevBar.close);
    const low_prevClose = Math.abs(bar.low - prevBar.close);

    const tr = Math.max(high_low, high_prevClose, low_prevClose);
    sumTR += tr;
  }

  return sumTR / period;
}

interface FVG {
  type: 'bullish' | 'bearish';
  midpoint: number;
  upper: number;
  lower: number;
  barIndex: number;
}

function detectFVG(
  bars: TopstepXFuturesBar[],
  currentIndex: number,
  atr: number,
): FVG | null {
  if (currentIndex < 2) return null;

  const current = bars[currentIndex];
  const twoAgo = bars[currentIndex - 2];

  if (current.low > twoAgo.high) {
    const fvgSize = current.low - twoAgo.high;
    const minSize = Math.max(
      CONFIG.minFVGSizeTicks * tickSize,
      CONFIG.minFVGSizeATR * atr
    );

    if (fvgSize < minSize) return null;

    return {
      type: 'bullish',
      midpoint: (current.low + twoAgo.high) / 2,
      upper: current.low,
      lower: twoAgo.high,
      barIndex: currentIndex,
    };
  }

  if (current.high < twoAgo.low) {
    const fvgSize = twoAgo.low - current.high;
    const minSize = Math.max(
      CONFIG.minFVGSizeTicks * tickSize,
      CONFIG.minFVGSizeATR * atr
    );

    if (fvgSize < minSize) return null;

    return {
      type: 'bearish',
      midpoint: (current.high + twoAgo.low) / 2,
      upper: twoAgo.low,
      lower: current.high,
      barIndex: currentIndex,
    };
  }

  return null;
}

async function enterPosition(
  side: 'long' | 'short',
  fvg: FVG,
  asiaHigh: number,
  asiaLow: number,
  londonSweep: number,
  timestamp: string,
) {
  if (!tradingEnabled) {
    log('Cannot enter: trading is disabled');
    return;
  }

  if (position) {
    log('Cannot enter: position already active');
    return;
  }

  if (!orderManager) {
    log('Order manager not initialized; cannot place entry');
    return;
  }

  if (accountStatus.dailyPnL <= -CONFIG.dailyLossLimit) {
    log(`[SAFETY] Daily loss limit reached (${formatCurrency(accountStatus.dailyPnL)}). Skipping entry.`);
    return;
  }

  const tradeId = nextTradeId();
  const entrySide: OrderSide = side === 'long' ? 'Buy' : 'Sell';
  const exitSide: OrderSide = side === 'long' ? 'Sell' : 'Buy';

  const entryPrice = roundToTick(fvg.midpoint);

  // CRITICAL: Correct stop placement
  const stopLoss = roundToTick(
    side === 'long'
      ? fvg.lower - CONFIG.stopLossBuffer * tickSize  // BELOW entry for longs
      : fvg.upper + CONFIG.stopLossBuffer * tickSize  // ABOVE entry for shorts
  );

  const asiaMid = (asiaHigh + asiaLow) / 2;
  const tp1 = roundToTick(asiaMid);

  const rangeFromMid = side === 'long'
    ? (asiaHigh - asiaMid) * CONFIG.tp2RangePercent
    : (asiaMid - asiaLow) * CONFIG.tp2RangePercent;
  const tp2 = roundToTick(side === 'long' ? asiaMid + rangeFromMid : asiaMid - rangeFromMid);

  log(
    `[ENTRY] Attempting ${side.toUpperCase()} MARKET @ ${entryPrice.toFixed(2)}, ` +
    `Stop @ ${stopLoss.toFixed(2)}, TP1 @ ${tp1.toFixed(2)}, TP2 @ ${tp2.toFixed(2)}`
  );

  let bracketResult;
  try {
    bracketResult = await orderManager.placeBracketEntry(
      entrySide,
      stopLoss,
      tp1,
      CONFIG.numberOfContracts,
    );
  } catch (err: any) {
    log(`[ERROR] Failed to place bracket order: ${err.message}`);
    return;
  }

  position = {
    tradeId,
    symbol: CONFIG.symbol,
    contractId: resolvedContractId ?? '',
    side,
    entryPrice,
    entryTime: timestamp,
    stopLoss,
    tp1,
    tp2,
    totalQty: CONFIG.numberOfContracts,
    remainingQty: CONFIG.numberOfContracts,
    scaledQty: 0,
    entryOrderId: bracketResult.entryOrderId,
    stopOrderId: bracketResult.stopOrderId,
    targetOrderId: bracketResult.targetOrderId,
    stopFilled: bracketResult.stopFilled,
    targetFilled: bracketResult.targetFilled,
    stopLimitPending: !bracketResult.stopFilled,
    monitoringStop: false,
    asiaHigh,
    asiaLow,
    londonSweep,
    fvgMidpoint: fvg.midpoint,
    tp1Hit: false,
  };

  log(
    `ENTERED ${side.toUpperCase()} MARKET @ ~${entryPrice.toFixed(2)} ` +
    `(FVG: ${fvg.midpoint.toFixed(2)}, Asia: ${asiaLow.toFixed(2)}-${asiaHigh.toFixed(2)}, London: ${londonSweep.toFixed(2)})`
  );

  logTradeEvent({
    type: 'entry',
    accountId,
    tradeId,
    side: side.toUpperCase(),
    price: entryPrice,
    orderType: 'MARKET',
    qty: CONFIG.numberOfContracts,
    asiaHigh,
    asiaLow,
    londonSweep,
    fvgMidpoint: fvg.midpoint,
    stopLoss,
    tp1,
    tp2,
    entryOrderId: bracketResult.entryOrderId,
    stopOrderId: bracketResult.stopOrderId,
    targetOrderId: bracketResult.targetOrderId,
  });

  if (chartHistory.length > 0) {
    chartHistory[chartHistory.length - 1].entry = true;
  }

  currentDay.enteredToday = true;
  saveState();
  broadcastDashboardUpdate();

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
  reason: 'tp1' | 'tp2' | 'stop' | 'end_of_session' | 'manual' | 'daily_loss_limit',
  qty: number,
  isMarketStop: boolean = false,
) {
  if (!position) {
    return;
  }

  const actualEntryFee = position.entryCommission ?? (commissionPerSide * position.totalQty);
  const actualExitFee = position.exitCommission ?? (commissionPerSide * qty);
  const fees = actualEntryFee + actualExitFee;
  const isActualFees = position.entryCommission !== undefined && position.exitCommission !== undefined;

  const pnl = calculatePnL(position.entryPrice, price, position.side, qty);
  realizedPnL += pnl;

  const durationSeconds = (new Date(timestamp).getTime() - new Date(position.entryTime).getTime()) / 1000;

  log(
    `EXITED ${position.side.toUpperCase()} @ ${price.toFixed(2)} (${reason}${isMarketStop ? ' - MARKET' : ''}) ` +
    `| Qty: ${qty} | PnL: ${formatCurrency(pnl)} | Fees: ${formatCurrency(fees)}${isActualFees ? ' (actual)' : ' (est.)'} | Duration: ${durationSeconds.toFixed(0)}s | Cumulative: ${formatCurrency(realizedPnL)}`
  );

  logTradeEvent({
    type: 'exit',
    accountId,
    tradeId: position.tradeId,
    side: position.side.toUpperCase(),
    entryPrice: position.entryPrice,
    exitPrice: price,
    reason,
    isMarketStop,
    qty,
    pnl,
    durationSeconds,
    cumulativePnL: realizedPnL,
    fees,
    feesActual: isActualFees,
    entryCommission: position.entryCommission,
    exitCommission: position.exitCommission,
    stopOrderId: position.stopOrderId,
    targetOrderId: position.targetOrderId,
  });

  if (chartHistory.length > 0) {
    chartHistory[chartHistory.length - 1].exit = true;
  }

  if (qty >= position.remainingQty) {
    position = null;
    saveState();
  } else {
    position.remainingQty -= qty;
    saveState();
  }

  broadcastDashboardUpdate();
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
  lastMarketDataTime = new Date();
  broadcastDashboardUpdate();

  const timestamp = new Date(quote.timestamp || quote.lastTradeTimestamp || Date.now());

  const barMinute = new Date(timestamp);
  barMinute.setSeconds(0, 0);

  if (!barStartTime || barStartTime.getTime() !== barMinute.getTime()) {
    if (currentBar) {
      processBar(currentBar);
    }

    barStartTime = barMinute;
    currentBar = {
      timestamp: barMinute.toISOString(),
      open: price,
      high: price,
      low: price,
      close: price,
    };
  } else if (currentBar) {
    currentBar.high = Math.max(currentBar.high, price);
    currentBar.low = Math.min(currentBar.low, price);
    currentBar.close = price;
  }

  if (currentBar) {
    io.emit('tick', {
      timestamp: currentBar.timestamp,
      open: currentBar.open,
      high: currentBar.high,
      low: currentBar.low,
      close: currentBar.close,
    });
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

  marketHub.onreconnected(() => {
    log('⚠️ TopstepX market hub RECONNECTED - resubscribing to market data');
    subscribeMarket();
  });

  marketHub.onreconnecting((error) => {
    log(`⚠️ TopstepX market hub connection lost, attempting to reconnect... ${error?.message || ''}`);
  });

  marketHub.onclose((error) => {
    log(`❌ TopstepX market hub connection CLOSED: ${error?.message || 'Unknown reason'}`);
    log('⚠️ Live market data streaming has stopped. Restart server to reconnect.');
  });

  await marketHub.start();
  log('✅ TopstepX market hub connected');
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

    log(`[FILL EVENT] ${JSON.stringify(ev)}`);

    const side = ev.side === 0 ? 'Buy' : 'Sell';
    const qty = Math.abs(Number(ev.size ?? ev.quantity ?? ev.qty ?? 0));
    const price = Number(ev.price ?? ev.avgPrice ?? 0);
    const commission = ev.commission ?? ev.fee ?? null;

    if (qty && price) {
      log(`User trade ${side} ${qty}@${price.toFixed(2)}${commission !== null ? ` | Fee: ${commission}` : ''}`);

      if (position) {
        const orderId = ev.orderId ?? ev.id;

        if (orderId === position.entryOrderId) {
          position.entryPrice = price;
          if (commission !== null) {
            position.entryCommission = commission;
          }
          log(`Entry filled @ ${price.toFixed(2)}${position.entryCommission ? ` | Entry fee: ${position.entryCommission}` : ' (market order)'}`);
          broadcastDashboardUpdate();
        } else if (orderId === position.stopOrderId) {
          position.stopFilled = true;
          if (commission !== null) {
            position.exitCommission = commission;
          }
          handlePositionExit(price, nowIso(), 'stop', position.remainingQty + position.scaledQty, false);
        } else if (orderId === position.targetOrderId) {
          if (!position.tp1Hit) {
            // TP1 hit (scale out)
            position.tp1Hit = true;
            const scaleQty = Math.floor(CONFIG.numberOfContracts * CONFIG.scaleOutPercent);
            position.scaledQty = scaleQty;
            position.remainingQty = CONFIG.numberOfContracts - scaleQty;
            if (commission !== null) {
              position.exitCommission = commission;
            }
            log(`TP1 (Asia mid) filled @ ${price.toFixed(2)} - scaled ${scaleQty} contracts`);

            // Move stop to breakeven
            position.stopLoss = position.entryPrice;
            log(`Stop moved to breakeven @ ${position.stopLoss.toFixed(2)}`);

            // Cancel old stop order and place new one at breakeven
            if (orderManager && position.stopOrderId) {
              orderManager.cancelOrder(position.stopOrderId).then(() => {
                const exitSide: OrderSide = position!.side === 'long' ? 'Sell' : 'Buy';
                return orderManager!.placeLimitIOC(exitSide, position!.remainingQty, position!.stopLoss);
              }).then(response => {
                if (position) {
                  position.stopOrderId = response?.orderId ?? response?.id;
                  log(`New breakeven stop placed: ${position.stopOrderId}`);
                }
              }).catch(err => {
                log(`[ERROR] Failed to update stop to breakeven: ${err.message}`);
              });
            }

            // Update target order to TP2
            if (orderManager && position.targetOrderId) {
              orderManager.cancelOrder(position.targetOrderId).then(() => {
                const exitSide: OrderSide = position!.side === 'long' ? 'Sell' : 'Buy';
                return orderManager!.placeLimitIOC(exitSide, position!.remainingQty, position!.tp2);
              }).then(response => {
                if (position) {
                  position.targetOrderId = response?.orderId ?? response?.id;
                  log(`TP2 order placed: ${position.targetOrderId} @ ${position.tp2.toFixed(2)}`);
                }
              }).catch(err => {
                log(`[ERROR] Failed to place TP2 order: ${err.message}`);
              });
            }

            handlePositionExit(price, nowIso(), 'tp1', scaleQty, false);
            saveState();
          } else {
            // TP2 hit (exit remaining)
            position.targetFilled = true;
            if (commission !== null) {
              position.exitCommission = commission;
            }
            handlePositionExit(price, nowIso(), 'tp2', position.remainingQty, false);
          }
        }
      }
    }
  });

  userHub.on('GatewayUserOrder', data => {
    log(`User order event: ${JSON.stringify(data)}`);
  });

  userHub.on('GatewayUserAccount', (_cid: string, data: any) => {
    if (data) {
      accountStatus = {
        balance: data.cashBalance || data.balance || 0,
        buyingPower: data.buyingPower || data.availableBalance || 0,
        dailyPnL: data.dailyNetPnL || data.dailyPnl || 0,
        openPositions: data.openPositions || 0,
        dailyLossLimit: CONFIG.dailyLossLimit,
        isAtRisk: false,
      };

      if (accountStatus.dailyPnL <= -CONFIG.dailyLossLimit * 0.8) {
        accountStatus.isAtRisk = true;
        log(`[WARNING] Approaching daily loss limit: ${formatCurrency(accountStatus.dailyPnL)}`);
      }

      if (accountStatus.dailyPnL <= -CONFIG.dailyLossLimit) {
        log(`[SAFETY] Daily loss limit exceeded: ${formatCurrency(accountStatus.dailyPnL)}`);
        if (position && orderManager) {
          const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
          orderManager.placeMarketIOC(exitSide, position.remainingQty + position.scaledQty).catch(err =>
            log(`[ERROR] Failed to flatten position: ${err.message}`)
          );
        }
        handlePositionExit(lastQuotePrice, nowIso(), 'daily_loss_limit', position?.remainingQty || 0);
        shutdown('daily_loss_limit');
      }

      broadcastDashboardUpdate();
    }
  });

  userHub.on('GatewayUserPosition', (_cid: string, data: any) => {
    if (data) {
      log(`[position] ${JSON.stringify(data)}`);

      const brokerQty = Math.abs(data.quantity || 0);

      if (position && brokerQty === 0) {
        log('[SYNC] Position closed externally - syncing local state');
        position = null;
        broadcastDashboardUpdate();
      }
    }
  });

  const subscribeUser = () => {
    if (!userHub) return;
    userHub.invoke('SubscribeAccounts').catch(err => console.error('[user] Subscribe accounts failed', err));
    userHub.invoke('SubscribeOrders', accountId).catch(err => console.error('[user] Subscribe orders failed', err));
    userHub.invoke('SubscribePositions', accountId).catch(err => console.error('[user] Subscribe positions failed', err));
    userHub.invoke('SubscribeTrades', accountId).catch(err => console.error('[user] Subscribe trades failed', err));
  };

  userHub.onreconnected(() => {
    log('⚠️ TopstepX user hub RECONNECTED - resubscribing to account data');
    subscribeUser();
  });

  userHub.onreconnecting((error) => {
    log(`⚠️ TopstepX user hub connection lost, attempting to reconnect... ${error?.message || ''}`);
  });

  userHub.onclose((error) => {
    log(`❌ TopstepX user hub connection CLOSED: ${error?.message || 'Unknown reason'}`);
    log('⚠️ Live account/order/position updates have stopped. Restart server to reconnect.');
  });

  await userHub.start();
  log('✅ TopstepX user hub connected');
  subscribeUser();
}

async function processBar(bar: TopstepXFuturesBar) {
  if (bar.timestamp === lastProcessedBarTime) {
    return;
  }
  lastProcessedBarTime = bar.timestamp;

  bars.push(bar);
  barIndex++;

  const maxHistory = 1000;
  if (bars.length > maxHistory) {
    bars = bars.slice(-maxHistory);
  }

  const session = getSessionType(bar.timestamp);
  const tradingDay = getTradingDay(bar.timestamp);

  if (tradingDay !== currentDay.tradingDay) {
    currentDay = {
      tradingDay,
      asiaHigh: null,
      asiaLow: null,
      londonManipulation: null,
      londonSweepPrice: null,
      londonSweepBarIndex: null,
      enteredToday: false,
    };
    saveState();
  }

  const atr = calculateATR(bars, bars.length - 1, 14);

  const chartPoint: ChartData = {
    timestamp: bar.timestamp,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    asiaHigh: currentDay.asiaHigh || undefined,
    asiaLow: currentDay.asiaLow || undefined,
    asiaMidpoint: (currentDay.asiaHigh && currentDay.asiaLow) ? (currentDay.asiaHigh + currentDay.asiaLow) / 2 : undefined,
    londonSweep: currentDay.londonSweepPrice || undefined,
    manipulation: currentDay.londonManipulation,
    signal: null,
  };

  chartHistory.push(chartPoint);

  if (chartHistory.length > 500) {
    chartHistory = chartHistory.slice(-500);
  }

  io.emit('bar', chartPoint);

  // Manage existing position
  if (position) {
    const direction = position.side === 'long' ? 1 : -1;

    // Check TP1 (only if not already hit)
    if (!position.tp1Hit && position.remainingQty === CONFIG.numberOfContracts) {
      const hitTP1 = (direction === 1 && bar.high >= position.tp1) ||
                     (direction === -1 && bar.low <= position.tp1);

      if (hitTP1) {
        // TP1 will be handled by user trade event
        log(`[TP1] Asia midpoint reached @ ${bar.close.toFixed(2)} - waiting for fill confirmation`);
      }
    }

    // Check TP2
    if (position.tp1Hit && position.remainingQty > 0) {
      const hitTP2 = (direction === 1 && bar.high >= position.tp2) ||
                     (direction === -1 && bar.low <= position.tp2);

      if (hitTP2) {
        // TP2 will be handled by user trade event
        log(`[TP2] Asia range target reached @ ${bar.close.toFixed(2)} - waiting for fill confirmation`);
      }
    }

    // Check stop loss
    if (position.remainingQty > 0) {
      const hitStop = (direction === 1 && bar.low <= position.stopLoss) ||
                     (direction === -1 && bar.high >= position.stopLoss);

      if (hitStop) {
        // Stop will be handled by user trade event
        log(`[STOP] Stop level reached @ ${bar.close.toFixed(2)} - waiting for fill confirmation`);
      }
    }

    if (shouldFlattenForClose(new Date(bar.timestamp))) {
      log('Flattening position for end of session');
      if (orderManager) {
        const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
        try {
          await orderManager.placeMarketIOC(exitSide, position.remainingQty + position.scaledQty);
        } catch (err: any) {
          log(`[ERROR] Session flatten failed: ${err.message}`);
        }
      }
      await handlePositionExit(bar.close, bar.timestamp, 'end_of_session', position.remainingQty + position.scaledQty);
      return;
    }

    if (!isTradingAllowed(bar.timestamp)) {
      log('Closing position - outside trading hours');
      if (orderManager) {
        const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
        try {
          await orderManager.placeMarketIOC(exitSide, position.remainingQty + position.scaledQty);
        } catch (err: any) {
          log(`[ERROR] Hours close failed: ${err.message}`);
        }
      }
      await handlePositionExit(bar.close, bar.timestamp, 'end_of_session', position.remainingQty + position.scaledQty);
    }

    return;
  }

  // Track Asia range (Accumulation)
  if (session === 'asia') {
    if (currentDay.asiaHigh === null || bar.high > currentDay.asiaHigh) {
      currentDay.asiaHigh = bar.high;
      saveState();
    }
    if (currentDay.asiaLow === null || bar.low < currentDay.asiaLow) {
      currentDay.asiaLow = bar.low;
      saveState();
    }
  }

  // Detect London manipulation (with filters)
  if (session === 'london' && currentDay.asiaHigh !== null && currentDay.asiaLow !== null && atr > 0) {
    const asiaRange = currentDay.asiaHigh - currentDay.asiaLow;
    const minAsiaRange = CONFIG.minAsiaRangeATR * atr;

    if (asiaRange < minAsiaRange) {
      return;
    }

    if (!currentDay.londonManipulation && bar.high > currentDay.asiaHigh) {
      const sweepSize = bar.high - currentDay.asiaHigh;
      const minSweep = CONFIG.minSweepTicks * tickSize;

      if (sweepSize >= minSweep) {
        currentDay.londonManipulation = 'bearish';
        currentDay.londonSweepPrice = bar.high;
        currentDay.londonSweepBarIndex = barIndex;
        saveState();
        log(
          `[LONDON] BEARISH MANIPULATION @ ${bar.high.toFixed(2)} ` +
          `(swept Asia high: ${currentDay.asiaHigh.toFixed(2)} by ${sweepSize.toFixed(2)}, Asia range: ${asiaRange.toFixed(2)})`
        );
      }
    } else if (!currentDay.londonManipulation && bar.low < currentDay.asiaLow) {
      const sweepSize = currentDay.asiaLow - bar.low;
      const minSweep = CONFIG.minSweepTicks * tickSize;

      if (sweepSize >= minSweep) {
        currentDay.londonManipulation = 'bullish';
        currentDay.londonSweepPrice = bar.low;
        currentDay.londonSweepBarIndex = barIndex;
        saveState();
        log(
          `[LONDON] BULLISH MANIPULATION @ ${bar.low.toFixed(2)} ` +
          `(swept Asia low: ${currentDay.asiaLow.toFixed(2)} by ${sweepSize.toFixed(2)}, Asia range: ${asiaRange.toFixed(2)})`
        );
      }
    }
  }

  // Trade NY distribution (back into Asia range)
  if (session === 'ny' &&
      currentDay.asiaHigh !== null &&
      currentDay.asiaLow !== null &&
      currentDay.londonManipulation !== null &&
      currentDay.londonSweepPrice !== null &&
      currentDay.londonSweepBarIndex !== null &&
      !currentDay.enteredToday &&
      tradingEnabled &&
      atr > 0) {

    const barsAfterSweep = barIndex - currentDay.londonSweepBarIndex;
    if (barsAfterSweep < CONFIG.minBarsAfterSweep) {
      return;
    }

    const fvg = detectFVG(bars, bars.length - 1, atr);

    if (fvg) {
      // Long setup: After bullish London manipulation, look for bullish FVG in NY
      if (currentDay.londonManipulation === 'bullish' && fvg.type === 'bullish') {
        const entryPrice = roundToTick(fvg.midpoint);

        if (entryPrice >= currentDay.asiaHigh) {
          return; // Entry too high
        }

        chartPoint.signal = 'long';
        await enterPosition(
          'long',
          fvg,
          currentDay.asiaHigh,
          currentDay.asiaLow,
          currentDay.londonSweepPrice,
          bar.timestamp
        );
        currentDay.londonManipulation = null;
        saveState();
      }
      // Short setup: After bearish London manipulation, look for bearish FVG in NY
      else if (currentDay.londonManipulation === 'bearish' && fvg.type === 'bearish') {
        const entryPrice = roundToTick(fvg.midpoint);

        if (entryPrice <= currentDay.asiaLow) {
          return; // Entry too low
        }

        chartPoint.signal = 'short';
        await enterPosition(
          'short',
          fvg,
          currentDay.asiaHigh,
          currentDay.asiaLow,
          currentDay.londonSweepPrice,
          bar.timestamp
        );
        currentDay.londonManipulation = null;
        saveState();
      }
    }
  }
}

async function bootstrapHistoricalData(contractId: string) {
  log(`Fetching initial ${CONFIG.initialBackfillBars} 1-minute bars for bootstrap...`);

  const initialBars = (await fetchTopstepXFuturesBars({
    contractId,
    startTime: new Date(Date.now() - CONFIG.initialBackfillBars * 60 * 1000).toISOString(),
    endTime: new Date().toISOString(),
    unit: 2,
    unitNumber: 1,
    limit: CONFIG.initialBackfillBars,
  })).reverse();

  log(`Loaded ${initialBars.length} initial bars`);

  for (const bar of initialBars) {
    bars.push(bar);
    barIndex++;

    const session = getSessionType(bar.timestamp);
    const tradingDay = getTradingDay(bar.timestamp);

    if (tradingDay !== currentDay.tradingDay) {
      currentDay = {
        tradingDay,
        asiaHigh: null,
        asiaLow: null,
        londonManipulation: null,
        londonSweepPrice: null,
        londonSweepBarIndex: null,
        enteredToday: false,
      };
    }

    if (session === 'asia') {
      if (currentDay.asiaHigh === null || bar.high > currentDay.asiaHigh) {
        currentDay.asiaHigh = bar.high;
      }
      if (currentDay.asiaLow === null || bar.low < currentDay.asiaLow) {
        currentDay.asiaLow = bar.low;
      }
    }

    chartHistory.push({
      timestamp: bar.timestamp,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      asiaHigh: currentDay.asiaHigh || undefined,
      asiaLow: currentDay.asiaLow || undefined,
      asiaMidpoint: (currentDay.asiaHigh && currentDay.asiaLow) ? (currentDay.asiaHigh + currentDay.asiaLow) / 2 : undefined,
      londonSweep: currentDay.londonSweepPrice || undefined,
      manipulation: currentDay.londonManipulation,
      signal: null,
    });
  }

  if (initialBars.length > 0) {
    lastProcessedBarTime = initialBars[initialBars.length - 1].timestamp;
  }

  log(`Bootstrap complete. ${chartHistory.length} bars loaded`);

  io.emit('chartHistory', chartHistory);
}

// Express routes
app.use(cors());
app.use(express.static('public'));
app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'mgc-po3-dashboard.html'));
});

app.get('/api/status', (req, res) => {
  res.json({
    strategy: 'MGC_PO3',
    symbol: CONFIG.symbol,
    status: position ? 'IN_POSITION' : currentDay.londonManipulation ? 'MANIPULATION_DETECTED' : 'SCANNING',
    tradingEnabled,
    position,
    currentDay,
    accountStatus,
    performance: {
      realizedPnL,
      tradesCount: tradeSequence,
      lastQuote: lastQuotePrice,
    },
    config: {
      contracts: CONFIG.numberOfContracts,
      scaleOut: CONFIG.scaleOutPercent * 100,
      stopBuffer: CONFIG.stopLossBuffer,
      minAsiaRangeATR: CONFIG.minAsiaRangeATR,
      minFVGSizeTicks: CONFIG.minFVGSizeTicks,
      minSweepTicks: CONFIG.minSweepTicks,
      tp2RangePercent: CONFIG.tp2RangePercent * 100,
    },
    timestamp: nowIso(),
  });
});

app.get('/api/chart', (req, res) => {
  res.json(chartHistory);
});

app.get('/api/trades', (req, res) => {
  try {
    const logContent = require('fs').readFileSync(TRADE_LOG_FILE, 'utf-8');
    const lines = logContent.trim().split('\n');
    const allTrades = lines
      .filter(line => line.trim())
      .map(line => JSON.parse(line));

    const accountTrades = allTrades
      .filter(trade => trade.accountId === accountId)
      .slice(-100);

    res.json(accountTrades);
  } catch (err) {
    res.json([]);
  }
});

app.get('/api/accounts', async (req, res) => {
  try {
    const accounts = await fetchTopstepXAccounts(true);
    const mappedAccounts = accounts.map(acc => ({
      id: acc.id,
      name: acc.name || `Account ${acc.id}`,
      balance: acc.balance || 0,
      canTrade: acc.canTrade !== false,
      isVisible: true,
      isCurrent: acc.id === accountId,
    }));
    res.json(mappedAccounts);
  } catch (error) {
    log(`[ERROR] Failed to fetch accounts: ${error}`);
    res.json([{
      id: accountId,
      name: `Account ${accountId}`,
      balance: accountStatus.balance,
      canTrade: true,
      isVisible: true,
      isCurrent: true,
    }]);
  }
});

app.post('/api/account/:id', async (req, res) => {
  const selectedAccountId = parseInt(req.params.id);

  if (!selectedAccountId) {
    return res.status(400).json({ success: false, message: 'Invalid account ID' });
  }

  if (selectedAccountId === accountId) {
    return res.json({ success: true, message: `Already using account ${selectedAccountId}` });
  }

  if (tradingEnabled) {
    return res.status(400).json({
      success: false,
      message: 'Cannot switch accounts while trading is active. Please stop trading first.'
    });
  }

  if (position) {
    return res.status(400).json({
      success: false,
      message: 'Cannot switch accounts with an open position. Please close position first.'
    });
  }

  try {
    const accounts = await fetchTopstepXAccounts(true);
    const targetAccount = accounts.find(a => a.id === selectedAccountId);

    if (!targetAccount) {
      return res.status(404).json({ success: false, message: `Account ${selectedAccountId} not found` });
    }

    log(`[ACCOUNT] Switching account: ${accountId} -> ${selectedAccountId}`);

    if (!topstepRest || !contractId) {
      throw new Error('System not fully initialized. Please wait for startup to complete.');
    }

    accountId = selectedAccountId;
    orderManager = new TopstepOrderManager(topstepRest, accountId, contractId, tickSize);
    log(`[ACCOUNT] Order manager recreated for account ${accountId}`);

    await updateAccountStatus();
    recalculateAccountStats(accountId);
    broadcastDashboardUpdate();

    log(`[ACCOUNT] Successfully switched to account ${selectedAccountId}`);

    res.json({
      success: true,
      message: `Successfully switched to account ${selectedAccountId} (${targetAccount.name})`
    });
  } catch (error) {
    log(`[ERROR] Failed to switch account: ${error}`);
    res.status(500).json({ success: false, message: 'Failed to switch account' });
  }
});

app.post('/api/trading/start', (req, res) => {
  tradingEnabled = true;
  saveState();
  log(`[CONTROL] Trading STARTED via dashboard | Account: ${accountId}`);
  broadcastDashboardUpdate();
  res.json({ success: true, message: 'Trading started' });
});

app.post('/api/trading/stop', (req, res) => {
  tradingEnabled = false;
  saveState();
  log(`[CONTROL] Trading STOPPED via dashboard | Account: ${accountId}`);
  broadcastDashboardUpdate();
  res.json({ success: true, message: 'Trading stopped' });
});

app.post('/api/position/flatten', async (req, res) => {
  if (!position) {
    return res.json({ success: false, message: 'No position to flatten' });
  }

  log(`[CONTROL] Flatten position requested via dashboard | Account: ${accountId}`);

  if (orderManager) {
    const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
    try {
      await orderManager.placeMarketIOC(exitSide, position.remainingQty + position.scaledQty);
      await handlePositionExit(lastQuotePrice || position.entryPrice, nowIso(), 'manual', position.remainingQty + position.scaledQty);
      res.json({ success: true, message: 'Position flattened' });
    } catch (err: any) {
      log(`[ERROR] Failed to flatten position: ${err.message}`);
      res.json({ success: false, message: err.message });
    }
  } else {
    res.json({ success: false, message: 'Order manager not initialized' });
  }
});

// Socket.IO events
io.on('connection', (socket) => {
  log(`Dashboard client connected: ${socket.id}`);

  socket.emit('config', {
    symbol: CONFIG.symbol,
    contracts: CONFIG.numberOfContracts,
    scaleOut: CONFIG.scaleOutPercent * 100,
    stopBuffer: CONFIG.stopLossBuffer,
  });

  if (chartHistory.length > 0) {
    log(`[SOCKET] Sending ${chartHistory.length} bars to dashboard client ${socket.id}`);
    socket.emit('chartHistory', chartHistory);
  }
  broadcastDashboardUpdate();

  socket.on('chartHistory', () => {
    socket.emit('chartHistory', chartHistory);
  });

  socket.on('disconnect', () => {
    log(`Dashboard client disconnected: ${socket.id}`);
  });
});

async function shutdown(reason: string) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;

  log(`Shutting down (${reason})...`);

  if (position && orderManager) {
    const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
    try {
      await orderManager.placeMarketIOC(exitSide, position.remainingQty + position.scaledQty);
      await handlePositionExit(lastQuotePrice || position.entryPrice, nowIso(), 'manual', position.remainingQty + position.scaledQty);
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
  console.log('TOPSTEPX LIVE MGC POWER OF THREE (PO3) STRATEGY');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`Sessions: Asia (20:00-00:00), London (02:00-05:00), NY (09:30-11:30, 13:30-15:30 ET)`);
  console.log(`Contracts: ${CONFIG.numberOfContracts} | Scale out: ${CONFIG.scaleOutPercent * 100}% @ TP1`);
  console.log(`Stop Buffer: ${CONFIG.stopLossBuffer} ticks | TP2: ${CONFIG.tp2RangePercent * 100}% of Asia range`);
  console.log(`Daily Loss Limit: $${CONFIG.dailyLossLimit}`);
  console.log(`Dashboard: http://localhost:${DASHBOARD_PORT}`);
  console.log('='.repeat(80));

  log('Main function started.');

  const savedState = loadState();
  tradingEnabled = savedState.tradingEnabled;
  const savedPosition = savedState.position;
  if (savedState.currentDay) {
    currentDay = savedState.currentDay;
  }

  server.listen(DASHBOARD_PORT, () => {
    log(`Dashboard server running on http://localhost:${DASHBOARD_PORT}`);
    if (!tradingEnabled) {
      log(`⚠️ Trading is DISABLED. Use dashboard to start trading.`);
    }
  });

  log('Resolving contract metadata...');
  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey);

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  contractId = metadata.id;
  multiplier = metadata.tickValue && metadata.tickSize
    ? metadata.tickValue / metadata.tickSize
    : metadata.multiplier || 10;

  resolvedContractId = contractId;

  log(`Resolved contract: ${metadata.name} (${contractId})`);
  log(`Point multiplier: ${multiplier}`);

  tickSize = metadata.tickSize;
  if (!tickSize || !Number.isFinite(tickSize) || tickSize <= 0) {
    throw new Error(`Unable to resolve tick size`);
  }

  commissionPerSide = process.env.TOPSTEPX_MGC_LIVE_COMMISSION
    ? Number(process.env.TOPSTEPX_MGC_LIVE_COMMISSION)
    : inferFuturesCommissionPerSide([CONFIG.contractId, CONFIG.symbol, metadata.id], 0.86);

  log(`Tick size: ${tickSize}`);
  log(`Commission/side: ${commissionPerSide.toFixed(2)} USD`);

  log('Resolving account ID...');
  accountId = await resolveAccountId();
  log('Creating ProjectX REST client...');
  topstepRest = createProjectXRest();
  orderManager = new TopstepOrderManager(topstepRest, accountId, contractId, tickSize);
  log(`Using TopstepX account ${accountId}`);

  await bootstrapHistoricalData(contractId);

  await updateAccountStatus();
  log(`Account balance: $${accountStatus.balance.toFixed(2)}`);

  await reconcilePosition(savedPosition);

  setInterval(async () => {
    await updateAccountStatus();
    broadcastDashboardUpdate();

    const now = new Date();
    const dataStale = lastMarketDataTime && (now.getTime() - lastMarketDataTime.getTime()) > 120000;
    const noDataYet = !lastMarketDataTime;

    let statusText: string;
    if (noDataYet || dataStale) {
      statusText = '⚠️ NO MARKET DATA - STRATEGY CANNOT RUN';
    } else if (tradingEnabled) {
      statusText = '✅ RUNNING';
    } else {
      statusText = '⏸ PAUSED';
    }

    const posText = position ? `| Position: ${position.side.toUpperCase()} ${position.remainingQty + position.scaledQty}` : '| No position';
    const dayText = currentDay.enteredToday ? '| Entered today' : currentDay.londonManipulation ? `| ${currentDay.londonManipulation} manip` : '';
    log(`🚀 Strategy ${statusText} | Symbol: ${CONFIG.symbol} | Account: ${accountId} ${posText} ${dayText}`);
  }, 30000);

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

  log('='.repeat(80));
  log('🚀 STRATEGY FULLY INITIALIZED AND RUNNING');
  log(`   Symbol: ${CONFIG.symbol} | Account: ${accountId}`);
  log(`   Trading: ${tradingEnabled ? 'ENABLED ✅' : 'DISABLED ⏸ (use dashboard to start)'}`);
  log(`   Dashboard: http://localhost:${DASHBOARD_PORT}`);
  if (position) {
    log(`   Active Position: ${position.side.toUpperCase()} ${position.remainingQty + position.scaledQty} @ ${position.entryPrice.toFixed(2)}`);
  }
  if (currentDay.londonManipulation) {
    log(`   London ${currentDay.londonManipulation.toUpperCase()} manipulation detected @ ${currentDay.londonSweepPrice?.toFixed(2)}`);
  }
  log('='.repeat(80));
  await new Promise(() => {});
}

main().catch(err => {
  console.error('TopstepX MGC PO3 strategy failed:', err);
  process.exit(1);
});
