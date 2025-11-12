#!/usr/bin/env tsx
/**
 * TopstepX Live NQ Winner Strategy - ENHANCED VERSION
 *
 * Features:
 * - Proper historical data bootstrap with all indicators
 * - Full dashboard with TradingView chart
 * - Account monitoring and safety limits
 * - WebSocket for real-time updates
 * - REST API status endpoints
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
  bbUpper?: number;
  bbMiddle?: number;
  bbBasis?: number;  // For dashboard compatibility
  bbLower?: number;
  rsi?: number;
  ttmMomentum?: number;  // Flattened for dashboard
  squeeze?: {
    momentum: number;
    squeezeFiring: boolean;
  };
  signal?: 'long' | 'short' | null;
  entry?: boolean;
  exit?: boolean;
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
  stopLimitPending: boolean;
  monitoringStop: boolean;
  unrealizedPnL?: number;
  entryCommission?: number;
  exitCommission?: number;
}

interface PendingSetup {
  side: 'long' | 'short';
  setupTime: string;
  setupPrice: number;
  rsi: number;
  bb: { upper: number; middle: number; lower: number };
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

// =============================================================================
// PERMANENT CONDITION LOGGING SYSTEM
// =============================================================================
// All condition logs are saved permanently to help diagnose issues and improve the strategy

function logCondition(message: string, logFile = 'logs/conditions.log') {
  const timestamp = nowIso();
  const logMsg = `[${timestamp}] ${message}`;
  console.log(logMsg);
  try {
    require('fs').appendFileSync(logFile, logMsg + '\n');
  } catch (e) {
    console.error('[CONDITION LOG FAILED]', e);
  }
}

function logIndividualConditions(
  symbol: string,
  price: number,
  rsi: number | undefined,
  bb: { upper: number; middle: number; lower: number } | undefined,
  ttmSqueeze: { squeezeOn: boolean; momentum: number } | null
) {
  const timestamp = nowIso();
  const checks: string[] = [];

  // RSI condition checks
  if (rsi !== undefined) {
    if (rsi < 30) {
      checks.push(`RSI OVERSOLD: ${rsi.toFixed(2)} < 30`);
    } else if (rsi > 70) {
      checks.push(`RSI OVERBOUGHT: ${rsi.toFixed(2)} > 70`);
    }
  }

  // Bollinger Band condition checks
  if (bb) {
    const distanceToLower = ((price - bb.lower) / bb.lower * 100).toFixed(4);
    const distanceToUpper = ((price - bb.upper) / bb.upper * 100).toFixed(4);

    if (price <= bb.lower) {
      checks.push(`BB LOWER TOUCHED/BROKE: Price ${price.toFixed(2)} <= ${bb.lower.toFixed(2)} (${distanceToLower}%)`);
    } else if (price >= bb.upper) {
      checks.push(`BB UPPER TOUCHED/BROKE: Price ${price.toFixed(2)} >= ${bb.upper.toFixed(2)} (${distanceToUpper}%)`);
    }
  }

  // TTM Squeeze state
  if (ttmSqueeze) {
    if (ttmSqueeze.squeezeOn) {
      checks.push(`TTM SQUEEZE FIRING: Momentum ${ttmSqueeze.momentum.toFixed(2)}`);
    }
  }

  // Only log if any conditions are met
  if (checks.length > 0) {
    const msg = `[${symbol}] ${checks.join(' | ')}`;
    logCondition(msg, 'logs/individual-conditions.log');
  }
}

function logSetupProgress(
  symbol: string,
  setupType: 'LONG' | 'SHORT',
  step: 'STEP1_RSI_BB' | 'STEP2_WAITING_TTM' | 'STEP3_TTM_FIRED' | 'ENTRY',
  details: string
) {
  const msg = `[${symbol}] ${setupType} ${step}: ${details}`;
  logCondition(msg, 'logs/setup-progression.log');
}

const CONFIG: StrategyConfig = {
  symbol: process.env.TOPSTEPX_NQ_LIVE_SYMBOL || 'NQZ5',
  contractId: process.env.TOPSTEPX_NQ_LIVE_CONTRACT_ID,
  bbPeriod: Number(process.env.TOPSTEPX_NQ_BB_PERIOD || '20'),
  bbStdDev: Number(process.env.TOPSTEPX_NQ_BB_STDDEV || '3'),
  rsiPeriod: Number(process.env.TOPSTEPX_NQ_RSI_PERIOD || '24'),
  rsiOversold: Number(process.env.TOPSTEPX_NQ_RSI_OVERSOLD || '30'),
  rsiOverbought: Number(process.env.TOPSTEPX_NQ_RSI_OVERBOUGHT || '70'),
  stopLossPercent: Number(process.env.TOPSTEPX_NQ_STOP_PERCENT || '0.0001'),
  takeProfitPercent: Number(process.env.TOPSTEPX_NQ_TP_PERCENT || '0.0005'),
  numberOfContracts: Number(process.env.TOPSTEPX_NQ_CONTRACTS || '3'),
  pollIntervalMs: Number(process.env.TOPSTEPX_NQ_POLL_MS || '60000'),
  initialBackfillBars: Number(process.env.TOPSTEPX_NQ_BACKFILL || '250'), // Fetch extra for indicator warm-up
  dailyLossLimit: Number(process.env.TOPSTEPX_NQ_DAILY_LOSS_LIMIT || '2000'), // $2000 daily loss limit
};

const STOP_MONITOR_DELAY_MS = Number(process.env.TOPSTEPX_NQ_STOP_MONITOR_MS || '1500');
const DASHBOARD_PORT = Number(process.env.TOPSTEPX_NQ_DASHBOARD_PORT || '3333');
const TOPSTEPX_LIVE_ACCOUNT_ID = process.env.TOPSTEPX_ACCOUNT_ID || process.env.TOPSTEPX_NQ_LIVE_ACCOUNT_ID;
const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market';
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user';

const TRADE_LOG_FILE = process.env.TOPSTEPX_NQ_TRADE_LOG || './logs/topstepx-nq-winner-enhanced.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);
const STATE_FILE = './logs/.nq-state.json';

const CT_OFFSET_MINUTES = 6 * 60;
const CUT_OFF_MINUTES = (15 * 60) + 10;  // 3:10 PM CT
const REOPEN_MINUTES = 17 * 60;  // 5:00 PM CT (maintenance window ends)
const WEEKEND_REOPEN_MINUTES = 19 * 60;  // 7:00 PM CT Sunday (no maintenance window)

// Express and Socket.IO setup
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

let pendingSetup: PendingSetup | null = null;
let position: ActivePosition | null = null;
let closes: number[] = [];
let bars: TopstepXFuturesBar[] = [];
let chartHistory: ChartData[] = [];
let multiplier = 20;
let realizedPnL = 0;
let tradingEnabled = false;  // Start with trading disabled - must be enabled via dashboard
let lastProcessedBarTime = '';
let shuttingDown = false;
let tickSize = 0.25;
let commissionPerSide = 1.40;
let accountId = 0;
let contractId: string | null = null;
let resolvedContractId: string | null = null;
let topstepRest: ReturnType<typeof createProjectXRest> | null = null;
let orderManager: TopstepOrderManager | null = null;
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let marketReconnectTimer: NodeJS.Timeout | null = null;
let userReconnectTimer: NodeJS.Timeout | null = null;
let isReconnectingMarket = false;
let isReconnectingUser = false;
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
        stopOrderId: position.stopOrderId,
        targetOrderId: position.targetOrderId,
        entryOrderId: position.entryOrderId,
        stopLoss: position.stopLoss,
        target: position.target,
        entryTime: position.entryTime,
        tradeId: position.tradeId,
      } : null,
      timestamp: nowIso()
    };
    writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  } catch (err: any) {
    log(`[STATE] Failed to save state: ${err.message}`);
  }
}

function loadState(): { tradingEnabled: boolean; position: any } {
  try {
    if (!existsSync(STATE_FILE)) {
      log('[STATE] No previous state found - starting with trading DISABLED');
      return { tradingEnabled: false, position: null };
    }
    const data = readFileSync(STATE_FILE, 'utf-8');
    const state = JSON.parse(data);
    const wasEnabled = state.tradingEnabled || false;
    const savedPosition = state.position || null;

    if (wasEnabled) {
      log(`[STATE] Restored previous state: Trading was ENABLED - auto-resuming`);
    } else {
      log(`[STATE] Restored previous state: Trading was DISABLED`);
    }

    if (savedPosition) {
      log(`[STATE] Found saved position: ${savedPosition.side.toUpperCase()} ${savedPosition.totalQty} @ ${savedPosition.entryPrice}`);
    }

    return { tradingEnabled: wasEnabled, position: savedPosition };
  } catch (err: any) {
    log(`[STATE] Failed to load state: ${err.message} - defaulting to DISABLED`);
    return { tradingEnabled: false, position: null };
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

    // Find position for our contract
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

    // If we have saved position data and it matches, restore full position with order IDs
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
        target: savedPosition.target,
        totalQty: savedPosition.totalQty,
        entryRSI: 50, // Unknown
        entryOrderId: savedPosition.entryOrderId,
        stopOrderId: savedPosition.stopOrderId,
        targetOrderId: savedPosition.targetOrderId,
        stopFilled: false,
        targetFilled: false,
        stopLimitPending: savedPosition.stopOrderId ? true : false,
        monitoringStop: false,
      };
      log(`   Stop Order ID: ${position.stopOrderId} | Target Order ID: ${position.targetOrderId}`);
      log(`   Resume monitoring position with full order management`);
      broadcastDashboardUpdate();
      return;
    }

    // No saved position or mismatch - flatten for safety
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

  // Determine log type based on message content
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

  // Broadcast log to dashboard
  io.emit('log', { timestamp: nowIso(), message, type });
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

    // Calculate realized PnL from exit trades
    realizedPnL = accountTrades
      .filter(trade => trade.type === 'exit')
      .reduce((sum, trade) => sum + (trade.pnl || 0), 0);

    // Get the highest trade ID to continue sequence
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

    // Emit trade event to dashboard
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

function broadcastDashboardUpdate() {
  const currentPnL = position ? calculatePnL(position.entryPrice, lastQuotePrice, position.side, position.totalQty) : 0;

  io.emit('status', {
    tradingEnabled,
    position: position ? {
      ...position,
      unrealizedPnL: currentPnL,
    } : null,
    pendingSetup: pendingSetup?.side,
    realizedPnL,
    accountStatus,
    lastQuote: lastQuotePrice,
    currentBar,
    timestamp: nowIso(),
  });

  // Update position unrealized PnL
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

  // Check daily loss limit
  if (accountStatus.dailyPnL <= -CONFIG.dailyLossLimit) {
    log(`[SAFETY] Daily loss limit reached (${formatCurrency(accountStatus.dailyPnL)}). Skipping entry.`);
    return;
  }

  const tradeId = nextTradeId();
  const entrySide: OrderSide = side === 'long' ? 'Buy' : 'Sell';
  const exitSide: OrderSide = side === 'long' ? 'Sell' : 'Buy';

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

  // CRITICAL: Log order attempt to append-only file
  const criticalLogMsg = `[CRITICAL ORDER ATTEMPT] ${nowIso()} | ${side.toUpperCase()} MARKET | Qty: ${CONFIG.numberOfContracts} | Stop: ${stopPrice.toFixed(2)} | Target: ${targetPrice.toFixed(2)} | Account: ${accountId}`;
  log(criticalLogMsg);
  try {
    require('fs').appendFileSync('logs/critical-orders.log', criticalLogMsg + '\n');
  } catch (e) {
    console.error('[CRITICAL LOG FAILED]', e);
  }

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
    const errorMsg = `[CRITICAL ORDER FAILED] ${nowIso()} | ${side.toUpperCase()} FAILED | Error: ${err.message} | Account: ${accountId}`;
    log(`[ERROR] Failed to place bracket order: ${err.message}`);
    try {
      require('fs').appendFileSync('logs/critical-orders.log', errorMsg + '\n');
    } catch (e) {
      console.error('[CRITICAL LOG FAILED]', e);
    }
    return;
  }

  // Validate order IDs were returned
  if (!bracketResult || !bracketResult.entryOrderId || !bracketResult.stopOrderId || !bracketResult.targetOrderId) {
    const errorMsg = `[CRITICAL ORDER INVALID] ${nowIso()} | ${side.toUpperCase()} | Invalid order IDs returned! Entry: ${bracketResult?.entryOrderId}, Stop: ${bracketResult?.stopOrderId}, Target: ${bracketResult?.targetOrderId}`;
    log(errorMsg);
    try {
      require('fs').appendFileSync('logs/critical-orders.log', errorMsg + '\n');
    } catch (e) {
      console.error('[CRITICAL LOG FAILED]', e);
    }
    return;
  }

  // Log successful order placement
  const successMsg = `[CRITICAL ORDER SUCCESS] ${nowIso()} | ${side.toUpperCase()} | Entry ID: ${bracketResult.entryOrderId} | Stop ID: ${bracketResult.stopOrderId} | Target ID: ${bracketResult.targetOrderId}`;
  log(successMsg);
  try {
    require('fs').appendFileSync('logs/critical-orders.log', successMsg + '\n');
  } catch (e) {
    console.error('[CRITICAL LOG FAILED]', e);
  }

  const estimatedEntryPrice = price;

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
    stopLimitPending: !bracketResult.stopFilled,
    monitoringStop: false,
  };

  log(
    `ENTERED ${side.toUpperCase()} MARKET @ ~${estimatedEntryPrice.toFixed(2)} ` +
    `(RSI ${rsi.toFixed(1)}, Entry: ${bracketResult.entryOrderId}, Stop: ${bracketResult.stopOrderId}, Target: ${bracketResult.targetOrderId})`
  );

  logTradeEvent({
    type: 'entry',
    accountId,
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

  // Add entry marker to chart
  if (chartHistory.length > 0) {
    chartHistory[chartHistory.length - 1].entry = true;
  }

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
  reason: 'stop' | 'target' | 'end_of_session' | 'manual' | 'daily_loss_limit',
  isMarketStop: boolean = false,
) {
  if (!position) {
    return;
  }

  // Use actual commission data if available from TopstepX, otherwise use estimated
  const actualEntryFee = position.entryCommission ?? (commissionPerSide * position.totalQty);
  const actualExitFee = position.exitCommission ?? (commissionPerSide * position.totalQty);
  const fees = actualEntryFee + actualExitFee;
  const isActualFees = position.entryCommission !== undefined && position.exitCommission !== undefined;

  const pnl = calculatePnL(position.entryPrice, price, position.side, position.totalQty);
  realizedPnL += pnl;

  const durationSeconds = (new Date(timestamp).getTime() - new Date(position.entryTime).getTime()) / 1000;

  log(
    `EXITED ${position.side.toUpperCase()} @ ${price.toFixed(2)} (${reason}${isMarketStop ? ' - MARKET' : ''}) ` +
    `| PnL: ${formatCurrency(pnl)} | Fees: ${formatCurrency(fees)}${isActualFees ? ' (actual)' : ' (est.)'} | Duration: ${durationSeconds.toFixed(0)}s | Cumulative: ${formatCurrency(realizedPnL)}`
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
    qty: position.totalQty,
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

  // Add exit marker to chart
  if (chartHistory.length > 0) {
    chartHistory[chartHistory.length - 1].exit = true;
  }

  position = null;
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
  lastMarketDataTime = new Date();  // Track when we last received data
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

  // Emit live bar update (in-progress bar without indicators)
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

  marketHub.onclose(async (error) => {
    log(`❌ TopstepX market hub connection CLOSED: ${error?.message || 'Unknown reason'}`);
    log('🔄 Will automatically reconnect every 5 seconds until connection is restored...');

    // Start aggressive reconnection attempts
    if (!isReconnectingMarket) {
      isReconnectingMarket = true;
      attemptMarketReconnect();
    }
  });

  async function attemptMarketReconnect() {
    if (!contractId) {
      log('[RECONNECT] Cannot reconnect market hub: contractId is null');
      isReconnectingMarket = false;
      return;
    }

    try {
      log('[RECONNECT] Attempting to reconnect market hub...');

      // Clear existing connection
      if (marketHub) {
        try {
          await marketHub.stop();
        } catch (e) {
          // Ignore stop errors
        }
      }

      // Recreate connection
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

      marketHub.onclose(async (error) => {
        log(`❌ TopstepX market hub connection CLOSED: ${error?.message || 'Unknown reason'}`);
        log('🔄 Will automatically reconnect every 5 seconds until connection is restored...');

        if (!isReconnectingMarket) {
          isReconnectingMarket = true;
          attemptMarketReconnect();
        }
      });

      await marketHub.start();
      log('✅ TopstepX market hub RECONNECTED successfully!');
      subscribeMarket();

      // Success - stop reconnection attempts
      if (marketReconnectTimer) {
        clearTimeout(marketReconnectTimer);
        marketReconnectTimer = null;
      }
      isReconnectingMarket = false;

    } catch (error: any) {
      log(`[RECONNECT] Market hub reconnection failed: ${error?.message || 'Unknown error'}`);

      // Schedule next attempt in 5 seconds
      marketReconnectTimer = setTimeout(() => {
        attemptMarketReconnect();
      }, 5000);
    }
  }

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

    // Log full event to see available fields (including potential fee data)
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
          handlePositionExit(price, nowIso(), 'stop', false);
        } else if (orderId === position.targetOrderId) {
          position.targetFilled = true;
          if (commission !== null) {
            position.exitCommission = commission;
          }
          handlePositionExit(price, nowIso(), 'target', false);
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

      // Check if at risk
      if (accountStatus.dailyPnL <= -CONFIG.dailyLossLimit * 0.8) {
        accountStatus.isAtRisk = true;
        log(`[WARNING] Approaching daily loss limit: ${formatCurrency(accountStatus.dailyPnL)}`);
      }

      // Force shutdown if limit hit
      if (accountStatus.dailyPnL <= -CONFIG.dailyLossLimit) {
        log(`[SAFETY] Daily loss limit exceeded: ${formatCurrency(accountStatus.dailyPnL)}`);
        if (position && orderManager) {
          const exitSide: OrderSide = position.side === 'long' ? 'Sell' : 'Buy';
          orderManager.placeMarketIOC(exitSide, position.totalQty).catch(err =>
            log(`[ERROR] Failed to flatten position: ${err.message}`)
          );
        }
        handlePositionExit(lastQuotePrice, nowIso(), 'daily_loss_limit');
        shutdown('daily_loss_limit');
      }

      broadcastDashboardUpdate();
    }
  });

  userHub.on('GatewayUserPosition', (_cid: string, data: any) => {
    if (data) {
      log(`[position] ${JSON.stringify(data)}`);

      // Reconcile position
      const brokerQty = Math.abs(data.quantity || 0);
      const brokerSide = data.quantity > 0 ? 'long' : 'short';

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

  userHub.onclose(async (error) => {
    log(`❌ TopstepX user hub connection CLOSED: ${error?.message || 'Unknown reason'}`);
    log('🔄 Will automatically reconnect every 5 seconds until connection is restored...');

    // Start aggressive reconnection attempts
    if (!isReconnectingUser) {
      isReconnectingUser = true;
      attemptUserReconnect();
    }
  });

  async function attemptUserReconnect() {
    if (!accountId) {
      log('[RECONNECT] Cannot reconnect user hub: accountId is null');
      isReconnectingUser = false;
      return;
    }

    try {
      log('[RECONNECT] Attempting to reconnect user hub...');

      // Clear existing connection
      if (userHub) {
        try {
          await userHub.stop();
        } catch (e) {
          // Ignore stop errors
        }
      }

      // Recreate connection (duplicate all event handlers from startUserStream)
      await startUserStream(accountId);

      log('✅ TopstepX user hub RECONNECTED successfully!');

      // Success - stop reconnection attempts
      if (userReconnectTimer) {
        clearTimeout(userReconnectTimer);
        userReconnectTimer = null;
      }
      isReconnectingUser = false;

    } catch (error: any) {
      log(`[RECONNECT] User hub reconnection failed: ${error?.message || 'Unknown error'}`);

      // Schedule next attempt in 5 seconds
      userReconnectTimer = setTimeout(() => {
        attemptUserReconnect();
      }, 5000);
    }
  }

  await userHub.start();
  log('✅ TopstepX user hub connected');
  subscribeUser();
}

async function processBar(bar: TopstepXFuturesBar) {
  if (bar.timestamp === lastProcessedBarTime) {
    return;
  }
  lastProcessedBarTime = bar.timestamp;

  closes.push(bar.close);
  bars.push(bar);

  const maxHistory = Math.max(CONFIG.bbPeriod + 100, 200);
  if (closes.length > maxHistory) {
    closes = closes.slice(-maxHistory);
  }
  if (bars.length > maxHistory) {
    bars = bars.slice(-maxHistory);
  }

  // Calculate indicators
  const bb = calculateBollingerBands(closes, CONFIG.bbPeriod, CONFIG.bbStdDev);
  const rsiValues = closes.length >= CONFIG.rsiPeriod ?
    RSI.calculate({ values: closes, period: CONFIG.rsiPeriod }) : [];
  const currentRSI = rsiValues.length > 0 ? rsiValues[rsiValues.length - 1] : undefined;

  const ttmBars = bars.slice(Math.max(0, bars.length - 21));
  const ttmSqueeze = ttmBars.length >= 21 ?
    calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 }) : null;

  // Log TTM Squeeze state when waiting for trigger (only on new bars)
  if (ttmSqueeze && pendingSetup && lastProcessedBarTime !== bar.timestamp) {
    const currentSqueezeState = ttmSqueeze.squeezeOn ? 'ON' : 'OFF';
    log(
      `[TTM] Squeeze ${currentSqueezeState} | BB: ${ttmSqueeze.bbUpper.toFixed(2)}-${ttmSqueeze.bbLower.toFixed(2)} | ` +
      `KC: ${ttmSqueeze.kcUpper.toFixed(2)}-${ttmSqueeze.kcLower.toFixed(2)} | Waiting for ${pendingSetup.side.toUpperCase()} trigger`
    );
  }

  // Create chart data point
  const chartPoint: ChartData = {
    timestamp: bar.timestamp,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    close: bar.close,
    bbUpper: bb?.upper,
    bbBasis: bb?.middle,  // Dashboard expects bbBasis not bbMiddle
    bbMiddle: bb?.middle,  // Keep for compatibility
    bbLower: bb?.lower,
    rsi: currentRSI,
    ttmMomentum: ttmSqueeze?.momentum,  // Flatten for dashboard
    squeeze: ttmSqueeze ? {
      momentum: ttmSqueeze.momentum,
      squeezeFiring: ttmSqueeze.squeezeOn
    } : undefined,
    signal: null,
  };

  // Detect signals
  if (bb && currentRSI !== undefined && !position) {
    const price = bar.close;
    const longSignal = price <= bb.lower && currentRSI < CONFIG.rsiOversold;
    const shortSignal = price >= bb.upper && currentRSI > CONFIG.rsiOverbought;

    if (longSignal) chartPoint.signal = 'long';
    if (shortSignal) chartPoint.signal = 'short';
  }

  chartHistory.push(chartPoint);

  // Keep chart history limited
  if (chartHistory.length > 500) {
    chartHistory = chartHistory.slice(-500);
  }

  // Broadcast bar update to dashboard
  io.emit('bar', chartPoint);

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

  if (position || closes.length < CONFIG.bbPeriod) {
    return;
  }

  if (!bb || currentRSI === undefined || !ttmSqueeze) {
    return;
  }

  // Skip signal detection if trading is disabled
  if (!tradingEnabled) {
    return;
  }

  // Log individual condition checks (every bar when conditions are met)
  logIndividualConditions(CONFIG.symbol, bar.close, currentRSI, bb, ttmSqueeze);

  // Entry logic - two-stage system
  const price = bar.close;
  const longSetupDetected = price <= bb.lower && currentRSI < CONFIG.rsiOversold;
  const shortSetupDetected = price >= bb.upper && currentRSI > CONFIG.rsiOverbought;

  if (!pendingSetup && longSetupDetected) {
    pendingSetup = {
      side: 'long',
      setupTime: bar.timestamp,
      setupPrice: bar.close,
      rsi: currentRSI,
      bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
    };
    log(
      `LONG setup detected @ ${bar.close.toFixed(2)} (RSI ${currentRSI.toFixed(1)}) | ` +
      `TTM Squeeze currently ${ttmSqueeze.squeezeOn ? 'ON' : 'OFF'} - awaiting trigger`
    );

    // Log setup progression - Step 1
    logSetupProgress(
      CONFIG.symbol,
      'LONG',
      'STEP1_RSI_BB',
      `Price ${bar.close.toFixed(2)} <= BB Lower ${bb.lower.toFixed(2)} | RSI ${currentRSI.toFixed(2)} < ${CONFIG.rsiOversold}`
    );

    // Log waiting for TTM - Step 2
    logSetupProgress(
      CONFIG.symbol,
      'LONG',
      'STEP2_WAITING_TTM',
      `TTM Squeeze currently ${ttmSqueeze.squeezeOn ? 'FIRING (ready for entry!)' : 'OFF (waiting...)'}`
    );

    broadcastDashboardUpdate();
  } else if (!pendingSetup && shortSetupDetected) {
    pendingSetup = {
      side: 'short',
      setupTime: bar.timestamp,
      setupPrice: bar.close,
      rsi: currentRSI,
      bb: { upper: bb.upper, middle: bb.middle, lower: bb.lower },
    };
    log(
      `SHORT setup detected @ ${bar.close.toFixed(2)} (RSI ${currentRSI.toFixed(1)}) | ` +
      `TTM Squeeze currently ${ttmSqueeze.squeezeOn ? 'ON' : 'OFF'} - awaiting trigger`
    );

    // Log setup progression - Step 1
    logSetupProgress(
      CONFIG.symbol,
      'SHORT',
      'STEP1_RSI_BB',
      `Price ${bar.close.toFixed(2)} >= BB Upper ${bb.upper.toFixed(2)} | RSI ${currentRSI.toFixed(2)} > ${CONFIG.rsiOverbought}`
    );

    // Log waiting for TTM - Step 2
    logSetupProgress(
      CONFIG.symbol,
      'SHORT',
      'STEP2_WAITING_TTM',
      `TTM Squeeze currently ${ttmSqueeze.squeezeOn ? 'FIRING (ready for entry!)' : 'OFF (waiting...)'}`
    );

    broadcastDashboardUpdate();
  }

  // If we have a pending setup, log TTM status updates
  if (pendingSetup && !ttmSqueeze.squeezeOn) {
    // Update waiting status periodically
    const setupAge = Date.now() - new Date(pendingSetup.setupTime).getTime();
    const setupAgeMinutes = Math.floor(setupAge / 60000);
    if (setupAgeMinutes > 0 && setupAge % 60000 < 60000) { // Log once per minute
      logSetupProgress(
        CONFIG.symbol,
        pendingSetup.side.toUpperCase() as 'LONG' | 'SHORT',
        'STEP2_WAITING_TTM',
        `Still waiting... Setup age: ${setupAgeMinutes}m | TTM Squeeze: OFF`
      );
    }
  }

  if (pendingSetup && ttmSqueeze.squeezeOn) {
    const setup = { ...pendingSetup };
    log(
      `TTM Squeeze trigger fired - entering ${setup.side.toUpperCase()} @ ${bar.close.toFixed(2)} ` +
      `(setup was @ ${setup.setupPrice.toFixed(2)})`
    );

    // Log TTM fired - Step 3
    logSetupProgress(
      CONFIG.symbol,
      setup.side.toUpperCase() as 'LONG' | 'SHORT',
      'STEP3_TTM_FIRED',
      `TTM Squeeze FIRING! Momentum: ${ttmSqueeze.momentum.toFixed(2)} | Entry price: ${bar.close.toFixed(2)}`
    );

    await enterPosition(setup.side, bar.close, bar.timestamp, setup.rsi, setup.bb);

    // Log final entry - Step 4
    logSetupProgress(
      CONFIG.symbol,
      setup.side.toUpperCase() as 'LONG' | 'SHORT',
      'ENTRY',
      `Position entered @ ${bar.close.toFixed(2)} | Setup was @ ${setup.setupPrice.toFixed(2)}`
    );

    pendingSetup = null;
  }
}

async function bootstrapHistoricalData(contractId: string) {
  log(`Fetching initial ${CONFIG.initialBackfillBars} 1-minute bars for bootstrap...`);

  const initialBars = (await fetchTopstepXFuturesBars({
    contractId,
    startTime: new Date(Date.now() - CONFIG.initialBackfillBars * 60 * 1000).toISOString(),
    endTime: new Date().toISOString(),
    unit: 2, // Minutes
    unitNumber: 1,
    limit: CONFIG.initialBackfillBars,
  })).reverse();

  log(`Loaded ${initialBars.length} initial bars`);

  // Process each bar properly to calculate all indicators
  for (const bar of initialBars) {
    closes.push(bar.close);
    bars.push(bar);

    // Calculate indicators for historical bars
    const bb = calculateBollingerBands(closes, CONFIG.bbPeriod, CONFIG.bbStdDev);
    const rsiValues = closes.length >= CONFIG.rsiPeriod ?
      RSI.calculate({ values: closes, period: CONFIG.rsiPeriod }) : [];
    const currentRSI = rsiValues.length > 0 ? rsiValues[rsiValues.length - 1] : undefined;

    const ttmBars = bars.slice(Math.max(0, bars.length - 21));
    const ttmSqueeze = ttmBars.length >= 21 ?
      calculateTtmSqueeze(ttmBars, { lookback: 20, bbStdDev: 2, atrMultiplier: 1.5 }) : null;

    // Only add to chart history once we have valid indicators
    // (need at least 24 bars for RSI and 20 for BB)
    if (closes.length >= CONFIG.rsiPeriod && bb && currentRSI !== undefined) {
      chartHistory.push({
        timestamp: bar.timestamp,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        bbUpper: bb?.upper,
        bbBasis: bb?.middle,  // Dashboard expects bbBasis not bbMiddle
        bbLower: bb?.lower,
        rsi: currentRSI,
        ttmMomentum: ttmSqueeze?.momentum,  // Flatten for dashboard
        squeeze: ttmSqueeze ? {
          momentum: ttmSqueeze.momentum,
          squeezeFiring: ttmSqueeze.squeezeOn
        } : undefined,
        signal: null,
      });
    }
  }

  // Mark the last processed bar time
  if (initialBars.length > 0) {
    lastProcessedBarTime = initialBars[initialBars.length - 1].timestamp;
  }

  log(`Bootstrap complete. Indicators calculated for ${chartHistory.length} bars`);

  // Send chart data to any connected dashboard clients after bootstrap
  const completeData = getCompleteChartData();
  io.emit('chartHistory', completeData);
}

// Helper function to filter out bars with incomplete indicator data
function getCompleteChartData(): ChartData[] {
  return chartHistory.filter(bar =>
    typeof bar.bbUpper === 'number' && Number.isFinite(bar.bbUpper) &&
    typeof bar.bbLower === 'number' && Number.isFinite(bar.bbLower) &&
    typeof bar.bbBasis === 'number' && Number.isFinite(bar.bbBasis) &&
    typeof bar.rsi === 'number' && Number.isFinite(bar.rsi) &&
    typeof bar.ttmMomentum === 'number' && Number.isFinite(bar.ttmMomentum)
  );
}

// Express routes
app.use(cors()); // Enable CORS for multi-symbol dashboard
app.use(express.static('public'));
app.use(express.json());

// Serve dashboard at root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'nq-winner-dashboard.html'));
});

app.get('/api/status', (req, res) => {
  res.json({
    strategy: 'NQ_WINNER_ENHANCED',
    symbol: CONFIG.symbol,
    status: position ? 'IN_POSITION' : pendingSetup ? 'SETUP_PENDING' : 'SCANNING',
    tradingEnabled,
    position,
    pendingSetup,
    accountStatus,
    performance: {
      realizedPnL,
      tradesCount: tradeSequence,
      lastQuote: lastQuotePrice,
    },
    config: {
      bbPeriod: CONFIG.bbPeriod,
      bbStdDev: CONFIG.bbStdDev,
      rsiPeriod: CONFIG.rsiPeriod,
      rsiOversold: CONFIG.rsiOversold,
      rsiOverbought: CONFIG.rsiOverbought,
      stopLoss: CONFIG.stopLossPercent * 100,
      takeProfit: CONFIG.takeProfitPercent * 100,
      contracts: CONFIG.numberOfContracts,
    },
    timestamp: nowIso(),
  });
});

app.get('/api/chart', (req, res) => {
  const completeData = getCompleteChartData();
  const squeezeCount = completeData.filter(bar => bar.squeeze).length;
  log(`[CHART API] Sending ${completeData.length} bars with complete indicators (${chartHistory.length} total bars, ${chartHistory.length - completeData.length} filtered out)`);
  res.json(completeData);
});

app.get('/api/trades', (req, res) => {
  // Read trades from log file and filter by current account
  try {
    const logContent = require('fs').readFileSync(TRADE_LOG_FILE, 'utf-8');
    const lines = logContent.trim().split('\n');
    const allTrades = lines
      .filter(line => line.trim())
      .map(line => JSON.parse(line));

    // Filter trades for current account and return last 100
    const accountTrades = allTrades
      .filter(trade => trade.accountId === accountId)
      .slice(-100);

    res.json(accountTrades);
  } catch (err) {
    res.json([]);
  }
});

// Account endpoint
app.get('/api/accounts', async (req, res) => {
  try {
    // Fetch all TopstepX accounts
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
    // Fallback to current account only
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

// Account selection endpoint
app.post('/api/account/:id', async (req, res) => {
  const selectedAccountId = parseInt(req.params.id);

  if (!selectedAccountId) {
    return res.status(400).json({ success: false, message: 'Invalid account ID' });
  }

  if (selectedAccountId === accountId) {
    return res.json({ success: true, message: `Already using account ${selectedAccountId}` });
  }

  // Prevent account switching while trading is active
  if (tradingEnabled) {
    return res.status(400).json({
      success: false,
      message: 'Cannot switch accounts while trading is active. Please stop trading first.'
    });
  }

  // Prevent account switching if there's an open position
  if (position) {
    return res.status(400).json({
      success: false,
      message: 'Cannot switch accounts with an open position. Please close position first.'
    });
  }

  try {
    // Verify the account exists
    const accounts = await fetchTopstepXAccounts(true);
    const targetAccount = accounts.find(a => a.id === selectedAccountId);

    if (!targetAccount) {
      return res.status(404).json({ success: false, message: `Account ${selectedAccountId} not found` });
    }

    log(`[ACCOUNT] Switching account: ${accountId} -> ${selectedAccountId}`);

    // Verify required components are initialized
    if (!topstepRest || !contractId) {
      throw new Error('System not fully initialized. Please wait for startup to complete.');
    }

    // Update the account ID
    accountId = selectedAccountId;

    // Recreate the order manager with new account ID
    orderManager = new TopstepOrderManager(topstepRest, accountId, contractId, tickSize);
    log(`[ACCOUNT] Order manager recreated for account ${accountId}`);

    // Fetch updated account status
    try {
      await updateAccountStatus();
      log(`[ACCOUNT] Account status updated: Balance ${accountStatus.balance}, Daily P&L ${accountStatus.dailyPnL}`);
    } catch (error) {
      log(`[ACCOUNT] Warning: Could not fetch account status: ${error}`);
    }

    // Recalculate performance stats for this account
    recalculateAccountStats(accountId);

    // Broadcast update to all connected dashboards
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

// Control endpoints - using correct paths
app.post('/api/trading/start', (req, res) => {
  tradingEnabled = true;
  saveState();
  log(`[CONTROL] Trading STARTED via dashboard | Account: ${accountId}`);
  broadcastDashboardUpdate();
  res.json({ success: true, message: 'Trading started' });
});

app.post('/api/trading/stop', (req, res) => {
  tradingEnabled = false;
  pendingSetup = null; // Clear any pending setups
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
      await orderManager.placeMarketIOC(exitSide, position.totalQty);
      await handlePositionExit(lastQuotePrice || position.entryPrice, nowIso(), 'manual');
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

  // Send initial data
  socket.emit('config', {
    symbol: CONFIG.symbol,
    bbPeriod: CONFIG.bbPeriod,
    bbStdDev: CONFIG.bbStdDev,
    rsiPeriod: CONFIG.rsiPeriod,
    rsiOversold: CONFIG.rsiOversold,
    rsiOverbought: CONFIG.rsiOverbought,
  });

  // Only send chart history if we have data (after bootstrap)
  const completeData = getCompleteChartData();
  if (completeData.length > 0) {
    log(`[SOCKET] Sending ${completeData.length} bars to dashboard client ${socket.id} (${chartHistory.length} total, ${chartHistory.length - completeData.length} filtered)`);
    socket.emit('chartHistory', completeData);
  } else {
    log(`[SOCKET] No chart history available yet for client ${socket.id}`);
  }
  broadcastDashboardUpdate();

  // Handle chart history request
  socket.on('chartHistory', () => {
    const data = getCompleteChartData();
    socket.emit('chartHistory', data);
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
  console.log('TOPSTEPX LIVE NQ WINNER STRATEGY - ENHANCED VERSION');
  console.log('='.repeat(80));
  console.log(`Symbol: ${CONFIG.symbol}`);
  console.log(`BB Period: ${CONFIG.bbPeriod} bars | Std Dev: ${CONFIG.bbStdDev}`);
  console.log(`RSI Period: ${CONFIG.rsiPeriod} | Oversold: ${CONFIG.rsiOversold} | Overbought: ${CONFIG.rsiOverbought}`);
  console.log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Take Profit: ${(CONFIG.takeProfitPercent * 100).toFixed(3)}%`);
  console.log(`Contracts: ${CONFIG.numberOfContracts}`);
  console.log(`Daily Loss Limit: $${CONFIG.dailyLossLimit}`);
  console.log(`Dashboard: http://localhost:${DASHBOARD_PORT}`);
  console.log('='.repeat(80));

  log('Main function started.');

  // Load previous trading state
  const savedState = loadState();
  tradingEnabled = savedState.tradingEnabled;
  const savedPosition = savedState.position;

  // Start dashboard server
  server.listen(DASHBOARD_PORT, () => {
    log(`Dashboard server running on http://localhost:${DASHBOARD_PORT}`);
    if (!tradingEnabled) {
      log(`⚠️ Trading is DISABLED. Use dashboard to start trading.`);
    }
  });

  // Resolve contract metadata
  log('Resolving contract metadata...');
  const lookupKey = CONFIG.contractId || CONFIG.symbol;
  const metadata = await fetchTopstepXFuturesMetadata(lookupKey);

  if (!metadata) {
    throw new Error(`Unable to resolve metadata for ${lookupKey}`);
  }

  contractId = metadata.id;
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
  accountId = await resolveAccountId();
  log('Creating ProjectX REST client...');
  topstepRest = createProjectXRest();
  orderManager = new TopstepOrderManager(topstepRest, accountId, contractId, tickSize);
  log(`Using TopstepX account ${accountId}`);

  // Bootstrap historical data
  await bootstrapHistoricalData(contractId);

  // Update account status initially
  await updateAccountStatus();
  log(`Account balance: $${accountStatus.balance.toFixed(2)}`);

  // Reconcile any existing positions
  await reconcilePosition(savedPosition);

  // Update account status and heartbeat every 30 seconds
  setInterval(async () => {
    await updateAccountStatus();
    broadcastDashboardUpdate();

    // Heartbeat log - check if we're actually receiving market data
    const now = new Date();
    const dataStale = lastMarketDataTime && (now.getTime() - lastMarketDataTime.getTime()) > 120000; // 2 minutes
    const noDataYet = !lastMarketDataTime;

    let statusText: string;
    if (noDataYet || dataStale) {
      statusText = '⚠️ NO MARKET DATA - STRATEGY CANNOT RUN';
    } else if (tradingEnabled) {
      statusText = '✅ RUNNING';
    } else {
      statusText = '⏸ PAUSED';
    }

    const posText = position ? `| Position: ${position.side.toUpperCase()} ${position.totalQty}` : '| No position';
    log(`🚀 Strategy ${statusText} | Symbol: ${CONFIG.symbol} | Account: ${accountId} ${posText}`);
  }, 30000);

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

  log('='.repeat(80));
  log('🚀 STRATEGY FULLY INITIALIZED AND RUNNING');
  log(`   Symbol: ${CONFIG.symbol} | Account: ${accountId}`);
  log(`   Trading: ${tradingEnabled ? 'ENABLED ✅' : 'DISABLED ⏸ (use dashboard to start)'}`);
  log(`   Dashboard: http://localhost:${DASHBOARD_PORT}`);
  if (position) {
    log(`   Active Position: ${position.side.toUpperCase()} ${position.totalQty} @ ${position.entryPrice.toFixed(2)}`);
  }
  log('='.repeat(80));
  await new Promise(() => {});
}

main().catch(err => {
  console.error('TopstepX NQ winner enhanced strategy failed:', err);
  process.exit(1);
});