#!/usr/bin/env tsx
/**
 * TopstepX Live Multi-Symbol Strategy
 *
 * Trades MNQ, MES, MGC, M6E simultaneously on a single account
 * - Independent indicators and signals per symbol
 * - Shared account balance and daily P&L
 * - Unified dashboard with 4 charts
 * - Per-symbol enable/disable controls
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
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'fs';
import * as path from 'path';
import { inferFuturesCommissionPerSide } from './lib/futuresFees';
import { createProjectXRest } from './projectx-rest';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import express from 'express';
import { Server } from 'socket.io';
import http from 'http';
import { TopstepOrderManager } from './lib/topstepOrderManager';

// Symbol configuration
const SYMBOLS = {
  MNQ: 'MNQZ5',  // Micro E-mini Nasdaq-100
  MES: 'MESZ5',  // Micro E-mini S&P 500
  MGC: 'MGCZ5',  // Micro Gold
  M6E: 'M6EZ5',  // Micro Euro FX
} as const;

type SymbolKey = keyof typeof SYMBOLS;

// Interfaces
interface ChartData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  bbUpper?: number;
  bbMiddle?: number;
  bbBasis?: number;
  bbLower?: number;
  rsi?: number;
  ttmMomentum?: number;
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

// Per-symbol context
interface SymbolContext {
  symbol: string;
  contractId: string | null;
  tickSize: number;
  multiplier: number;
  commissionPerSide: number;

  // Market data
  bars: TopstepXFuturesBar[];
  closes: number[];
  chartHistory: ChartData[];
  currentBar: TopstepXFuturesBar | null;
  barStartTime: Date | null;
  lastProcessedBarTime: string;
  lastQuotePrice: number;

  // Trading state
  enabled: boolean;
  position: ActivePosition | null;
  pendingSetup: PendingSetup | null;
  tradeSequence: number;
  realizedPnL: number;

  // Market hub connection
  marketHub: HubConnection | null;

  // Order manager
  orderManager: TopstepOrderManager | null;
}

// Strategy configuration
interface StrategyConfig {
  bbPeriod: number;
  bbStdDev: number;
  rsiPeriod: number;
  rsiOversold: number;
  rsiOverbought: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  numberOfContracts: number;
  initialBackfillBars: number;
  dailyLossLimit: number;
}

const CONFIG: StrategyConfig = {
  bbPeriod: parseInt(process.env.TOPSTEPX_BB_PERIOD || '20'),
  bbStdDev: parseFloat(process.env.TOPSTEPX_BB_STDDEV || '3'),
  rsiPeriod: parseInt(process.env.TOPSTEPX_RSI_PERIOD || '24'),
  rsiOversold: parseInt(process.env.TOPSTEPX_RSI_OVERSOLD || '30'),
  rsiOverbought: parseInt(process.env.TOPSTEPX_RSI_OVERBOUGHT || '70'),
  stopLossPercent: parseFloat(process.env.TOPSTEPX_STOP_LOSS_PERCENT || '0.0001'),
  takeProfitPercent: parseFloat(process.env.TOPSTEPX_TAKE_PROFIT_PERCENT || '0.0005'),
  numberOfContracts: parseInt(process.env.TOPSTEPX_CONTRACTS || '3'),
  initialBackfillBars: 250,
  dailyLossLimit: parseFloat(process.env.TOPSTEPX_DAILY_LOSS_LIMIT || '2000'),
};

const MARKET_HUB_URL = process.env.TOPSTEPX_MARKET_HUB_URL || 'https://rtc.topstepx.com/hubs/market';
const USER_HUB_URL = process.env.TOPSTEPX_USER_HUB_URL || 'https://rtc.topstepx.com/hubs/user';
const TRADE_LOG_FILE = process.env.TOPSTEPX_MULTI_TRADE_LOG || './logs/topstepx-multi-symbol.jsonl';
const TRADE_LOG_DIR = path.dirname(TRADE_LOG_FILE);
const STOP_MONITOR_DELAY_MS = 3000;

// Global state (shared across symbols)
let accountId = 0;
let topstepRest: ReturnType<typeof createProjectXRest> | null = null;
let userHub: HubConnection | null = null;
let accountStatus: AccountStatus = {
  balance: 0,
  buyingPower: 0,
  dailyPnL: 0,
  openPositions: 0,
  dailyLossLimit: CONFIG.dailyLossLimit,
  isAtRisk: false,
};
let shuttingDown = false;

// Symbol contexts (one per symbol)
const symbolContexts = new Map<SymbolKey, SymbolContext>();

// Express and Socket.IO setup
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*' },
  transports: ['websocket', 'polling'],
});

app.use(express.json());
app.use(express.static('public'));

// Logging
function log(message: string, symbolKey?: SymbolKey) {
  const timestamp = new Date().toISOString();
  const prefix = symbolKey ? `[${symbolKey}]` : '';
  console.log(`[${timestamp}]${prefix} ${message}`);
}

function nowIso(): string {
  return new Date().toISOString();
}

function formatCurrency(amount: number): string {
  return `$${amount.toFixed(2)}`;
}

// Ensure trade log directory exists
function ensureTradeLogDir() {
  if (!existsSync(TRADE_LOG_DIR)) {
    mkdirSync(TRADE_LOG_DIR, { recursive: true });
  }
}

// Recalculate account stats from trade log
function recalculateAccountStats(targetAccountId: number) {
  try {
    if (!existsSync(TRADE_LOG_FILE)) {
      // Reset all symbols
      symbolContexts.forEach(ctx => {
        ctx.realizedPnL = 0;
        ctx.tradeSequence = 0;
      });
      return;
    }

    const logContent = readFileSync(TRADE_LOG_FILE, 'utf-8');
    const lines = logContent.trim().split('\n').filter(line => line.trim());
    const accountTrades = lines
      .map(line => JSON.parse(line))
      .filter(trade => trade.accountId === targetAccountId);

    // Calculate per-symbol stats
    symbolContexts.forEach((ctx, symbolKey) => {
      const symbolTrades = accountTrades.filter(trade => trade.symbol === symbolKey);

      ctx.realizedPnL = symbolTrades
        .filter(trade => trade.type === 'exit')
        .reduce((sum, trade) => sum + (trade.pnl || 0), 0);

      ctx.tradeSequence = symbolTrades
        .filter(trade => trade.type === 'entry')
        .reduce((max, trade) => Math.max(max, trade.tradeId || 0), 0);

      log(`Stats recalculated: ${symbolTrades.length} trades, PnL: ${ctx.realizedPnL.toFixed(2)}, Last trade ID: ${ctx.tradeSequence}`, symbolKey);
    });
  } catch (error) {
    log(`[ACCOUNT] Warning: Could not recalculate stats: ${error}`);
    symbolContexts.forEach(ctx => {
      ctx.realizedPnL = 0;
      ctx.tradeSequence = 0;
    });
  }
}

// Log trade event
function logTradeEvent(event: Record<string, any>) {
  try {
    ensureTradeLogDir();
    const tradeData = { timestamp: nowIso(), ...event };
    appendFileSync(TRADE_LOG_FILE, `${JSON.stringify(tradeData)}\n`);

    // Emit trade event to dashboard
    if (event.type === 'exit') {
      const trade = {
        tradeId: event.tradeId,
        symbol: event.symbol,
        side: event.side,
        entryPrice: event.entryPrice,
        exitPrice: event.exitPrice,
        entryTime: event.entryTime || event.timestamp,
        exitTime: event.timestamp,
        pnl: event.pnl,
        reason: event.reason,
      };
      io.emit('trade', trade);
    }
  } catch (error) {
    log(`[ERROR] Failed to log trade: ${error}`);
  }
}

// Calculate P&L
function calculatePnL(entryPrice: number, exitPrice: number, side: 'long' | 'short', qty: number, multiplier: number): number {
  const direction = side === 'long' ? 1 : -1;
  return (exitPrice - entryPrice) * direction * multiplier * qty;
}

// Round price to tick size
function roundToTick(price: number, tickSize: number): number {
  return Math.round(price / tickSize) * tickSize;
}

// Broadcast dashboard update
function broadcastDashboardUpdate() {
  const positions: any[] = [];
  const charts: Record<string, ChartData[]> = {};
  let totalRealizedPnL = 0;
  let totalTrades = 0;

  symbolContexts.forEach((ctx, symbolKey) => {
    if (ctx.position) {
      const currentPnL = calculatePnL(
        ctx.position.entryPrice,
        ctx.lastQuotePrice,
        ctx.position.side,
        ctx.position.totalQty,
        ctx.multiplier
      );
      positions.push({
        ...ctx.position,
        unrealizedPnL: currentPnL,
      });
    }

    charts[symbolKey] = ctx.chartHistory;
    totalRealizedPnL += ctx.realizedPnL;
    totalTrades += ctx.tradeSequence;
  });

  io.emit('update', {
    symbols: Object.fromEntries(
      Array.from(symbolContexts.entries()).map(([key, ctx]) => [
        key,
        {
          enabled: ctx.enabled,
          position: ctx.position,
          pendingSetup: ctx.pendingSetup,
          realizedPnL: ctx.realizedPnL,
          lastQuote: ctx.lastQuotePrice,
        },
      ])
    ),
    charts,
    accountStatus,
    totalRealizedPnL,
    totalTrades,
    timestamp: nowIso(),
  });
}

log('='.repeat(80));
log('TOPSTEPX LIVE MULTI-SYMBOL STRATEGY');
log('='.repeat(80));
log(`Symbols: ${Object.entries(SYMBOLS).map(([k, v]) => `${k}(${v})`).join(', ')}`);
log(`BB Period: ${CONFIG.bbPeriod} bars | Std Dev: ${CONFIG.bbStdDev}`);
log(`RSI Period: ${CONFIG.rsiPeriod} | Oversold: ${CONFIG.rsiOversold} | Overbought: ${CONFIG.rsiOverbought}`);
log(`Stop Loss: ${(CONFIG.stopLossPercent * 100).toFixed(3)}% | Take Profit: ${(CONFIG.takeProfitPercent * 100).toFixed(3)}%`);
log(`Contracts: ${CONFIG.numberOfContracts} per symbol`);
log(`Daily Loss Limit: $${CONFIG.dailyLossLimit}`);
log(`Dashboard: http://localhost:3333`);
log('='.repeat(80));

// Main function to be continued...
async function main() {
  log('Main function started.');

  // TODO: Continue implementation
  // 1. Authenticate
  // 2. Resolve account
  // 3. Initialize all symbol contexts
  // 4. Start user hub
  // 5. Start market hubs for each symbol
  // 6. Start Express server
}

// Start the application
main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
