/**
 * FABIO INTRADAY POSITION TRADING - MES (Micro E-mini S&P 500)
 *
 * Focus: Larger intraday moves (50-150 points), multiple setups per day
 * Timeframes: 1H (primary), 15min (secondary), 5min/1min (entry optimization)
 * Analysis: Session volume profile + previous day, higher timeframe structure
 * Updates: Every 1-2 minutes (semi-real-time)
 * Risk Management: Medium stops for position trades (30-60 points)
 *
 * Philosophy:
 * - Find 2-5 quality setups per day (not 50 scalps, not 1 swing)
 * - Hold positions hours to 3 days max (ideally same day close)
 * - Higher timeframe bias (1H/4H) but trade intraday moves
 * - Level 2 data is bonus, not critical
 * - Target 50-150 point moves (bigger than scalps, faster than swings)
 */

import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  fetchTopstepXAccounts,
  TopstepXFuturesBar,
  authenticate,
} from './lib/topstepx';
import { createProjectXRest } from './projectx-rest';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import {
  buildFuturesMarketData,
  callDeepSeekReasoner,
  ExecutionManager,
  HigherTimeframeSnapshot,
  SessionVolumeProfileSummary,
} from './lib/fabioOpenAIIntegration';
import { analyzeFuturesMarket } from './lib/openaiTradingAgent';

// Self-learning bypassed for intraday position trading
// This system uses pure AI analysis without historical database learning
import { createExecutionManager } from './lib/executionManager';
import { analyzePositionRisk } from './lib/riskManagementAgent';

// Configuration
const SYMBOL = 'MES';
const CONTRACT_ID = parseInt(process.env.TOPSTEPX_CONTRACT_ID_MES || '0');
const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '0');
const DASHBOARD_PORT = 3350; // Unique port for swing trading dashboard
const CONTRACTS = 1;
const ANALYSIS_INTERVAL_MS = 120_000; // Analyze every 2 minutes (120 seconds) - multiple setups per day
const RISK_MGMT_INTERVAL_MS = 120_000; // Risk management every 2 minutes (intraday position management)

// Intraday Position Trading Parameters
const LOOKBACK_DAYS = 2; // Analyze current + previous day
const MIN_TARGET_POINTS = 50; // Minimum 50 point target for position trades
const MAX_STOP_POINTS = 60; // Maximum 60 point stop for position trades (can be flexible)
const TYPICAL_HOLD_HOURS = 4; // Typical hold time: 2-8 hours (same day close ideal)
const MAX_HOLD_DAYS = 3; // Maximum 3 days before forcing review

// Multi-Position Trading Parameters
const MAX_CONCURRENT_POSITIONS = 5; // Allow up to 5 simultaneous positions
const STRATEGIC_POSITIONING = true; // Allow AI to place stops/targets strategically
const POSITION_CORRELATION_AWARE = true; // AI considers correlation between positions

// Market Profile Types
interface MarketProfileNode {
  price: number;
  timeSpent: number; // Total seconds price traded at this level
  volume: number;
  firstTouch: Date;
  lastTouch: Date;
}

interface MarketProfile {
  poc: number; // Point of Control - price with most time
  vah: number; // Value Area High (top 70% of time)
  val: number; // Value Area Low (bottom 70% of time)
  tpo: MarketProfileNode[]; // Time Price Opportunity - all price levels
  totalTime: number;
  highVolNode: number; // Highest volume node (traditional VP POC)
}

// Volume Profile Types (multi-day)
interface MultiDayVolumeProfile {
  daily: VolumeProfile[];
  composite: VolumeProfile; // Combined profile for entire lookback period
  hvns: number[]; // High Volume Nodes across all days
  lvns: number[]; // Low Volume Nodes across all days
}

interface VolumeProfile {
  poc: number;
  vah: number;
  val: number;
  nodes: { price: number; volume: number }[];
  date: string;
}

// Higher Timeframe Structure (Intraday Focus)
interface IntradayTimeframeAnalysis {
  fourHour: {
    trend: 'bullish' | 'bearish' | 'neutral';
    ema20: number;
    ema50: number;
    structure: string;
    keyLevels: number[];
  };
  oneHour: {
    trend: 'bullish' | 'bearish' | 'neutral';
    recentHigh: number;
    recentLow: number;
    swingPoints: number[];
    ema20: number;
  };
  fifteenMin: {
    trend: 'bullish' | 'bearish' | 'neutral';
    entryZone: { low: number; high: number } | null;
    momentum: 'strong' | 'weak' | 'neutral';
  };
  fiveMin: {
    currentPrice: number;
    entryTrigger: boolean;
    microStructure: string;
  };
}

// Global State
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let orderManager: any = null;
let contractId: string = ''; // Resolved TopstepX contract ID for SYMBOL

// State
let bars: TopstepXFuturesBar[] = []; // 1-minute bars for real-time
let fiveMinBars: TopstepXFuturesBar[] = [];
let fifteenMinBars: TopstepXFuturesBar[] = [];
let oneHourBars: TopstepXFuturesBar[] = [];
let fourHourBars: TopstepXFuturesBar[] = [];
let marketProfile: MarketProfile | null = null;
let sessionVolumeProfile: MultiDayVolumeProfile | null = null; // Current + previous session

// Multi-Position State
let activePositions: Map<string, any> = new Map(); // decisionId -> position
let accountBalance = 100000;
let realizedPnL = 0;
let executionManager: ExecutionManager | null = null;
let lastAnalysisTime = 0;
let lastRiskMgmtTime = 0;
let positionRiskDecisions: Map<string, any> = new Map(); // decisionId -> risk decision

// Dashboard
const app = express();
const server = createServer(app);
const io = new SocketIOServer(server, {
  cors: { origin: '*' },
});

app.use(express.static('public'));

// Logging
function log(message: string, level: 'info' | 'success' | 'warn' | 'error' = 'info') {
  const timestamp = new Date().toISOString();
  const prefix = '[SWING-MES]';
  console.log(`[${timestamp}]${prefix} ${message}`);
}

/**
 * Calculate Market Profile from bars
 * Market Profile shows TIME spent at each price level (not just volume)
 */
function calculateMarketProfile(bars: TopstepXFuturesBar[]): MarketProfile {
  const priceTimeMap = new Map<number, MarketProfileNode>();

  // Assume each bar represents equal time (e.g., 5 minutes = 300 seconds)
  const timePerBar = 300; // 5 minutes in seconds

  for (const bar of bars) {
    const prices = [bar.open, bar.high, bar.low, bar.close];
    const barTime = new Date(bar.timestamp);

    // Add time to all price levels touched by this bar
    const minPrice = Math.floor(bar.low);
    const maxPrice = Math.ceil(bar.high);

    for (let price = minPrice; price <= maxPrice; price += 0.25) {
      if (price >= bar.low && price <= bar.high) {
        if (!priceTimeMap.has(price)) {
          priceTimeMap.set(price, {
            price,
            timeSpent: 0,
            volume: 0,
            firstTouch: barTime,
            lastTouch: barTime,
          });
        }

        const node = priceTimeMap.get(price)!;
        node.timeSpent += timePerBar;
        node.volume += (bar.volume || 0) / ((maxPrice - minPrice) / 0.25); // Distribute volume
        node.lastTouch = barTime;
      }
    }
  }

  // Convert to array and sort by time spent
  const tpo = Array.from(priceTimeMap.values()).sort((a, b) => b.timeSpent - a.timeSpent);

  if (tpo.length === 0) {
    return {
      poc: 0,
      vah: 0,
      val: 0,
      tpo: [],
      totalTime: 0,
      highVolNode: 0,
    };
  }

  // POC = price with most time
  const poc = tpo[0].price;

  // Calculate Value Area (70% of total time)
  const totalTime = tpo.reduce((sum, node) => sum + node.timeSpent, 0);
  const valueAreaTime = totalTime * 0.7;

  let accumulatedTime = 0;
  let vahIndex = 0;
  let valIndex = tpo.length - 1;

  // Find VAH and VAL
  for (let i = 0; i < tpo.length && accumulatedTime < valueAreaTime; i++) {
    accumulatedTime += tpo[i].timeSpent;
    vahIndex = i;
  }

  const vah = Math.max(...tpo.slice(0, vahIndex + 1).map(n => n.price));
  const val = Math.min(...tpo.slice(0, vahIndex + 1).map(n => n.price));

  // High volume node (traditional volume POC)
  const highVolNode = tpo.sort((a, b) => b.volume - a.volume)[0].price;

  return {
    poc,
    vah,
    val,
    tpo,
    totalTime,
    highVolNode,
  };
}

/**
 * Calculate multi-day volume profile
 */
function calculateMultiDayVolumeProfile(dailyBarsArray: TopstepXFuturesBar[][]): MultiDayVolumeProfile {
  const dailyProfiles: VolumeProfile[] = [];
  const compositeNodes = new Map<number, number>();

  for (const dayBars of dailyBarsArray) {
    if (dayBars.length === 0) continue;

    const volumeMap = new Map<number, number>();

    for (const bar of dayBars) {
      const price = Math.round(bar.close * 4) / 4; // Round to 0.25
      volumeMap.set(price, (volumeMap.get(price) || 0) + (bar.volume || 0));
      compositeNodes.set(price, (compositeNodes.get(price) || 0) + (bar.volume || 0));
    }

    const nodes = Array.from(volumeMap.entries())
      .map(([price, volume]) => ({ price, volume }))
      .sort((a, b) => b.volume - a.volume);

    if (nodes.length === 0) continue;

    const poc = nodes[0].price;
    const totalVolume = nodes.reduce((sum, n) => sum + n.volume, 0);
    const valueAreaVolume = totalVolume * 0.7;

    let accumulatedVolume = 0;
    let vahIndex = 0;
    for (let i = 0; i < nodes.length && accumulatedVolume < valueAreaVolume; i++) {
      accumulatedVolume += nodes[i].volume;
      vahIndex = i;
    }

    const vah = Math.max(...nodes.slice(0, vahIndex + 1).map(n => n.price));
    const val = Math.min(...nodes.slice(0, vahIndex + 1).map(n => n.price));

    dailyProfiles.push({
      poc,
      vah,
      val,
      nodes,
      date: dayBars[0].timestamp,
    });
  }

  // Composite profile
  const compositeNodesArray = Array.from(compositeNodes.entries())
    .map(([price, volume]) => ({ price, volume }))
    .sort((a, b) => b.volume - a.volume);

  const compositePoc = compositeNodesArray[0]?.price || 0;
  const compositeTotalVol = compositeNodesArray.reduce((sum, n) => sum + n.volume, 0);
  const compositeValueVol = compositeTotalVol * 0.7;

  let accVol = 0;
  let compVahIdx = 0;
  for (let i = 0; i < compositeNodesArray.length && accVol < compositeValueVol; i++) {
    accVol += compositeNodesArray[i].volume;
    compVahIdx = i;
  }

  const compositeVah = Math.max(...compositeNodesArray.slice(0, compVahIdx + 1).map(n => n.price));
  const compositeVal = Math.min(...compositeNodesArray.slice(0, compVahIdx + 1).map(n => n.price));

  // Find HVNs and LVNs
  const avgVolume = compositeTotalVol / compositeNodesArray.length;
  const hvns = compositeNodesArray.filter(n => n.volume > avgVolume * 1.5).map(n => n.price);
  const lvns = compositeNodesArray.filter(n => n.volume < avgVolume * 0.5).map(n => n.price);

  return {
    daily: dailyProfiles,
    composite: {
      poc: compositePoc,
      vah: compositeVah,
      val: compositeVal,
      nodes: compositeNodesArray,
      date: 'composite',
    },
    hvns,
    lvns,
  };
}

/**
 * Analyze higher timeframes for intraday position trading
 */
function analyzeIntradayTimeframes(): IntradayTimeframeAnalysis {
  // 4H analysis (bias/direction)
  const fourHourEma20 = calculateEMA(fourHourBars, 20);
  const fourHourEma50 = calculateEMA(fourHourBars, 50);

  const fourHourTrend = fourHourEma20 > fourHourEma50 ? 'bullish' :
                        fourHourEma20 < fourHourEma50 ? 'bearish' : 'neutral';

  const fourHourKeyLevels = findSwingHighsLows(fourHourBars, 3);

  // 1H analysis (primary timeframe for structure)
  const oneHourHigh = Math.max(...oneHourBars.slice(-12).map(b => b.high)); // Last 12 hours
  const oneHourLow = Math.min(...oneHourBars.slice(-12).map(b => b.low));
  const oneHourSwings = findSwingHighsLows(oneHourBars, 2);
  const oneHourEma20 = calculateEMA(oneHourBars, 20);
  const oneHourTrend = oneHourBars[oneHourBars.length - 1]?.close > oneHourEma20 ? 'bullish' : 'bearish';

  // 15min analysis (entry zone)
  const fifteenMinTrend = fifteenMinBars[fifteenMinBars.length - 1]?.close > calculateEMA(fifteenMinBars, 20) ? 'bullish' : 'bearish';
  const entryZone = findEntryZone(fifteenMinBars, fourHourTrend);

  // Momentum check
  const recentMomentum = fifteenMinBars.slice(-4);
  const strongMomentum = recentMomentum.every((bar, i) => i === 0 || bar.close > recentMomentum[i-1].close);
  const weakMomentum = recentMomentum.every((bar, i) => i === 0 || bar.close < recentMomentum[i-1].close);
  const momentum = strongMomentum ? 'strong' : weakMomentum ? 'weak' : 'neutral';

  // 5min analysis (entry trigger)
  const currentPrice = fiveMinBars[fiveMinBars.length - 1]?.close || 0;
  const entryTrigger = checkEntryTrigger(fiveMinBars, entryZone);
  const microStructure = analyzeMicroStructure(fiveMinBars);

  return {
    fourHour: {
      trend: fourHourTrend,
      ema20: fourHourEma20,
      ema50: fourHourEma50,
      structure: `EMA alignment: ${fourHourTrend}`,
      keyLevels: fourHourKeyLevels,
    },
    oneHour: {
      trend: oneHourTrend,
      recentHigh: oneHourHigh,
      recentLow: oneHourLow,
      swingPoints: oneHourSwings,
      ema20: oneHourEma20,
    },
    fifteenMin: {
      trend: fifteenMinTrend,
      entryZone,
      momentum,
    },
    fiveMin: {
      currentPrice,
      entryTrigger,
      microStructure,
    },
  };
}

function calculateEMA(bars: TopstepXFuturesBar[], period: number): number {
  if (bars.length < period) return bars[bars.length - 1]?.close || 0;

  const multiplier = 2 / (period + 1);
  let ema = bars.slice(0, period).reduce((sum, b) => sum + b.close, 0) / period;

  for (let i = period; i < bars.length; i++) {
    ema = (bars[i].close - ema) * multiplier + ema;
  }

  return ema;
}

function findSwingHighsLows(bars: TopstepXFuturesBar[], lookback: number): number[] {
  const swings: number[] = [];

  for (let i = lookback; i < bars.length - lookback; i++) {
    const isHigh = bars.slice(i - lookback, i).every(b => b.high < bars[i].high) &&
                   bars.slice(i + 1, i + lookback + 1).every(b => b.high < bars[i].high);
    const isLow = bars.slice(i - lookback, i).every(b => b.low > bars[i].low) &&
                  bars.slice(i + 1, i + lookback + 1).every(b => b.low > bars[i].low);

    if (isHigh) swings.push(bars[i].high);
    if (isLow) swings.push(bars[i].low);
  }

  return swings.slice(-10); // Return last 10 swing points
}

function findEntryZone(bars: TopstepXFuturesBar[], dailyTrend: string): { low: number; high: number } | null {
  if (bars.length < 20) return null;

  const ema20 = calculateEMA(bars, 20);
  const recentLow = Math.min(...bars.slice(-10).map(b => b.low));
  const recentHigh = Math.max(...bars.slice(-10).map(b => b.high));

  if (dailyTrend === 'bullish') {
    // Look for pullbacks to EMA for long entries
    return { low: ema20 - 5, high: ema20 + 5 };
  } else if (dailyTrend === 'bearish') {
    // Look for rallies to EMA for short entries
    return { low: ema20 - 5, high: ema20 + 5 };
  }

  return null;
}

function checkEntryTrigger(bars: TopstepXFuturesBar[], entryZone: { low: number; high: number } | null): boolean {
  if (!entryZone || bars.length === 0) return false;

  const currentPrice = bars[bars.length - 1].close;
  return currentPrice >= entryZone.low && currentPrice <= entryZone.high;
}

function analyzeMicroStructure(bars: TopstepXFuturesBar[]): string {
  if (bars.length < 10) return 'Insufficient data';

  const recentBars = bars.slice(-10);
  const higherHighs = recentBars.slice(1).every((bar, i) => bar.high > recentBars[i].high);
  const lowerLows = recentBars.slice(1).every((bar, i) => bar.low < recentBars[i].low);

  if (higherHighs) return 'Strong uptrend';
  if (lowerLows) return 'Strong downtrend';

  return 'Consolidating';
}

/**
 * Main analysis loop - runs every 2 minutes (intraday position trading)
 */
async function processIntradayAnalysis() {
  const nowMs = Date.now();

  if (fiveMinBars.length === 0) return;

  log(`🔄 Processing intraday position analysis (${fiveMinBars.length} bars)...`);

  // Calculate market profile for current session
  if (fiveMinBars.length > 0) {
    marketProfile = calculateMarketProfile(fiveMinBars.slice(-180)); // Last 15 hours of 5min bars (current session)
  }

  // Calculate session volume profile (current + previous session)
  if (fiveMinBars.length >= 100) {
    const todayBars = fiveMinBars.slice(-100); // Last ~8 hours
    const yesterdayBars = fiveMinBars.slice(-200, -100); // Previous ~8 hours
    const sessionBarsGrouped: TopstepXFuturesBar[][] = [todayBars, yesterdayBars].filter(b => b.length > 0);
    sessionVolumeProfile = calculateMultiDayVolumeProfile(sessionBarsGrouped);
  }

  // Analyze intraday timeframes
  const intradayAnalysis = analyzeIntradayTimeframes();

  log(`📊 Intraday Analysis: 4H=${intradayAnalysis.fourHour.trend}, 1H=${intradayAnalysis.oneHour.trend}, 15m=${intradayAnalysis.fifteenMin.trend}, Price=${intradayAnalysis.fiveMin.currentPrice.toFixed(2)}`);

  // Sync all positions from ExecutionManager
  if (executionManager) {
    const previousPositionCount = activePositions.size;

    // Get all active positions from execution manager
    // Note: ExecutionManager currently tracks one position, but we'll enhance it
    const currentPosition = executionManager.getActivePosition();

    // Clear and rebuild active positions map
    activePositions.clear();

    if (currentPosition) {
      activePositions.set(currentPosition.decisionId, currentPosition);
    }

    // Log position changes
    if (activePositions.size > previousPositionCount) {
      log(`🔄 [PositionSync] New position opened (${activePositions.size} total active)`);
    } else if (activePositions.size < previousPositionCount) {
      log(`🔄 [PositionSync] Position closed (${activePositions.size} remaining)`);
    }

    // Log all active positions
    if (activePositions.size > 0) {
      const positionsStr = Array.from(activePositions.values())
        .map(p => `${p.side.toUpperCase()} @ ${p.entryPrice.toFixed(2)}`)
        .join(', ');
      log(`📊 [Positions] Active: ${positionsStr}`);
    }
  }

  // Check for position exits and run risk management for ALL active positions
  if (executionManager && activePositions.size > 0) {
    const currentPrice = fiveMinBars[fiveMinBars.length - 1].close;

    // Process each position independently
    for (const [decisionId, position] of activePositions.entries()) {
      // Check if broker closed this position
      const exitResult = await updatePositionAndCheckExits(executionManager, currentPrice, fiveMinBars);

      if (exitResult.exited && exitResult.closedDecisionId === decisionId) {
        // Self-learning bypassed - calculate P&L directly without database
        const profitLoss = position.side === 'long'
          ? (currentPrice - position.entryPrice) * 5 * position.contracts
          : (position.entryPrice - currentPrice) * 5 * position.contracts;
        realizedPnL += profitLoss;
        log(`📊 Position #${decisionId.slice(-6)} closed: ${profitLoss >= 0 ? '+' : ''}$${profitLoss.toFixed(2)} P&L`, profitLoss >= 0 ? 'success' : 'error');

        // Remove from active positions
        activePositions.delete(decisionId);
        positionRiskDecisions.delete(decisionId);
      }
    }

    // Run strategic risk management for all active positions
    const timeSinceLastRiskMgmt = nowMs - lastRiskMgmtTime;
    if (timeSinceLastRiskMgmt >= RISK_MGMT_INTERVAL_MS && activePositions.size > 0) {
      lastRiskMgmtTime = nowMs;

      log(`🛡️ [RiskMgmt] Managing ${activePositions.size} active positions strategically...`);

      // Build context of all positions for correlation-aware risk management
      const allPositionsContext = Array.from(activePositions.values()).map(p => ({
        id: p.decisionId,
        side: p.side,
        entryPrice: p.entryPrice,
        stopLoss: p.stopLoss,
        target: p.target,
        contracts: p.contracts,
        pnl: p.side === 'long'
          ? (currentPrice - p.entryPrice) * 5 * p.contracts
          : (p.entryPrice - currentPrice) * 5 * p.contracts,
      }));

      // Manage each position with awareness of the portfolio
      for (const [decisionId, position] of activePositions.entries()) {
        try {
          log(`🛡️ [RiskMgmt] Analyzing position #${decisionId.slice(-6)} (${position.side.toUpperCase()} @ ${position.entryPrice.toFixed(2)})...`);

          const riskDecision = await analyzePositionRisk(position, {
            currentPrice,
            recentBars: fiveMinBars.slice(-20),
            cvd: 0,
            cvdTrend: 'neutral',
            orderFlowPressure: 'neutral',
            volumeProfile: marketProfile ? {
              poc: marketProfile.poc,
              vah: marketProfile.vah,
              val: marketProfile.val,
            } : undefined,
            marketStructure: intradayAnalysis.oneHour.trend,
            allPositionsContext: POSITION_CORRELATION_AWARE ? allPositionsContext : undefined, // Share portfolio context
          });

          log(`🛡️ [RiskMgmt] Position #${decisionId.slice(-6)} Decision: ${riskDecision.action}`);

          positionRiskDecisions.set(decisionId, {
            timestamp: new Date().toISOString(),
            action: riskDecision.action,
            urgency: riskDecision.urgency,
            reasoning: riskDecision.reasoning,
            newStopLoss: riskDecision.newStopLoss,
            newTarget: riskDecision.newTarget,
          });

          // Apply decision
          if (riskDecision.action === 'CLOSE_POSITION') {
            log(`🛡️ [RiskMgmt] Closing position #${decisionId.slice(-6)} based on risk analysis`);
            const closedId = await executionManager.closePosition(
              decisionId,
              currentPrice,
              'risk_management_close'
            );

            if (closedId) {
              log(`🛡️ [RiskMgmt] ✅ Position #${decisionId.slice(-6)} closed`);
              activePositions.delete(decisionId);
              positionRiskDecisions.delete(decisionId);
            }
          } else if (riskDecision.action !== 'HOLD_BRACKETS') {
            const adjusted = await executionManager.adjustActiveProtection(
              riskDecision.newStopLoss,
              riskDecision.newTarget
            );

            if (adjusted) {
              log(`🛡️ [RiskMgmt] ✅ Position #${decisionId.slice(-6)} brackets adjusted`);
              // Refresh position data
              const updatedPosition = executionManager.getActivePosition();
              if (updatedPosition && updatedPosition.decisionId === decisionId) {
                activePositions.set(decisionId, updatedPosition);
              }
            }
          }
        } catch (error: any) {
          log(`🛡️ [RiskMgmt] Error managing position #${decisionId.slice(-6)}: ${error.message}`, 'error');
        }
      }
    }
  }

  // Analyze for new trades if we have room for more positions (up to MAX_CONCURRENT_POSITIONS)
  const timeSinceLastAnalysis = nowMs - lastAnalysisTime;
  const hasRoomForMorePositions = activePositions.size < MAX_CONCURRENT_POSITIONS;
  const shouldAnalyze = timeSinceLastAnalysis >= ANALYSIS_INTERVAL_MS && hasRoomForMorePositions;

  if (shouldAnalyze) {
    lastAnalysisTime = nowMs;

    try {
      log(`🧠 [Intraday] Analyzing for position opportunities (${activePositions.size}/${MAX_CONCURRENT_POSITIONS} positions)...`);

      // Build market data for intraday position trading with multi-position awareness
      const intradayMarketData = buildIntradayMarketData(intradayAnalysis);

      // Add current portfolio context for strategic positioning
      intradayMarketData.currentPortfolio = {
        activePositions: activePositions.size,
        maxPositions: MAX_CONCURRENT_POSITIONS,
        positions: Array.from(activePositions.values()).map(p => ({
          side: p.side,
          entryPrice: p.entryPrice,
          stopLoss: p.stopLoss,
          target: p.target,
        })),
        totalPnL: Array.from(activePositions.values()).reduce((sum, p) => {
          const pnl = p.side === 'long'
            ? (fiveMinBars[fiveMinBars.length - 1].close - p.entryPrice) * 5 * p.contracts
            : (p.entryPrice - fiveMinBars[fiveMinBars.length - 1].close) * 5 * p.contracts;
          return sum + pnl;
        }, 0),
      };

      // AI decides if/how to enter, considering correlation with existing positions
      const decision = await analyzeFuturesMarket(intradayMarketData);

      if (decision && decision.decision !== 'HOLD') {
        log(`🎯 [Intraday] AI suggests ${decision.decision} @ ${decision.entryPrice?.toFixed(2)} | Confidence: ${decision.confidence}%`);

        // Strategic positioning: AI can place stops/targets flexibly
        if (STRATEGIC_POSITIONING) {
          log(`🎯 [Strategic] Stop: ${decision.stopLoss?.toFixed(2)}, Target: ${decision.target?.toFixed(2)}`);
        }

        // Execute trade if confidence is high enough (55%+ for multi-position portfolio)
        if (decision.confidence && decision.confidence >= 55) {
          log(`🚀 [Intraday] Executing position #${activePositions.size + 1}...`);

          const executionResult = await executionManager!.executeDecision(
            {
              side: decision.decision.toLowerCase() as 'buy' | 'sell',
              entry: { side: decision.decision.toLowerCase() as 'buy' | 'sell' },
            },
            fiveMinBars[fiveMinBars.length - 1].close,
            {
              stopLoss: decision.stopLoss,
              takeProfit: decision.target,
              entryPrice: decision.entryPrice,
            }
          );

          if (executionResult.executed) {
            const newPosition = executionManager!.getActivePosition();
            if (newPosition) {
              activePositions.set(newPosition.decisionId, newPosition);
              log(`✅ [Intraday] Position #${activePositions.size} opened: ${newPosition.side.toUpperCase()} @ ${newPosition.entryPrice.toFixed(2)}`, 'success');
              log(`📊 [Portfolio] Now managing ${activePositions.size} positions strategically`);
            }
          }
        } else {
          log(`⏸️ [Intraday] Confidence ${decision.confidence}% below threshold (55%), waiting for better setup`);
        }
      }
    } catch (error: any) {
      log(`❌ [Intraday] Analysis error: ${error.message}`, 'error');
    }
  }

  // Broadcast dashboard update
  broadcastDashboardUpdate();
}

function buildIntradayMarketData(intradayAnalysis: IntradayTimeframeAnalysis): any {
  return {
    symbol: SYMBOL,
    timeframe: 'INTRADAY_POSITION',
    currentPrice: intradayAnalysis.fiveMin.currentPrice,
    fourHourTrend: intradayAnalysis.fourHour.trend,
    oneHourTrend: intradayAnalysis.oneHour.trend,
    fifteenMinTrend: intradayAnalysis.fifteenMin.trend,
    momentum: intradayAnalysis.fifteenMin.momentum,
    marketProfile: marketProfile,
    sessionVolumeProfile: sessionVolumeProfile,
    swingLevels: intradayAnalysis.oneHour.swingPoints,
    entryZone: intradayAnalysis.fifteenMin.entryZone,
    microStructure: intradayAnalysis.fiveMin.microStructure,
  };
}

async function updatePositionAndCheckExits(
  manager: ExecutionManager,
  currentPrice: number,
  bars: TopstepXFuturesBar[]
): Promise<{ exited: boolean; closedDecisionId?: string }> {
  // For swing trades, we rely on broker execution of brackets
  // Just check if position still exists
  const position = manager.getActivePosition();
  if (!position) {
    return { exited: true };
  }

  return { exited: false };
}

function broadcastDashboardUpdate() {
  if (!io) return;

  const currentBar = fiveMinBars[fiveMinBars.length - 1];

  if (currentBar) {
    io.emit('bar', {
      timestamp: currentBar.timestamp,
      open: currentBar.open,
      high: currentBar.high,
      low: currentBar.low,
      close: currentBar.close,
      volume: currentBar.volume || 0,
    });
  }

  // Convert active positions map to array for dashboard
  const positionsArray = Array.from(activePositions.values()).map(pos => ({
    id: pos.decisionId,
    side: pos.side,
    entry_price: pos.entryPrice,
    contracts: pos.contracts,
    pnl: pos.side === 'long'
      ? ((currentBar?.close || 0) - pos.entryPrice) * 5 * pos.contracts
      : (pos.entryPrice - (currentBar?.close || 0)) * 5 * pos.contracts,
    stop_loss: pos.stopLoss,
    target: pos.target,
    entryTime: pos.entryTime,
    riskDecision: positionRiskDecisions.get(pos.decisionId),
  }));

  // Calculate total portfolio P&L
  const totalPnL = positionsArray.reduce((sum, pos) => sum + pos.pnl, 0);

  io.emit('status', {
    balance: accountBalance,
    positions: positionsArray, // Array of all active positions
    positionCount: activePositions.size,
    maxPositions: MAX_CONCURRENT_POSITIONS,
    totalPortfolioPnL: totalPnL,
    market_profile: marketProfile,
    session_vp: sessionVolumeProfile,
    daily_pnl: realizedPnL,
    strategicPositioning: STRATEGIC_POSITIONING,
    correlationAware: POSITION_CORRELATION_AWARE,
  });
}

function initDashboard() {
  server.listen(DASHBOARD_PORT, () => {
    log(`📊 Intraday Position Trading Dashboard running at http://localhost:${DASHBOARD_PORT}`);
  });

  io.on('connection', (socket) => {
    log('📱 Dashboard client connected');

    socket.on('disconnect', () => {
      log('📱 Dashboard client disconnected');
    });
  });
}

// Load historical data for all timeframes
async function loadHistoricalData() {
  log('Loading historical data...');

  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - LOOKBACK_DAYS * 24 * 60 * 60 * 1000);

  try {
    // Load 1-minute bars
    const oneMinData = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 60,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });
    bars = oneMinData;
    log(`✅ Loaded ${bars.length} 1-minute bars`);

    // Load 5-minute bars
    const fiveMinData = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 300,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });
    fiveMinBars = fiveMinData;
    log(`✅ Loaded ${fiveMinBars.length} 5-minute bars`);

    // Load 15-minute bars
    const fifteenMinData = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 900,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });
    fifteenMinBars = fifteenMinData;
    log(`✅ Loaded ${fifteenMinBars.length} 15-minute bars`);

    // Load 1-hour bars
    const oneHourData = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 3600,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });
    oneHourBars = oneHourData;
    log(`✅ Loaded ${oneHourBars.length} 1-hour bars`);

    // Load 4-hour bars
    const fourHourData = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 14400,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });
    fourHourBars = fourHourData;
    log(`✅ Loaded ${fourHourBars.length} 4-hour bars`);

  } catch (error: any) {
    log(`Error loading historical data: ${error.message}`, 'error');
    throw error;
  }
}

// Refresh bars with latest data
async function refreshBars() {
  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - 60 * 60 * 1000); // Last hour

  try {
    // Refresh 1-minute bars
    const newBars = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 60,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });

    // Merge new bars
    for (const bar of newBars) {
      const existingIndex = bars.findIndex(b => b.timestamp === bar.timestamp);
      if (existingIndex >= 0) {
        bars[existingIndex] = bar; // Update
      } else {
        bars.push(bar); // Add new
      }
    }
    bars.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    if (bars.length > 1000) bars = bars.slice(-1000);

    // Refresh 5-minute bars
    const newFiveMinBars = await fetchTopstepXFuturesBars({
      contractId,
      periodSeconds: 300,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
    });
    for (const bar of newFiveMinBars) {
      const existingIndex = fiveMinBars.findIndex(b => b.timestamp === bar.timestamp);
      if (existingIndex >= 0) {
        fiveMinBars[existingIndex] = bar;
      } else {
        fiveMinBars.push(bar);
      }
    }
    fiveMinBars.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    if (fiveMinBars.length > 1000) fiveMinBars = fiveMinBars.slice(-1000);

  } catch (error: any) {
    log(`Error refreshing bars: ${error.message}`, 'error');
  }
}

async function main() {
  log('================================================================================');
  log('🧠 FABIO INTRADAY POSITION TRADING - MES (MULTI-POSITION STRATEGIC)');
  log('================================================================================');
  log(`Symbol: ${SYMBOL}`);
  log(`Timeframes: 1H (primary), 15min/5min (secondary), 1min (entry)`);
  log(`Analysis Interval: Every 2 minutes`);
  log(`Lookback Period: ${LOOKBACK_DAYS} sessions (current + previous)`);
  log(`Hold Time: ${TYPICAL_HOLD_HOURS} hours typical (max ${MAX_HOLD_DAYS} days)`);
  log(`Target: ${MIN_TARGET_POINTS}-150 points | Stop: 30-${MAX_STOP_POINTS} points (flexible)`);
  log(``);
  log(`🎯 STRATEGIC MULTI-POSITION CAPABILITIES:`);
  log(`   - Max Concurrent Positions: ${MAX_CONCURRENT_POSITIONS}`);
  log(`   - Strategic Stop/Target Placement: ${STRATEGIC_POSITIONING ? 'ENABLED' : 'DISABLED'}`);
  log(`   - Position Correlation Awareness: ${POSITION_CORRELATION_AWARE ? 'ENABLED' : 'DISABLED'}`);
  log(`   - OCO Brackets: All positions managed independently`);
  log(`   - Risk Management: Strategic per-position analysis`);
  log(``);
  log(`Dashboard: http://localhost:${DASHBOARD_PORT}`);
  log(`Self-Learning: BYPASSED (pure AI analysis, no historical database)`);
  log('================================================================================');

  initDashboard();

  // Resolve contract ID
  const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
  if (!metadata) {
    throw new Error(`Failed to resolve contract for symbol ${SYMBOL}`);
  }
  contractId = metadata.id;
  log(`Resolved ${SYMBOL} to contractId=${contractId}`);

  // Load historical data
  await loadHistoricalData();
  log('✅ Loaded historical bars for all timeframes');

  // Create order manager
  const { orderManager: om } = await createProjectXRest();
  orderManager = om;
  log('✅ Order Manager initialized');

  // Initialize execution manager (needs the order manager)
  executionManager = await createExecutionManager(orderManager, parseInt(contractId), ACCOUNT_ID);
  log('✅ Execution Manager initialized');

  // Start periodic analysis (every 2 minutes)
  setInterval(async () => {
    try {
      await refreshBars();
      await processIntradayAnalysis();
    } catch (err: any) {
      log(`Error in periodic analysis: ${err.message}`, 'error');
    }
  }, ANALYSIS_INTERVAL_MS);

  // Initial analysis
  await processIntradayAnalysis();

  log('🚀 Intraday position trading agent is running...');
  log('📈 Looking for 2-5 quality setups per day with 50-150 point targets');
}

main().catch(err => {
  log(`Fatal error: ${err.message}`, 'error');
  process.exit(1);
});
