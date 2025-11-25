#!/usr/bin/env tsx
/**
 * Fabio Agent - Volume Profile & Order Flow Trading
 * Based on Fabio's Playbook with Level 2 data and decision making
 */

import 'dotenv/config';
import { fabioPlaybook, MarketState, SetupModel } from './lib/fabioPlaybook';
import {
  fetchTopstepXFuturesBars,
  fetchTopstepXFuturesMetadata,
  fetchTopstepXAccounts,
  TopstepXFuturesBar,
  authenticate,
} from './lib/topstepx';
import { createProjectXRest } from './projectx-rest';
import { HubConnection, HubConnectionBuilder, HttpTransportType, LogLevel } from '@microsoft/signalr';
import express from 'express';
import cors from 'cors';
import { Server } from 'socket.io';
import http from 'http';
import { writeFileSync, readFileSync, existsSync } from 'fs';
import {
  buildFuturesMarketData,
  processOpenAIDecision,
  updatePositionAndCheckExits,
  getTradeStats,
  logTradeStats,
  analyzeConfidenceCalibration,
  ExecutionManager,
  HigherTimeframeSnapshot,
  SessionVolumeProfileSummary,
} from './lib/fabioOpenAIIntegration';
import { analyzeFuturesMarket } from './lib/openaiTradingAgent';
import { createExecutionManager } from './lib/executionManager';
import { analyzePositionRisk } from './lib/riskManagementAgent';

// Configuration - GOLD (MGC) Instance - Micro Gold
const SYMBOL = process.env.TOPSTEPX_SYMBOL || 'MGCZ5';
const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '0');
const DASHBOARD_PORT = 3338; // Different port for GC instance
const CONTRACTS = 1;
// Analyze with DeepSeek Reasoner every 60 seconds (same cadence as NQ)
const ANALYSIS_INTERVAL_MS = 60_000;
const RISK_MGMT_INTERVAL_MS = 3_000; // Risk management checks every 3 seconds (aggressive stop tightening)

// Volume Profile Types
interface VolumeNode {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
}

interface VolumeProfile {
  nodes: VolumeNode[];
  poc: number; // Point of Control
  vah: number; // Value Area High
  val: number; // Value Area Low
  lvns: number[]; // Low Volume Nodes
}

// Level 2 Data Types
interface L2Level {
  price: number;
  bidSize: number;
  askSize: number;
}

interface OrderFlowData {
  bigPrints: Array<{ price: number; size: number; side: 'buy' | 'sell'; timestamp: number }>;
  cvd: number; // Cumulative Volume Delta
  footprintImbalance: { [price: number]: number }; // Delta at each price
  absorption: { buy: number; sell: number };
  exhaustion: { buy: number; sell: number };
  cvdHistory: Array<{ timestamp: number; cvd: number; delta: number }>;
  volumeAtPrice: { [price: number]: { buy: number; sell: number; timestamp: number } }; // Volume concentration
}

// Market Structure
interface MarketStructure {
  state: MarketState;
  impulseLegs: Array<{
    direction: 'up' | 'down';
    start: number;
    end: number;
    startTime: string;
    endTime: string;
  }>;
  balanceAreas: Array<{
    poc: number;
    vah: number;
    val: number;
    startTime: string;
    endTime: string;
  }>;
  failedBreakouts: Array<{
    direction: 'above' | 'below';
    level: number;
    timestamp: string;
  }>;
}

// Agent Decision State
interface AgentDecision {
  model: SetupModel | null;
  marketState: MarketState;
  location: 'at_lvn' | 'at_poc' | 'at_vah' | 'at_val' | 'neutral';
  orderFlowConfirmation: boolean;
  entry: {
    side: 'long' | 'short' | null;
    reason: string;
    confidence: number; // 0-100
  };
  riskManagement: {
    stopLoss: number;
    target: number;
    riskAmount: number;
  };
}

// Global State
let marketHub: HubConnection | null = null;
let userHub: HubConnection | null = null;
let orderManager: any = null;
let io: Server;
let app: express.Application;
let server: http.Server;
let contractId: string = ''; // Resolved TopstepX contract ID for SYMBOL
let resolvedContractId: string | null = null; // Prefer contractId learned from live position feed

// Market Data Storage
let bars: TopstepXFuturesBar[] = [];  // Real-time 1-min bars built from trades
let historicalSessionBars: TopstepXFuturesBar[] = []; // Historical 10-sec bars from session start
let l2Data: L2Level[] = [];
let orderFlowData: OrderFlowData = {
  bigPrints: [],
  cvd: 0,
  footprintImbalance: {},
  absorption: { buy: 0, sell: 0 },
  exhaustion: { buy: 0, sell: 0 },
  cvdHistory: [],
  volumeAtPrice: {},
};

// CVD OHLC tracking for current 5-minute bar
let currentCvdBar: {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
} | null = null;
// Track contractId from live position feed to accept market data even if REST resolve fails
const noteContractIdFromPosition = (pos: any) => {
  if (pos?.contractId) {
    resolvedContractId = String(pos.contractId);
  }
};
let volumeProfile: VolumeProfile | null = null;
let marketStructure: MarketStructure = {
  state: 'balanced',
  impulseLegs: [],
  balanceAreas: [],
  failedBreakouts: [],
};
let currentDecision: AgentDecision | null = null;
let higherTimeframeSnapshots: HigherTimeframeSnapshot[] = [];
let recentSessionProfiles: SessionVolumeProfileSummary[] = [];
let lastHigherTimeframeRefresh = 0;
let lastVolumeProfileRefresh = 0;
let cvdMinuteBars: CurrentCvdBar[] = [];

// Position Management
let currentPosition: any = null;
let lastRiskMgmtTime = Date.now(); // Initialize to now to prevent immediate spam
let lastRiskMgmtDecision: any = null; // Store last Risk Management decision for dashboard
let accountBalance = 50000;

// OpenAI + Execution Integration
let executionManager: ExecutionManager | null = null;
let realizedPnL = 0; // Track realized P&L from closed positions

// OpenAI Rate Limiting - Only call once per completed 5-minute candle
let lastOpenAIAnalysisTime = 0;

// Utility Functions
function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}][FABIO] ${message}`);

  if (io) {
    io.emit('log', { timestamp, message, type });
  }
}

// Calculate Absorption: Detects when large volume hits a level without price breaking through
function calculateAbsorption(): { buy: number; sell: number } {
  const now = Date.now();
  const recentWindow = 60000; // 60 seconds

  // Clean up old volume data
  Object.keys(orderFlowData.volumeAtPrice).forEach(priceKey => {
    const price = parseFloat(priceKey);
    if (now - orderFlowData.volumeAtPrice[price].timestamp > recentWindow) {
      delete orderFlowData.volumeAtPrice[price];
    }
  });

  if (Object.keys(orderFlowData.volumeAtPrice).length === 0) {
    return { buy: 0, sell: 0 };
  }

  // Find price levels with high volume concentration
  const volumeEntries = Object.entries(orderFlowData.volumeAtPrice)
    .map(([priceStr, vol]) => ({
      price: parseFloat(priceStr),
      ...vol,
    }));

  if (volumeEntries.length < 3) {
    return { buy: 0, sell: 0 };
  }

  const currentPrice = bars[bars.length - 1]?.close || 0;

  // Calculate total volumes
  const totalBuyVolume = volumeEntries.reduce((sum, v) => sum + v.buy, 0);
  const totalSellVolume = volumeEntries.reduce((sum, v) => sum + v.sell, 0);
  const totalVolume = totalBuyVolume + totalSellVolume;

  if (totalVolume === 0) {
    return { buy: 0, sell: 0 };
  }

  // Calculate volume ratios
  const buyRatio = totalBuyVolume / totalVolume;
  const sellRatio = totalSellVolume / totalVolume;

  // Absorption requires BOTH volume AND price resistance
  // Scale absorption based on volume concentration and recent price action
  const recentBars = bars.slice(-5);
  if (recentBars.length < 2) {
    return { buy: 0, sell: 0 };
  }

  const priceChange = recentBars[recentBars.length - 1].close - recentBars[0].open;
  const priceRange = Math.max(...recentBars.map(b => b.high)) - Math.min(...recentBars.map(b => b.low));

  // Absorption = high volume + low price movement (resistance)
  // If price is moving a lot, it's trending not absorbing
  const priceStability = priceRange > 0 ? 1 - Math.min(1, Math.abs(priceChange) / priceRange) : 0;

  // Buy absorption: Buy volume proportion + price stability
  // Shows on both sides simultaneously - balanced markets show ~50% on each side
  const buyAbsorptionStrength = buyRatio * priceStability;

  // Sell absorption: Sell volume proportion + price stability
  // Shows on both sides simultaneously - balanced markets show ~50% on each side
  const sellAbsorptionStrength = sellRatio * priceStability;

  return {
    buy: buyAbsorptionStrength,
    sell: sellAbsorptionStrength,
  };
}

// Calculate Exhaustion: Detects when momentum is dying out
function calculateExhaustion(): { buy: number; sell: number } {
  const HISTORY_WINDOW = 80;
  const MIN_EVENTS = 12;
  const SHORT_PERIOD = 5;
  const LONG_PERIOD = 18;
  const SMOOTHING = 0.35; // Dampens visual flicker

  const history = orderFlowData.cvdHistory.slice(-HISTORY_WINDOW);
  if (history.length < MIN_EVENTS) {
    return orderFlowData.exhaustion || { buy: 0, sell: 0 };
  }

  const buySeries = history
    .filter(h => h.delta > 0)
    .map(h => h.delta);
  const sellSeries = history
    .filter(h => h.delta < 0)
    .map(h => Math.abs(h.delta));

  const buyTarget = computeMomentumFadeScore(buySeries, SHORT_PERIOD, LONG_PERIOD);
  const sellTarget = computeMomentumFadeScore(sellSeries, SHORT_PERIOD, LONG_PERIOD);

  const previous = orderFlowData.exhaustion || { buy: 0, sell: 0 };

  return {
    buy: previous.buy + (buyTarget - previous.buy) * SMOOTHING,
    sell: previous.sell + (sellTarget - previous.sell) * SMOOTHING,
  };
}

function computeMomentumFadeScore(
  series: number[],
  shortPeriod: number,
  longPeriod: number
): number {
  if (series.length < 4) {
    return 0;
  }

  const short = computeEMA(series, shortPeriod);
  const long = computeEMA(series, longPeriod);

  if (long <= 0) {
    return 0;
  }

  // Positive score when the fast momentum falls below the slower average
  const declineRatio = Math.max(0, (long - short) / long);
  return Math.min(1, declineRatio * 1.3);
}

function computeEMA(values: number[], period: number): number {
  if (values.length === 0) return 0;
  const smoothing = 2 / (period + 1);
  let ema = values[0];

  for (let i = 1; i < values.length; i += 1) {
    ema = values[i] * smoothing + ema * (1 - smoothing);
  }

  return ema;
}

// Process L2 Data - Extract walls, whale prints, and detect wall pulls
let previousL2Walls: Array<{ side: 'bid' | 'ask'; price: number; size: number }> = [];

function processL2Data(l2Levels: L2Level[], currentPrice: number): {
  walls: Array<{ side: 'bid' | 'ask'; price: number; size: number }>;
  whalePrints: Array<{ side: 'bid' | 'ask'; size: number; price: number }>;
  wallPullDetected: boolean;
} {
  if (!l2Levels || l2Levels.length === 0) {
    return { walls: [], whalePrints: [], wallPullDetected: false };
  }

  // Calculate average sizes for comparison
  const bidSizes = l2Levels.map(l => l.bidSize || 0).filter(s => s > 0);
  const askSizes = l2Levels.map(l => l.askSize || 0).filter(s => s > 0);

  const avgBidSize = bidSizes.length > 0 ? bidSizes.reduce((a, b) => a + b, 0) / bidSizes.length : 0;
  const avgAskSize = askSizes.length > 0 ? askSizes.reduce((a, b) => a + b, 0) / askSizes.length : 0;

  // Identify walls (orders significantly larger than average)
  const WALL_THRESHOLD = 3; // 3x average size
  const walls: Array<{ side: 'bid' | 'ask'; price: number; size: number }> = [];

  l2Levels.forEach(level => {
    if (level.bidSize > avgBidSize * WALL_THRESHOLD && level.bidSize > 50) {
      walls.push({ side: 'bid', price: level.price, size: level.bidSize });
    }
    if (level.askSize > avgAskSize * WALL_THRESHOLD && level.askSize > 50) {
      walls.push({ side: 'ask', price: level.price, size: level.askSize });
    }
  });

  // Detect wall pulls (large orders that disappeared)
  let wallPullDetected = false;
  if (previousL2Walls.length > 0) {
    for (const prevWall of previousL2Walls) {
      // Check if this wall still exists with similar size
      const stillExists = walls.some(
        w => w.side === prevWall.side &&
             Math.abs(w.price - prevWall.price) < 1 &&
             w.size >= prevWall.size * 0.5
      );
      if (!stillExists && prevWall.size > 100) {
        wallPullDetected = true;
        break;
      }
    }
  }
  previousL2Walls = walls;

  // Extract whale prints from order flow data (already tracked separately)
  const whalePrints = (orderFlowData.bigPrints || []).slice(-5);

  return {
    walls,
    whalePrints,
    wallPullDetected,
  };
}

// Emit LLM Decision
function emitLLMDecision(analysis: any) {
  const normalizedConfidence = typeof analysis.confidence === 'number'
    ? Math.max(0, Math.min(1, analysis.confidence > 1 ? analysis.confidence / 100 : analysis.confidence))
    : 0;

  const decision = {
    timestamp: new Date().toISOString(),
    marketState: analysis.marketState,
    model: analysis.model,
    location: analysis.location,
    orderFlow: analysis.orderFlow,
    decision: analysis.decision,
    reasoning: analysis.reasoning,
    confidence: normalizedConfidence,
  };

  log(`🤖 LLM Analysis: ${analysis.reasoning}`, 'info');

  if (io) {
    io.emit('llm_decision', decision);
  }
}

// Build Risk Snapshot for Risk Management Agent
function buildRiskSnapshot(
  position: any,
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile,
  orderFlowData: any,
  marketStructure: any,
  currentCvdBar: CurrentCvdBar | null,
  l2Data: any
): any {
  const currentBar = bars[bars.length - 1];
  const currentPrice = currentBar?.close || position.entryPrice;

  // Process L2 data to extract walls, whale prints, wall pulls
  const l2Analysis = processL2Data(l2Data, currentPrice);

  return {
    currentPrice,
    recentBars: bars.slice(-20), // Last 20 bars
    cvd: orderFlowData.cvd,
    cvdTrend: orderFlowData.cvdTrend,
    orderFlowPressure: orderFlowData.cvdTrend === 'up' ? 'bullish' : orderFlowData.cvdTrend === 'down' ? 'bearish' : 'neutral',
    volumeProfile: {
      poc: volumeProfile.poc,
      vah: volumeProfile.vah,
      val: volumeProfile.val,
    },
    whaleActivity: orderFlowData.whaleActivity || 'No significant whale activity',
    marketStructure: marketStructure?.state || 'unknown',
    distToPoc: currentPrice - volumeProfile.poc,
    distToVah: currentPrice - volumeProfile.vah,
    distToVal: currentPrice - volumeProfile.val,
    deltaLast1m: orderFlowData.delta1m,
    deltaLast5m: orderFlowData.delta5m,
    cvdSlopeShort: currentCvdBar?.cvdSlope,
    cvdSlopeLong: orderFlowData.cvdSlope,
    cvdDivergence: orderFlowData.cvdDivergence || 'none',
    absorptionZone: orderFlowData.absorption,
    exhaustionFlag: orderFlowData.exhaustion,
    largePrints: l2Analysis.whalePrints,
    restingLiquidityWalls: l2Analysis.walls,
    liquidityPullDetected: l2Analysis.wallPullDetected,
    volRegime: marketStructure?.volatilityRegime || 'normal',
    structureState: marketStructure?.state,
  };
}

// Calculate Volume Profile
function calculateVolumeProfile(bars: TopstepXFuturesBar[]): VolumeProfile {
  const priceVolume = new Map<number, VolumeNode>();

  // Aggregate volume at each price level
  bars.forEach(bar => {
    const prices = [bar.open, bar.high, bar.low, bar.close];
    const volumePerPrice = bar.volume / 4; // Simple distribution

    prices.forEach(price => {
      const roundedPrice = Math.round(price * 4) / 4; // Round to tick
      const node = priceVolume.get(roundedPrice) || {
        price: roundedPrice,
        volume: 0,
        buyVolume: 0,
        sellVolume: 0,
      };

      node.volume += volumePerPrice;
      // Estimate buy/sell based on close vs open
      if (bar.close > bar.open) {
        node.buyVolume += volumePerPrice * 0.6;
        node.sellVolume += volumePerPrice * 0.4;
      } else {
        node.sellVolume += volumePerPrice * 0.6;
        node.buyVolume += volumePerPrice * 0.4;
      }

      priceVolume.set(roundedPrice, node);
    });
  });

  const nodes = Array.from(priceVolume.values()).sort((a, b) => a.price - b.price);

  // Handle edge case: not enough data
  if (nodes.length === 0) {
    return { nodes: [], poc: 0, vah: 0, val: 0, lvns: [] };
  }

  // Find POC (highest volume node)
  const poc = nodes.reduce((max, node) => node.volume > max.volume ? node : max).price;

  // Calculate Value Area (70% of volume around POC)
  const totalVolume = nodes.reduce((sum, node) => sum + node.volume, 0);
  const targetVolume = totalVolume * 0.7;

  // Initialize VAH and VAL differently to ensure they diverge
  let vah = poc, val = poc;
  let currentVolume = nodes.find(n => n.price === poc)?.volume || 0;
  let pocIndex = nodes.findIndex(n => n.price === poc);
  let upperIndex = pocIndex + 1, lowerIndex = pocIndex - 1;

  // Expand value area around POC
  while (currentVolume < targetVolume && (upperIndex < nodes.length || lowerIndex >= 0)) {
    const upperVolume = upperIndex < nodes.length ? nodes[upperIndex].volume : 0;
    const lowerVolume = lowerIndex >= 0 ? nodes[lowerIndex].volume : 0;

    if (upperVolume >= lowerVolume && upperIndex < nodes.length) {
      currentVolume += upperVolume;
      vah = nodes[upperIndex].price;
      upperIndex++;
    } else if (lowerIndex >= 0) {
      currentVolume += lowerVolume;
      val = nodes[lowerIndex].price;
      lowerIndex--;
    } else {
      break; // No more nodes to add
    }
  }

  // Fallback: if VAH/VAL still equal POC (single node case), use price extremes
  if (vah === poc && val === poc && nodes.length > 1) {
    vah = nodes[nodes.length - 1].price; // Highest price
    val = nodes[0].price; // Lowest price
  }

  // Find Low Volume Nodes (LVNs)
  const avgVolume = totalVolume / nodes.length;
  const lvns = nodes
    .filter(node => node.volume < avgVolume * 0.5) // Less than 50% of average
    .map(node => node.price)
    .slice(0, 5); // Top 5 LVNs

  return { nodes, poc, vah, val, lvns };
}

function computeATR(series: TopstepXFuturesBar[], period: number): number | null {
  if (!series || series.length < 2) return null;
  const recent = series.slice(-Math.max(period + 1, 2));
  if (recent.length < 2) return null;

  const trs: number[] = [];
  for (let i = 1; i < recent.length; i += 1) {
    const current = recent[i];
    const prevClose = recent[i - 1].close;
    const tr = Math.max(
      current.high - current.low,
      Math.abs(current.high - prevClose),
      Math.abs(current.low - prevClose),
    );
    trs.push(tr);
  }
  if (trs.length === 0) return null;
  const sum = trs.reduce((acc, v) => acc + v, 0);
  return sum / trs.length;
}

function findNearest(levels: number[], price: number, direction: 'above' | 'below'): number | undefined {
  if (!levels || levels.length === 0) return undefined;
  const filtered = levels
    .filter(l => (direction === 'above' ? l > price : l < price))
    .sort((a, b) => direction === 'above' ? a - b : b - a);
  return filtered[0];
}

function findRecentSwing(bars: TopstepXFuturesBar[]): { swingHigh?: number; swingLow?: number } {
  if (!bars || bars.length < 3) return {};
  for (let i = bars.length - 2; i >= 1; i -= 1) {
    const prev = bars[i - 1];
    const curr = bars[i];
    const next = bars[i + 1];
    if (!next) continue;
    if (curr.high > prev.high && curr.high > next.high) {
      return { swingHigh: curr.high };
    }
    if (curr.low < prev.low && curr.low < next.low) {
      return { swingLow: curr.low };
    }
  }
  return {};
}

function buildRestingLiquidityWalls(level2: L2Level[]): Array<{ side: 'bid' | 'ask'; price: number; size: number }> {
  if (!level2 || level2.length === 0) return [];
  const bidWalls = level2
    .filter(l => l.bidSize > 0)
    .map(l => ({ side: 'bid' as const, price: l.price, size: l.bidSize }))
    .sort((a, b) => b.size - a.size)
    .slice(0, 3);
  const askWalls = level2
    .filter(l => l.askSize > 0)
    .map(l => ({ side: 'ask' as const, price: l.price, size: l.askSize }))
    .sort((a, b) => b.size - a.size)
    .slice(0, 3);
  return [...bidWalls, ...askWalls];
}

function computeRiskSnapshot(
  position: any,
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile | null,
  orderFlowData: OrderFlowData,
  marketStructure: MarketStructure,
  currentCvdBar: { cvd?: number } | null,
  level2: L2Level[]
) {
  const lastBar = bars[bars.length - 1];
  const currentPrice = lastBar?.close || position?.entryPrice || 0;
  const sessionBars = getSessionBars();
  const sessionHigh = sessionBars.length ? Math.max(...sessionBars.map(b => b.high)) : undefined;
  const sessionLow = sessionBars.length ? Math.min(...sessionBars.map(b => b.low)) : undefined;

  const signedDist = (level?: number) => (typeof level === 'number' && !Number.isNaN(level) ? Number((currentPrice - level).toFixed(2)) : undefined);

  const atr5m = computeATR(bars, 14);
  const currentRange = lastBar ? lastBar.high - lastBar.low : undefined;
  const currentRangeVsAtr = atr5m && currentRange !== undefined && atr5m > 0 ? Number((currentRange / atr5m).toFixed(2)) : undefined;
  const volRegime = currentRangeVsAtr !== undefined
    ? currentRangeVsAtr > 1.5 ? 'high' : currentRangeVsAtr < 0.7 ? 'low' : 'normal'
    : undefined;

  const lvns = volumeProfile?.lvns || [];
  const hvnCandidates = volumeProfile?.nodes
    ? [...volumeProfile.nodes].sort((a, b) => b.volume - a.volume).map(n => n.price)
    : [];

  const nearestHvnAbove = findNearest(hvnCandidates, currentPrice, 'above');
  const nearestHvnBelow = findNearest(hvnCandidates, currentPrice, 'below');
  const nearestLvnAbove = findNearest(lvns, currentPrice, 'above');
  const nearestLvnBelow = findNearest(lvns, currentPrice, 'below');

  const distToNextHvn = position
    ? position.side === 'long'
      ? signedDist(nearestHvnAbove ?? nearestHvnBelow)
      : signedDist(nearestHvnBelow ?? nearestHvnAbove)
    : undefined;
  const distToNextLvn = position
    ? position.side === 'long'
      ? signedDist(nearestLvnAbove ?? nearestLvnBelow)
      : signedDist(nearestLvnBelow ?? nearestLvnAbove)
    : undefined;

  const history = orderFlowData.cvdHistory || [];
  const deltaWindow = (ms: number) => {
    const recent = history.filter(h => Date.now() - h.timestamp <= ms);
    if (recent.length < 2) return 0;
    return recent[recent.length - 1].cvd - recent[0].cvd;
  };
  const slopeWindow = (ms: number) => {
    const recent = history.filter(h => Date.now() - h.timestamp <= ms);
    if (recent.length < 2) return 0;
    const dtMin = (recent[recent.length - 1].timestamp - recent[0].timestamp) / 60000;
    if (dtMin <= 0) return 0;
    return (recent[recent.length - 1].cvd - recent[0].cvd) / dtMin;
  };

  const deltaLast1m = deltaWindow(60_000);
  const deltaLast5m = deltaWindow(5 * 60_000);
  const cvdSlopeShort = slopeWindow(60_000);
  const cvdSlopeLong = slopeWindow(5 * 60_000);

  const absorption = orderFlowData.absorption || { buy: 0, sell: 0 };
  const exhaustion = orderFlowData.exhaustion || { buy: 0, sell: 0 };

  const volumeLevels = Object.entries(orderFlowData.volumeAtPrice || {})
    .map(([priceStr, vol]) => ({
      price: Number(priceStr),
      buy: vol.buy || 0,
      sell: vol.sell || 0,
      total: (vol.buy || 0) + (vol.sell || 0),
    }))
    .filter(l => !Number.isNaN(l.price))
    .sort((a, b) => b.total - a.total);

  const topVolumeLevel = volumeLevels[0];
  const absorptionZone = topVolumeLevel && (absorption.buy >= 0.5 || absorption.sell >= 0.5)
    ? {
        side: absorption.buy >= absorption.sell ? 'bid' : 'ask',
        price: topVolumeLevel.price,
        strength: Number((absorption.buy >= absorption.sell ? absorption.buy : absorption.sell).toFixed(2)),
      }
    : null;

  const exhaustionFlag = (exhaustion.buy >= 0.5 || exhaustion.sell >= 0.5)
    ? {
        side: exhaustion.buy >= exhaustion.sell ? 'bid' : 'ask',
        strength: Number((exhaustion.buy >= exhaustion.sell ? exhaustion.buy : exhaustion.sell).toFixed(2)),
      }
    : null;

  const largePrints = (orderFlowData.bigPrints || [])
    .slice(-10)
    .sort((a, b) => b.timestamp - a.timestamp)
    .slice(0, 5)
    .map(p => ({ side: p.side, size: p.size, price: p.price }));

  const restingLiquidityWalls = buildRestingLiquidityWalls(level2);

  const trendStrength = (() => {
    const slice = bars.slice(-20);
    if (slice.length < 2) return 0;
    const move = slice[slice.length - 1].close - slice[0].close;
    const range = Math.max(...slice.map(b => b.high)) - Math.min(...slice.map(b => b.low));
    if (range <= 0) return 0;
    return Math.min(1, Math.abs(move) / range);
  })();

  const swings = findRecentSwing(bars);
  const invalidationPrice = position
    ? position.side === 'long' ? swings.swingLow : swings.swingHigh
    : undefined;
  const lastSwingPrice = swings.swingHigh ?? swings.swingLow;

  const orderFlowPressureScore = (() => {
    const base = orderFlowData.cvd || 0;
    const deltaBoost = deltaLast1m * 0.5;
    return base + deltaBoost;
  })();
  const orderFlowPressure = orderFlowPressureScore > 50
    ? 'bullish'
    : orderFlowPressureScore < -50
      ? 'bearish'
      : 'neutral';

  return {
    currentPrice,
    recentBars: bars.slice(-10),
    cvd: orderFlowData.cvd || currentCvdBar?.cvd || 0,
    cvdTrend: orderFlowData.cvd > 0 ? 'up' : orderFlowData.cvd < 0 ? 'down' : 'neutral',
    orderFlowPressure,
    volumeProfile: volumeProfile ? {
      poc: volumeProfile.poc,
      vah: volumeProfile.vah,
      val: volumeProfile.val,
    } : undefined,
    whaleActivity: largePrints.length > 0
      ? largePrints.map(p => `${p.side === 'buy' ? 'BUY' : 'SELL'} ${p.size} @ ${p.price.toFixed(2)}`).join(', ')
      : '',
    marketStructure: marketStructure.state,
    distToPoc: signedDist(volumeProfile?.poc),
    distToVah: signedDist(volumeProfile?.vah),
    distToVal: signedDist(volumeProfile?.val),
    distToNextHvn,
    distToNextLvn,
    distToNearestHvnAbove: signedDist(nearestHvnAbove),
    distToNearestHvnBelow: signedDist(nearestHvnBelow),
    distToNearestLvnAbove: signedDist(nearestLvnAbove),
    distToNearestLvnBelow: signedDist(nearestLvnBelow),
    distToSessionHigh: signedDist(sessionHigh),
    distToSessionLow: signedDist(sessionLow),
    distToRoundNumber: signedDist(Math.round(currentPrice / 5) * 5),
    singlePrintZoneNearby: null,
    inSinglePrintZone: false,
    deltaLast1m,
    deltaLast5m,
    cvdSlopeShort,
    cvdSlopeLong,
    cvdDivergence: 'none',
    absorptionZone,
    exhaustionFlag,
    largePrints,
    restingLiquidityWalls,
    liquidityPullDetected: false,
    atr1m: undefined,
    atr5m: atr5m ?? undefined,
    volRegime,
    currentRangeVsAtr,
    structureState: marketStructure.state as any,
    invalidationPrice,
    lastSwingPrice,
    trendStrength,
    PcEstimate: undefined,
    PrEstimate: undefined,
  };
}

async function refreshHigherTimeframes(force: boolean = false) {
  const refreshIntervalMs = 15 * 60 * 1000;
  if (!force && Date.now() - lastHigherTimeframeRefresh < refreshIntervalMs) {
    return;
  }
  if (!contractId) return;

  try {
    const now = new Date();
    const fourHourStart = new Date(now.getTime() - 4 * 60 * 60 * 1000 * 30);
    const dailyStart = new Date(now.getTime() - 24 * 60 * 60 * 1000 * 35);

    const [fourHourBars, dailyBars] = await Promise.all([
      fetchTopstepXFuturesBars({
        contractId,
        startTime: fourHourStart.toISOString(),
        endTime: now.toISOString(),
        unit: 3,
        unitNumber: 4,
        limit: 80,
      }),
      fetchTopstepXFuturesBars({
        contractId,
        startTime: dailyStart.toISOString(),
        endTime: now.toISOString(),
        unit: 4,
        unitNumber: 1,
        limit: 60,
      }),
    ]);

    const filteredFourHour = fourHourBars ? fourHourBars.slice(-20) : [];
    const filteredDaily = dailyBars ? dailyBars.slice(-20) : [];

    higherTimeframeSnapshots = [];
    if (filteredFourHour.length > 0) {
      higherTimeframeSnapshots.push({
        timeframe: '240m',
        candles: filteredFourHour,
      });
    }
    if (filteredDaily.length > 0) {
      higherTimeframeSnapshots.push({
        timeframe: '1d',
        candles: filteredDaily,
      });
    }

    lastHigherTimeframeRefresh = Date.now();
    log(`📈 Refreshed higher timeframe candles (4h/1d)`, 'info');
  } catch (error: any) {
    log(`⚠️ Failed to refresh higher timeframe candles: ${error.message}`, 'warning');
  }
}

async function refreshRecentVolumeProfiles(force: boolean = false) {
  const refreshIntervalMs = 30 * 60 * 1000;
  if (!force && Date.now() - lastVolumeProfileRefresh < refreshIntervalMs) {
    return;
  }
  if (!contractId) return;

  try {
    const now = new Date();
    const lookback = new Date(now.getTime() - 6 * 24 * 60 * 60 * 1000);

    const sessionBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: lookback.toISOString(),
      endTime: now.toISOString(),
      unit: 2,
      unitNumber: 5,
      limit: 6000,
    });

    const sessions = new Map<string, TopstepXFuturesBar[]>();

    (sessionBars || []).forEach(bar => {
      const ts = new Date(bar.timestamp);
      const sessionStart = getTradingDayStart(ts);
      const key = sessionStart.toISOString();
      const bucket = sessions.get(key);
      if (bucket) {
        bucket.push(bar);
      } else {
        sessions.set(key, [bar]);
      }
    });

    const sortedSessions = Array.from(sessions.entries())
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime());

    const summaries: SessionVolumeProfileSummary[] = [];

    sortedSessions.slice(-5).forEach(([sessionStartIso, dayBars]) => {
      if (!dayBars || dayBars.length === 0) {
        return;
      }

      const profile = calculateVolumeProfile(dayBars);
      const sessionHigh = dayBars.reduce((max, bar) => Math.max(max, bar.high), Number.NEGATIVE_INFINITY);
      const sessionLow = dayBars.reduce((min, bar) => Math.min(min, bar.low), Number.POSITIVE_INFINITY);
      const sessionStart = new Date(sessionStartIso);
      const sessionEnd = new Date(sessionStart.getTime() + 24 * 60 * 60 * 1000);

      summaries.push({
        sessionStart: sessionStart.toISOString(),
        sessionEnd: sessionEnd.toISOString(),
        poc: profile.poc,
        vah: profile.vah,
        val: profile.val,
        lvns: profile.lvns,
        sessionHigh: Number.isFinite(sessionHigh) ? sessionHigh : profile.vah,
        sessionLow: Number.isFinite(sessionLow) ? sessionLow : profile.val,
      });
    });

    recentSessionProfiles = summaries;
    lastVolumeProfileRefresh = Date.now();
    log(`📊 Refreshed recent session volume profiles (${recentSessionProfiles.length})`, 'info');
  } catch (error: any) {
    log(`⚠️ Failed to refresh recent session profiles: ${error.message}`, 'warning');
  }
}

// Detect Market State
function detectMarketState(): MarketState {
  if (bars.length < 20) return 'balanced';

  const recentBars = bars.slice(-20);
  const profile = calculateVolumeProfile(recentBars);

  // Check if price is rotating around POC
  const currentPrice = bars[bars.length - 1].close;
  const priceRange = Math.max(...recentBars.map(b => b.high)) - Math.min(...recentBars.map(b => b.low));
  const valueAreaRange = profile.vah - profile.val;

  // Balanced if price mostly within value area
  const barsInValue = recentBars.filter(b => b.close >= profile.val && b.close <= profile.vah).length;
  const percentInValue = barsInValue / recentBars.length;

  if (percentInValue > 0.7) {
    // Check for failed breakouts
    const recentHigh = Math.max(...recentBars.slice(-5).map(b => b.high));
    const recentLow = Math.min(...recentBars.slice(-5).map(b => b.low));

    if (recentHigh > profile.vah && currentPrice < profile.vah) {
      return 'in_value_failed_breakout_above';
    }
    if (recentLow < profile.val && currentPrice > profile.val) {
      return 'in_value_failed_breakout_below';
    }
    return 'in_value';
  }

  // Out of balance if strong directional move
  const trend = recentBars.reduce((sum, bar, i) => {
    if (i === 0) return 0;
    return sum + (bar.close - recentBars[i - 1].close);
  }, 0);

  if (trend > priceRange * 0.3) {
    return 'above_value';
  } else if (trend < -priceRange * 0.3) {
    return 'below_value';
  }

  return 'in_value';
}

// Analyze Order Flow
function analyzeOrderFlow(): boolean {
  // Check for big prints
  const recentBigPrints = orderFlowData.bigPrints.filter(p =>
    Date.now() - p.timestamp < 60000 // Last minute
  );

  // Check CVD direction
  const cvdTrending = Math.abs(orderFlowData.cvd) > 100;

  // Check footprint imbalance
  const currentPrice = bars[bars.length - 1]?.close || 0;
  const imbalanceAtPrice = orderFlowData.footprintImbalance[Math.round(currentPrice * 4) / 4] || 0;

  // Confirmation requires multiple signals
  const bigPrintSignal = recentBigPrints.length > 2;
  const cvdSignal = cvdTrending;
  const imbalanceSignal = Math.abs(imbalanceAtPrice) > 50;

  return (bigPrintSignal && cvdSignal) || (cvdSignal && imbalanceSignal) || (bigPrintSignal && imbalanceSignal);
}

// Make Trading Decision
function makeDecision(): AgentDecision {
  const state = detectMarketState();
  const profile = volumeProfile || calculateVolumeProfile(bars.slice(-50));
  const currentPrice = bars[bars.length - 1]?.close || 0;
  const orderFlowConfirmed = analyzeOrderFlow();

  // Determine location
  let location: AgentDecision['location'] = 'neutral';
  const priceTolerance = 0.5; // Within 0.5 points

  if (Math.abs(currentPrice - profile.poc) < priceTolerance) {
    location = 'at_poc';
  } else if (Math.abs(currentPrice - profile.vah) < priceTolerance) {
    location = 'at_vah';
  } else if (Math.abs(currentPrice - profile.val) < priceTolerance) {
    location = 'at_val';
  } else if (profile.lvns.some(lvn => Math.abs(currentPrice - lvn) < priceTolerance)) {
    location = 'at_lvn';
  }

  // Determine setup model
  let model: SetupModel | null = null;
  let entrySide: 'long' | 'short' | null = null;
  let reason = '';
  let confidence = 0;

  if (state === 'above_value' || state === 'below_value') {
    model = 'trend_continuation';

    if (state === 'above_value' && location === 'at_lvn' && orderFlowConfirmed) {
      if (orderFlowData.cvd > 0 && orderFlowData.bigPrints.filter(p => p.side === 'buy').length > 0) {
        entrySide = 'long';
        reason = 'Trend continuation: Pullback to LVN above value with buy aggression';
        confidence = 80;
      }
    } else if (state === 'below_value' && location === 'at_lvn' && orderFlowConfirmed) {
      if (orderFlowData.cvd < 0 && orderFlowData.bigPrints.filter(p => p.side === 'sell').length > 0) {
        entrySide = 'short';
        reason = 'Trend continuation: Pullback to LVN below value with sell aggression';
        confidence = 80;
      }
    }
  } else if (state.includes('failed_breakout')) {
    model = 'mean_reversion';

    if (state === 'in_value_failed_breakout_above' && location === 'at_vah' && orderFlowConfirmed) {
      // Failed breakout above means buy exhaustion - buyers couldn't push through
      if (orderFlowData.exhaustion.buy > 0.6) {
        entrySide = 'short';
        reason = 'Mean reversion: Failed breakout above VAH with buy exhaustion';
        confidence = 75;
      }
    } else if (state === 'in_value_failed_breakout_below' && location === 'at_val' && orderFlowConfirmed) {
      // Failed breakout below means sell exhaustion - sellers couldn't push through
      if (orderFlowData.exhaustion.sell > 0.6) {
        entrySide = 'long';
        reason = 'Mean reversion: Failed breakout below VAL with sell exhaustion';
        confidence = 75;
      }
    }
  }

  // Risk management
  const stopDistance = 2; // 2 points
  const targetDistance = model === 'trend_continuation' ? 10 : 5; // Trend targets farther

  const riskManagement = {
    stopLoss: entrySide === 'long' ? currentPrice - stopDistance : currentPrice + stopDistance,
    target: entrySide === 'long' ? currentPrice + targetDistance : currentPrice - targetDistance,
    riskAmount: stopDistance * 20 * CONTRACTS, // $20 per point for NQ
  };

  return {
    model,
    marketState: state,
    location,
    orderFlowConfirmation: orderFlowConfirmed,
    entry: { side: entrySide, reason, confidence },
    riskManagement,
  };
}

// Process Market Data Update (with OpenAI Integration)
async function processMarketUpdate() {
  if (bars.length < 3) return;

  log(`[DEBUG processMarketUpdate] Entered function with ${bars.length} bars`, 'info');

  if (executionManager) {
    await executionManager.syncWithBrokerState();
    currentPosition = executionManager.getActivePosition();
  }

  // Update volume profile using historical session bars (10-sec bars from session start)
  // Fall back to recent bars only if no historical data available
  const profileSource = historicalSessionBars.length > 0 ? historicalSessionBars : bars.slice(-50);
  volumeProfile = calculateVolumeProfile(profileSource);

  // Update market structure (existing)
  marketStructure.state = detectMarketState();

  log(`[DEBUG processMarketUpdate] About to call refreshHigherTimeframes()`, 'info');
  await refreshHigherTimeframes();
  log(`[DEBUG processMarketUpdate] refreshHigherTimeframes() completed`, 'info');

  log(`[DEBUG processMarketUpdate] About to call refreshRecentVolumeProfiles()`, 'info');
  await refreshRecentVolumeProfiles();
  log(`[DEBUG processMarketUpdate] refreshRecentVolumeProfiles() completed`, 'info');

  // ========== CRITICAL: Sync position from ExecutionManager FIRST ==========
  // This MUST happen before any trading decisions to prevent duplicate entries
  if (executionManager) {
    const previousPosition = currentPosition;
    currentPosition = executionManager.getActivePosition();
    if (currentPosition) {
      noteContractIdFromPosition(currentPosition);
    }

    if (currentPosition && !previousPosition) {
      log(`🔄 [PositionSync] Position detected from broker: ${currentPosition.side.toUpperCase()} @ ${currentPosition.entryPrice.toFixed(2)}`, 'info');
      // Initialize risk management timer to prevent immediate spam
      lastRiskMgmtTime = Date.now();
    } else if (!currentPosition && previousPosition) {
      log(`🔄 [PositionSync] Position closed`, 'info');
      lastRiskMgmtDecision = null; // Clear Risk Management data
    }
  }

  // ========== NEW: Build market data for OpenAI ==========
  const marketData = buildFuturesMarketData(
    SYMBOL,
    bars,
    volumeProfile,
    orderFlowData,
    marketStructure,
    currentCvdBar,
    accountBalance,
    currentPosition,
    realizedPnL,
    higherTimeframeSnapshots,
    recentSessionProfiles,
    cvdMinuteBars
  );

  // ========== NEW: Get OpenAI decision on a fixed interval (every minute) ==========
  let openaiDecision = null;
  const orderFlowConfirmed = analyzeOrderFlow();  // Check order flow confirmation

  log(`[DEBUG processMarketUpdate] About to check if should analyze`, 'info');
  const nowMs = Date.now();
  const currentCandleTimestamp = bars[bars.length - 1]?.timestamp;
  const timeSinceLastAnalysis = nowMs - lastOpenAIAnalysisTime;
  const shouldAnalyze = timeSinceLastAnalysis >= ANALYSIS_INTERVAL_MS && !currentPosition;

  log(`[DEBUG processMarketUpdate] Time since last analysis: ${timeSinceLastAnalysis}ms, threshold: ${ANALYSIS_INTERVAL_MS}ms, hasPosition: ${!!currentPosition}, shouldAnalyze: ${shouldAnalyze}`, 'info');

  // ONLY analyze for new trades when we have NO position
  // When we have a position, the Risk Management Agent handles everything
  if (shouldAnalyze) {
    lastOpenAIAnalysisTime = nowMs;

    try {
      log(`🧠 [OpenAI] Analyzing market for NEW TRADE opportunities (no active position)`, 'info');
      openaiDecision = await analyzeFuturesMarket(marketData);

      if (openaiDecision) {
        // Confidence gating: require flow/L2 signals for high confidence
        const hasFlowSignal =
          orderFlowConfirmed ||
          (orderFlowData.exhaustion && (orderFlowData.exhaustion.buy > 0.6 || orderFlowData.exhaustion.sell > 0.6)) ||
          (orderFlowData.absorption && (orderFlowData.absorption.buy > 0.6 || orderFlowData.absorption.sell > 0.6)) ||
          (orderFlowData.bigPrints && orderFlowData.bigPrints.length > 0) ||
          (l2Data && l2Data.some(l => (l.bidSize || 0) > 0 || (l.askSize || 0) > 0));

        let clampedConfidence = Math.max(0, Math.min(100, openaiDecision.confidence || 0));
        // Cap at 60% if no flow signal; allow >60 only when flow/L2 signals present
        if (!hasFlowSignal && clampedConfidence > 60) {
          clampedConfidence = 60;
        }
        const normalizedConfidence = clampedConfidence / 100;

        log(`🤖 [OpenAI] ${openaiDecision.decision} @ ${openaiDecision.entryPrice?.toFixed(2) || 'null'} | Confidence: ${clampedConfidence}% | Regime: ${openaiDecision.inferredRegime || 'unknown'}`, 'success');

        // Emit OpenAI decision to dashboard
        // Let the AI agent decide based on all order flow data it received
        // No secondary gating - trust the reasoner's analysis
        if (io) {
          const decisionPayload = {
            timestamp: new Date().toISOString(),
            decision: openaiDecision.decision,
            reasoning: openaiDecision.reasoning,
            confidence: normalizedConfidence,
            entryPrice: openaiDecision.entryPrice,
            stopLoss: openaiDecision.stopLoss,
            target: openaiDecision.target,
            riskManagementReasoning: openaiDecision.riskManagementReasoning || null, // SL/TP placement explanation
            inferredRegime: openaiDecision.inferredRegime,
            trade_decisions: openaiDecision.decision !== 'HOLD' ? [openaiDecision.decision] : [],
          };
          io.emit('llm_decision', decisionPayload);
          log(`📤 [Dashboard] Emitted llm_decision: ${openaiDecision.decision} @ ${openaiDecision.entryPrice} (confidence: ${(normalizedConfidence*100).toFixed(0)}%)`, 'info');
          log(`   Reasoning preview: ${(openaiDecision.reasoning || '').substring(0, 100)}...`, 'info');
        }
      }
    } catch (error: any) {
      log(`❌ [OpenAI] Analysis failed: ${error.message}`, 'error');
      // Continue with rule-based only if OpenAI fails
    }
  }

  // OLD RULE-BASED SYSTEM - DISABLED (DeepSeek Reasoner only)
  // const ruleBasedDecision = makeDecision();

  // ========== NEW: Execute if high confidence ==========
  if (executionManager && openaiDecision) {
    const executionResult = await processOpenAIDecision(
      openaiDecision,
      executionManager,
      bars[bars.length - 1].close,
      SYMBOL,
      orderFlowData,
      volumeProfile,
      marketStructure
    );

    if (executionResult.executed) {
      currentPosition = executionManager.getActivePosition();

      // ========== IMMEDIATE RISK MANAGEMENT AFTER ENTRY ==========
      // Call risk manager immediately after entering position (don't wait for interval)
      if (currentPosition && bars && bars.length > 0) {
        log('🛡️ [RiskMgmt] 🚀 NEW POSITION - Running IMMEDIATE Risk Management Analysis...', 'success');

        // CRITICAL: Set timer IMMEDIATELY to prevent regular 30s check from also firing
        lastRiskMgmtTime = nowMs;

        try {
          // Hard-sync protective orders before analysis to ensure correct IDs/legs
          await executionManager.syncProtectiveOrdersFromOpenOrders(currentPosition, !currentPosition.usesNativeBracket);

          currentPosition = executionManager.getActivePosition();
          if (!currentPosition) {
            log('🛡️ [RiskMgmt] Skipping immediate analysis: position not available after sync.', 'warn');
            // Skip immediate risk; wait for next cycle
            return;
          }

          const riskDecision = await analyzePositionRisk(
            currentPosition,
            buildRiskSnapshot(
              currentPosition,
              bars,
              volumeProfile,
              orderFlowData,
              marketStructure,
              currentCvdBar,
              l2Data,
            ),
            0.1 // Gold tick size
          );

          log(`🛡️ [RiskMgmt] Initial Decision: ${riskDecision.action} (${riskDecision.urgency} urgency)`, 'info');
          log(`🛡️ [RiskMgmt] ${riskDecision.reasoning.substring(0, 200)}...`, 'info');

          // Store for dashboard
          lastRiskMgmtDecision = {
            timestamp: new Date().toISOString(),
            action: riskDecision.action,
            urgency: riskDecision.urgency,
            reasoning: riskDecision.reasoning,
            newStopLoss: riskDecision.newStopLoss,
            newTarget: riskDecision.newTarget,
          };

          // Emit to dashboard
          if (io) {
            io.emit('llm_decision', {
              timestamp: new Date().toISOString(),
              decision: riskDecision.action === 'CLOSE_POSITION' ? (currentPosition.side === 'long' ? 'SELL' : 'BUY') : 'HOLD',
              reasoning: `🛡️ INITIAL RISK ASSESSMENT: ${riskDecision.reasoning}`,
              confidence: riskDecision.urgency === 'high' ? 0.9 : riskDecision.urgency === 'medium' ? 0.7 : 0.5,
              entryPrice: null,
              stopLoss: riskDecision.newStopLoss,
              target: riskDecision.newTarget,
              riskManagementReasoning: `Action: ${riskDecision.action} | Risk Level: ${riskDecision.riskLevel} | Urgency: ${riskDecision.urgency}`,
              inferredRegime: 'INITIAL_POSITION_SETUP',
              trade_decisions: [],
            });
          }

          // Apply risk management decision
          if (riskDecision.action === 'CLOSE_POSITION') {
            log('🛡️ [RiskMgmt] ⚠️ IMMEDIATE CLOSE recommended!', 'warning');
            const closePrice = bars[bars.length - 1].close;
            const closedDecisionId = await executionManager.closePosition(
              currentPosition.decisionId,
              closePrice,
              'immediate_risk_management_close'
            );
            if (closedDecisionId) {
              log(`🛡️ [RiskMgmt] ✅ Position closed immediately`, 'success');
              currentPosition = null;
              lastRiskMgmtTime = 0;
              lastRiskMgmtDecision = null;
            }
          } else if (riskDecision.action !== 'HOLD_BRACKETS') {
            // Adjust brackets immediately
            const adjusted = await executionManager.adjustActiveProtection(
              riskDecision.newStopLoss,
              riskDecision.newTarget,
              riskDecision.positionVersion
            );
            if (adjusted) {
              log(`🛡️ [RiskMgmt] ✅ Initial brackets set - Stop: ${riskDecision.newStopLoss?.toFixed(2) || 'unchanged'}, Target: ${riskDecision.newTarget?.toFixed(2) || 'unchanged'}`, 'success');
              currentPosition = executionManager.getActivePosition();
            }
          }

          // Timer already set at the start to prevent double-calling

        } catch (error: any) {
          log(`🛡️ [RiskMgmt] ❌ Error in immediate risk assessment: ${error.message}`, 'error');
          // Timer already set at the start
        }
      }
    }
  }
  // OLD RULE-BASED FALLBACK - DISABLED (DeepSeek Reasoner only)
  // else if (ruleBasedDecision.entry.side && ruleBasedDecision.entry.confidence >= 70 && !currentPosition) {
  //   log(`📊 Rule-based Decision: ${ruleBasedDecision.entry.side.toUpperCase()} (Confidence: ${ruleBasedDecision.entry.confidence}%)`, 'info');
  //   executeEntry(ruleBasedDecision);
  // }

  // ========== NEW: Update position and check exits ==========
  if (executionManager) {
    // Hard-sync with broker: if Topstep reports FLAT (no position + no protectives),
    // clear any stale local position so the risk manager doesn't keep firing when we're flat.
    const brokerFlat = await executionManager.clearIfBrokerFlat();
    if (brokerFlat) {
      currentPosition = null;
      lastRiskMgmtTime = 0;
      lastRiskMgmtDecision = null;
      log('🛡️ [RiskMgmt] Broker confirmed FLAT - clearing local position and skipping risk management.', 'info');
    }

    if (currentPosition) {
      currentPosition = executionManager.getActivePosition();
      if (!currentPosition) {
        log('🛡️ [RiskMgmt] Skipping management: no active position found.', 'warn');
      }

      const exitResult = await updatePositionAndCheckExits(
        executionManager,
        bars[bars.length - 1].close,
        bars
      );

      if (exitResult.exited) {
        // Update realized P&L
        const { tradingDB } = await import('./lib/tradingDatabase');
        const outcome = tradingDB.getOutcome(exitResult.closedDecisionId);
        if (outcome) {
          realizedPnL += outcome.profitLoss;
        }
        currentPosition = null;
        lastRiskMgmtTime = 0; // Reset risk management timer
        lastRiskMgmtDecision = null; // Clear old Risk Management decision from dashboard
      } else {
      // ========== RISK MANAGEMENT AGENT ==========
      // Run risk management analysis every 30 seconds for active drawdown prevention
      // This REPLACES the main trading agent when we have a position
      const timeSinceLastRiskMgmt = nowMs - lastRiskMgmtTime;
      if (timeSinceLastRiskMgmt >= RISK_MGMT_INTERVAL_MS) {
        try {
          log(`🛡️ [RiskMgmt] 🎯 ACTIVE POSITION - Running Risk Management Agent... (${Math.floor(timeSinceLastRiskMgmt / 1000)}s since last check)`, 'info');

          // CRITICAL: Update timer IMMEDIATELY to prevent duplicate calls during async operation
          lastRiskMgmtTime = Date.now();

          // Safety check: ensure we have bars before analyzing
          if (bars && bars.length > 0 && currentPosition) {
            // Hard-sync protective orders before analysis to ensure correct IDs/legs
            await executionManager.syncProtectiveOrdersFromOpenOrders(currentPosition, !currentPosition.usesNativeBracket);

            currentPosition = executionManager.getActivePosition();
            if (!currentPosition) {
              log('🛡️ [RiskMgmt] Skipping analysis: position not available after sync.', 'warn');
              // Skip this risk cycle; wait for next interval
              return;
            }

            const riskDecision = await analyzePositionRisk(
              currentPosition,
              buildRiskSnapshot(
                currentPosition,
                bars,
                volumeProfile,
                orderFlowData,
                marketStructure,
                currentCvdBar,
                l2Data,
              )
            );

          log(`🛡️ [RiskMgmt] Decision: ${riskDecision.action} (${riskDecision.urgency} urgency)`, 'info');
          log(`🛡️ [RiskMgmt] ${riskDecision.reasoning.substring(0, 200)}...`, 'info');

          // Store for dashboard broadcast
          lastRiskMgmtDecision = {
            timestamp: new Date().toISOString(),
            action: riskDecision.action,
            urgency: riskDecision.urgency,
            reasoning: riskDecision.reasoning,
            newStopLoss: riskDecision.newStopLoss,
            newTarget: riskDecision.newTarget,
          };

          // Emit Risk Management decision to dashboard
          if (io) {
            io.emit('llm_decision', {
              timestamp: new Date().toISOString(),
              decision: riskDecision.action === 'CLOSE_POSITION' ? (currentPosition.side === 'long' ? 'SELL' : 'BUY') : 'HOLD',
              reasoning: `🛡️ RISK MANAGEMENT: ${riskDecision.reasoning}`,
              confidence: riskDecision.urgency === 'high' ? 0.9 : riskDecision.urgency === 'medium' ? 0.7 : 0.5,
              entryPrice: null,
              stopLoss: riskDecision.newStopLoss,
              target: riskDecision.newTarget,
              riskManagementReasoning: `Action: ${riskDecision.action} | Risk Level: ${riskDecision.riskLevel} | Urgency: ${riskDecision.urgency}`,
              inferredRegime: 'POSITION_MANAGEMENT',
              trade_decisions: [],
            });
          }

          // Apply risk management decision
          if (riskDecision.action === 'CLOSE_POSITION') {
            log('🛡️ [RiskMgmt] ⚠️ CLOSING POSITION based on risk analysis!', 'warning');
            // Close position immediately by calling ExecutionManager directly
            const closePrice = bars[bars.length - 1].close;
            const closedDecisionId = await executionManager.closePosition(
              currentPosition.decisionId,
              closePrice,
              'risk_management_close'
            );

            if (closedDecisionId) {
              log(`🛡️ [RiskMgmt] ✅ Position closed successfully (Decision ID: ${closedDecisionId})`, 'success');
              currentPosition = null;
              lastRiskMgmtTime = 0;
              lastRiskMgmtDecision = null; // Clear old Risk Management decision
            } else {
              log('🛡️ [RiskMgmt] ❌ Failed to close position', 'error');
            }
          } else if (riskDecision.action !== 'HOLD_BRACKETS') {
            // Adjust brackets
            const adjusted = await executionManager.adjustActiveProtection(
              riskDecision.newStopLoss,
              riskDecision.newTarget,
              riskDecision.positionVersion
            );

            if (adjusted) {
              log(`🛡️ [RiskMgmt] ✅ Brackets adjusted - Stop: ${riskDecision.newStopLoss?.toFixed(2) || 'unchanged'}, Target: ${riskDecision.newTarget?.toFixed(2) || 'unchanged'}`, 'success');
              currentPosition = executionManager.getActivePosition(); // Refresh position
            } else {
              log('🛡️ [RiskMgmt] ⚠️ Failed to adjust brackets', 'warning');
            }
          } else {
            log('🛡️ [RiskMgmt] ✓ Holding current brackets - no adjustments needed', 'info');
          }
          } else {
            log('🛡️ [RiskMgmt] ⚠️ No bars available yet, skipping risk analysis', 'warn');
          }

        } catch (error: any) {
          log(`🛡️ [RiskMgmt] ❌ Error in risk management: ${error.message}`, 'error');
          // Timer already set at start of try block, don't reset
        }
      } else if (currentPosition && timeSinceLastRiskMgmt > 0) {
        // Log why we're not running (for debugging silent periods)
        const secondsRemaining = Math.ceil((RISK_MGMT_INTERVAL_MS - timeSinceLastRiskMgmt) / 1000);
        log(`🛡️ [RiskMgmt] ⏳ Next check in ${secondsRemaining}s (${Math.floor(timeSinceLastRiskMgmt / 1000)}s/${RISK_MGMT_INTERVAL_MS / 1000}s elapsed)`, 'debug');
      }
    }
  }

  // Emit decision (existing, can be enhanced)
  if (openaiDecision) {
    const clampedConfidence = Math.max(0, Math.min(100, openaiDecision.confidence || 0));
    emitLLMDecision({
      marketState: openaiDecision.marketState,
      model: openaiDecision.setupModel,
      location: openaiDecision.location,
      orderFlow: {
        cvd: orderFlowData.cvd,
        bigPrints: orderFlowData.bigPrints.length,
        confirmed: true,
      },
      decision: openaiDecision.decision.toLowerCase(),
      reasoning: openaiDecision.reasoning,
      confidence: clampedConfidence,
    });
  }
  // OLD RULE-BASED SIGNAL EMISSION - DISABLED (DeepSeek Reasoner only)
  // else if (ruleBasedDecision.entry.confidence > 60) {
  //   emitLLMDecision({
  //     marketState: ruleBasedDecision.marketState,
  //     model: ruleBasedDecision.model,
  //     location: ruleBasedDecision.location,
  //     orderFlow: {
  //       cvd: orderFlowData.cvd,
  //       bigPrints: orderFlowData.bigPrints.length,
  //       confirmed: ruleBasedDecision.orderFlowConfirmation,
  //     },
  //     decision: ruleBasedDecision.entry.side || 'hold',
  //     reasoning: ruleBasedDecision.entry.reason,
  //     confidence: ruleBasedDecision.entry.confidence,
  //   });
  // }

  // Update dashboard (existing)
  broadcastDashboardUpdate();
}
}

// Execute Trade Entry
async function executeEntry(decision: AgentDecision) {
  if (!decision.entry.side) return;

  log(`📈 Executing ${decision.entry.side.toUpperCase()} entry: ${decision.entry.reason}`, 'success');

  // Place market order (simplified for now)
  currentPosition = {
    side: decision.entry.side,
    entryPrice: bars[bars.length - 1].close,
    stopLoss: decision.riskManagement.stopLoss,
    target: decision.riskManagement.target,
    contracts: CONTRACTS,
    entryTime: new Date().toISOString(),
  };

  log(`Position opened: ${decision.entry.side} @ ${currentPosition.entryPrice}`, 'success');
}

// Broadcast Dashboard Update
function broadcastDashboardUpdate() {
  if (!io) return;

  const now = Date.now();
  const currentBar = bars[bars.length - 1];

  // Emit bar event
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

  // Emit tick event
  if (currentBar) {
    io.emit('tick', { price: currentBar.close });
  }

  // Emit status event
  io.emit('status', {
    balance: accountBalance,
    position: currentPosition ? {
      side: currentPosition.side,
      entry_price: currentPosition.entryPrice,
      contracts: currentPosition.contracts,
      pnl: currentPosition.side === 'long'
        ? ((currentBar?.close || 0) - currentPosition.entryPrice) * 20 * currentPosition.contracts
        : (currentPosition.entryPrice - (currentBar?.close || 0)) * 20 * currentPosition.contracts,
      stop_loss: currentPosition.stopLoss,
      target: currentPosition.target,
    } : null,
    risk_management: lastRiskMgmtDecision,
    trades_today: 0, // TODO: track this
    daily_pnl: 0, // TODO: track this
  });

  // Emit volume profile event
  if (volumeProfile) {
    io.emit('volume_profile', {
      poc: volumeProfile.poc,
      vah: volumeProfile.vah,
      val: volumeProfile.val,
      lvns: volumeProfile.lvns,
      session_high: Math.max(...bars.slice(-100).map(b => b.high)),
      session_low: Math.min(...bars.slice(-100).map(b => b.low)),
    });
  }

  // Emit CVD and order flow event
  const cvdData = {
    cvd_value: orderFlowData.cvd,
    cvd_trend: orderFlowData.cvd > 0 ? 'up' : orderFlowData.cvd < 0 ? 'down' : 'neutral',
    // CVD OHLC candlestick data
    cvd_ohlc: currentCvdBar ? {
      timestamp: currentCvdBar.timestamp,
      open: currentCvdBar.open,
      high: currentCvdBar.high,
      low: currentCvdBar.low,
      close: currentCvdBar.close,
    } : null,
    buy_absorption: orderFlowData.absorption.buy,
    sell_absorption: orderFlowData.absorption.sell,
    buy_exhaustion: orderFlowData.exhaustion.buy,
    sell_exhaustion: orderFlowData.exhaustion.sell,
    cvd_candles: [...cvdMinuteBars, ...(currentCvdBar ? [currentCvdBar] : [])],
    big_prints: orderFlowData.bigPrints.slice(-10),
  };
  log(`📡 Broadcasting CVD: ${orderFlowData.cvd.toFixed(2)}, trend=${cvdData.cvd_trend}, OHLC=[O:${currentCvdBar?.open.toFixed(1)}, H:${currentCvdBar?.high.toFixed(1)}, L:${currentCvdBar?.low.toFixed(1)}, C:${currentCvdBar?.close.toFixed(1)}], total candles=${cvdMinuteBars.length}`, 'info');
  io.emit('cvd', cvdData);

  // Emit L2 data event
  if (l2Data.length > 0) {
    const bids = l2Data.filter(l => l.bidSize > 0).slice(0, 5).map(l => [l.price, l.bidSize]);
    const asks = l2Data.filter(l => l.askSize > 0).slice(0, 5).map(l => [l.price, l.askSize]);
    io.emit('l2_data', {
      bids,
      asks,
      spread: asks.length > 0 && bids.length > 0 ? asks[0][0] - bids[0][0] : 0,
    });
  }

  // Emit market state event
  io.emit('market_state', {
    state: marketStructure.state,
    range_condition: 'normal', // TODO: calculate
    location_vs_value: volumeProfile && currentBar ?
      currentBar.close > volumeProfile.vah ? 'above' :
      currentBar.close < volumeProfile.val ? 'below' : 'inside' : 'unknown',
    location_vs_poc: volumeProfile && currentBar ?
      currentBar.close > volumeProfile.poc ? 'above' : 'below' : 'unknown',
    buyers_control: orderFlowData.cvd > 0 ? 0.7 : 0.3,
    sellers_control: orderFlowData.cvd < 0 ? 0.7 : 0.3,
    poc_crosses: 0, // TODO: track
    time_in_value: 0, // TODO: track
  });
}

// Initialize Dashboard
async function initDashboard() {
  app = express();
  app.use(cors());

  server = http.createServer(app);
  io = new Server(server, {
    cors: { origin: '*', methods: ['GET', 'POST'] }
  });

  // Serve dashboard HTML
  app.get('/', (req, res) => {
    res.sendFile('/Users/coreycosta/trading-agent/public/fabio-agent-dashboard-mgc.html');
  });

  // Socket.IO connection
  io.on('connection', (socket) => {
    log('Dashboard client connected', 'info');
    broadcastDashboardUpdate();

    socket.on('request_chart_history', () => {
      socket.emit('chart_history', getSessionBars());
    });

    socket.on('start_trading', () => {
      log('Trading enabled via dashboard', 'success');
    });

    socket.on('stop_trading', () => {
      log('Trading disabled via dashboard', 'warning');
    });
  });

  server.listen(DASHBOARD_PORT, () => {
    log(`Dashboard running on http://localhost:${DASHBOARD_PORT}`, 'success');
  });
}

// Connect to Market Data
async function connectMarketData() {
  const token = await authenticate();

  // Create market hub
  marketHub = new HubConnectionBuilder()
    .withUrl(`https://rtc.topstepx.com/hubs/market?access_token=${token}`, {
      transport: HttpTransportType.WebSockets,
      skipNegotiation: true,
    })
    .withAutomaticReconnect()
    .configureLogging(LogLevel.Warning)
    .build();

  // Subscribe to market events
  marketHub.on('gatewayquote', (id: string, quotes: any) => {
    const allowId = resolvedContractId || contractId;
    if (!id || !allowId || id !== allowId) return;
    // Process Level 2 data
    if (quotes.bids && quotes.asks) {
      l2Data = [];
      const depthLevels = Math.min(10, Math.max(quotes.bids.length, quotes.asks.length));
      const book: Record<number, { price: number; bidSize: number; askSize: number }> = {};
      for (let i = 0; i < depthLevels; i++) {
        if (quotes.bids[i]) {
          const price = quotes.bids[i].price;
          book[price] = book[price] || { price, bidSize: 0, askSize: 0 };
          book[price].bidSize += quotes.bids[i].size;
        }
        if (quotes.asks[i]) {
          const price = quotes.asks[i].price;
          book[price] = book[price] || { price, bidSize: 0, askSize: 0 };
          book[price].askSize += quotes.asks[i].size;
        }
      }
      l2Data = Object.values(book).sort((a, b) => a.price - b.price);
    }
    processMarketUpdate();
  });

  const handleDepth = (id: string, depth: any) => {
    const allowId = resolvedContractId || contractId;
    if (id && allowId && id !== allowId) return;
    const bids = depth?.bids || depth?.Bids;
    const asks = depth?.asks || depth?.Asks;
    if (!Array.isArray(bids) || !Array.isArray(asks)) return;

    l2Data = [];
    const depthLevels = Math.min(10, Math.max(bids.length, asks.length));
    const book: Record<number, { price: number; bidSize: number; askSize: number }> = {};
    for (let i = 0; i < depthLevels; i++) {
      if (bids[i]) {
        const price = bids[i].price;
        book[price] = book[price] || { price, bidSize: 0, askSize: 0 };
        book[price].bidSize += bids[i].size;
      }
      if (asks[i]) {
        const price = asks[i].price;
        book[price] = book[price] || { price, bidSize: 0, askSize: 0 };
        book[price].askSize += asks[i].size;
      }
    }
    l2Data = Object.values(book).sort((a, b) => a.price - b.price);
    processMarketUpdate();
  };

  marketHub.on('gatewaydepth', handleDepth);
  marketHub.on('GatewayMarketDepth', handleDepth);
  marketHub.on('gatewayDepth', handleDepth);
  marketHub.on('GatewayDepth', handleDepth);
  marketHub.on('gatewaylogout', (msg: any) => {
    log(`[MarketHub] gatewaylogout event: ${JSON.stringify(msg)}`, 'warn');
  });
  marketHub.on('GatewayLogout', (msg: any) => {
    log(`[MarketHub] GatewayLogout event: ${JSON.stringify(msg)}`, 'warn');
  });

  // Execution / fills stream (if provided by TopstepX feed)
  // Note: Use execution events as fast invalidators for position version
  marketHub.on('gatewayexecution', async (id: string, exec: any) => {
    const allowId = resolvedContractId || contractId;
    if (id !== allowId) return;
    const status = (exec?.status || '').toString().toLowerCase();
    const filledQty = Number(exec?.filledSize ?? exec?.fillSize ?? exec?.lastQty ?? exec?.lastFillQty ?? 0);
    const isFill = status.includes('fill') || filledQty > 0;
    if (!isFill) return;
    if (executionManager) {
      executionManager.bumpVersionExternal();
    }
  });

  marketHub.on('gatewaytrade', (id: string, trades: any[]) => {
    const allowId = resolvedContractId || contractId;
    if (!id || !allowId || id !== allowId) return;
    log(`📊 Received ${trades.length} trade(s) for ${id}`, 'info');
    trades.forEach((trade, idx) => {
      const price = parseFloat(trade.price);
      const size = parseFloat(trade.volume); // TopStepX uses "volume" not "size"
      const type = trade.type; // 0 or 1
      const side = type === 0 ? 'Buy' : 'Sell'; // type: 0=Buy, 1=Sell (TopStepX inverted)

      // Debug first trade to see format
      if (idx === 0) {
        log(`🔍 Trade: price=${price}, volume=${size}, type=${type} (${side})`, 'info');
      }

      // Skip invalid trades
      if (isNaN(price) || isNaN(size)) {
        log(`⚠️  Skipping invalid trade: price=${price}, volume=${size}, type=${type}`, 'warn');
        return;
      }

      const now = new Date();
      const timestamp = now.toISOString();

      // Build 5-minute bars from trades
      const currentMinute = Math.floor(now.getTime() / (60 * 1000)) * (60 * 1000);
      const currentMinuteISO = new Date(currentMinute).toISOString();
      let currentBar = bars[bars.length - 1];

      if (!currentBar || new Date(currentBar.timestamp).getTime() < currentMinute) {
        // New bar - use 1-minute boundary timestamp
        const newBar: TopstepXFuturesBar = {
          timestamp: currentMinuteISO,
          open: price,
          high: price,
          low: price,
          close: price,
          volume: size,
        };
        bars.push(newBar);
        log(`📊 NEW 1-MIN CANDLE: ${currentMinuteISO}, O=${price}, bars=${bars.length}`, 'info');
        // Keep last 1500 bars (~1 session)
        if (bars.length > 1500) bars.shift();
      } else {
        // Update existing bar
        currentBar.high = Math.max(currentBar.high, price);
        currentBar.low = Math.min(currentBar.low, price);
        currentBar.close = price;
        currentBar.volume = (currentBar.volume || 0) + size;
      }

      // Update CVD
      const delta = side === 'Buy' ? size : -size;
      const previousCvd = orderFlowData.cvd;
      orderFlowData.cvd += delta;

      // Track CVD OHLC for current 1-minute bar
      if (!currentCvdBar || currentCvdBar.timestamp !== currentMinuteISO) {
        if (currentCvdBar) {
          cvdMinuteBars.push(currentCvdBar);
          if (cvdMinuteBars.length > 1500) {
            cvdMinuteBars.shift();
          }
        }

        currentCvdBar = {
          timestamp: currentMinuteISO,
          open: orderFlowData.cvd,
          high: orderFlowData.cvd,
          low: orderFlowData.cvd,
          close: orderFlowData.cvd,
        };
        log(`📊 NEW CVD BAR: ${currentMinuteISO}, CVD=${orderFlowData.cvd.toFixed(2)}`, 'info');
      } else {
        // Update existing CVD bar
        currentCvdBar.high = Math.max(currentCvdBar.high, orderFlowData.cvd);
        currentCvdBar.low = Math.min(currentCvdBar.low, orderFlowData.cvd);
        currentCvdBar.close = orderFlowData.cvd;
      }

      // Track CVD history for exhaustion detection
      orderFlowData.cvdHistory.push({
        timestamp: Date.now(),
        cvd: orderFlowData.cvd,
        delta,
      });
      // Keep last 100 entries
      if (orderFlowData.cvdHistory.length > 100) {
        orderFlowData.cvdHistory.shift();
      }

      // Track volume at price for absorption detection
      const priceLevel = Math.round(price * 4) / 4;
      if (!orderFlowData.volumeAtPrice[priceLevel]) {
        orderFlowData.volumeAtPrice[priceLevel] = {
          buy: 0,
          sell: 0,
          timestamp: Date.now(),
        };
      }
      if (side === 'Buy') {
        orderFlowData.volumeAtPrice[priceLevel].buy += size;
      } else {
        orderFlowData.volumeAtPrice[priceLevel].sell += size;
      }
      orderFlowData.volumeAtPrice[priceLevel].timestamp = Date.now();
      const retentionWindow = 10 * 60 * 1000;
      Object.keys(orderFlowData.volumeAtPrice).forEach(level => {
        if (Date.now() - orderFlowData.volumeAtPrice[Number(level)].timestamp > retentionWindow) {
          delete orderFlowData.volumeAtPrice[Number(level)];
        }
      });

      // Check for big prints
      if (size >= 10) { // 10+ contracts is big
        orderFlowData.bigPrints.push({
          price,
          size,
          side: side === 'Buy' ? 'buy' : 'sell',
          timestamp: Date.now(),
        });

        // Keep only recent big prints
        orderFlowData.bigPrints = orderFlowData.bigPrints.filter(p =>
          Date.now() - p.timestamp < 300000 // 5 minutes
        );
      }

      // Update footprint imbalance
      orderFlowData.footprintImbalance[priceLevel] =
        (orderFlowData.footprintImbalance[priceLevel] || 0) + delta;

      // Calculate real absorption and exhaustion
      orderFlowData.absorption = calculateAbsorption();
      orderFlowData.exhaustion = calculateExhaustion();
    });

    log(`📈 Processed ${trades.length} trade(s), CVD: ${orderFlowData.cvd.toFixed(2)}, bars: ${bars.length}`, 'info');
    processMarketUpdate();
    broadcastDashboardUpdate();
  });

  await marketHub.start();
  log('Connected to market data hub', 'success');

  // Subscribe to contract
  let metadata: any = null;
  try {
    metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
    contractId = metadata?.id || '';
  } catch (err: any) {
    log(`[Init] ⚠️ Contract resolve failed for ${SYMBOL}: ${err?.message || err}. Will wait for contractId from position feed.`, 'warn');
    contractId = '';
  }

  // Initialize execution manager with live trading capability
  const LIVE_TRADING = process.env.LIVE_TRADING === 'true';  // Set to 'true' to enable real order submission

  // CRITICAL: Override tick sizes for known contracts (TopStepX API sometimes returns wrong values)
  const KNOWN_TICK_SIZES: Record<string, number> = {
    'GCZ5': 0.10,    // Gold futures
    'GCG6': 0.10,    // Gold futures
    'MGC': 0.10,     // Micro Gold
    'MGCZ5': 0.10,   // Micro Gold
    'NQZ5': 0.25,    // E-mini Nasdaq
    'NQH6': 0.25,    // E-mini Nasdaq
    'ESZ5': 0.25,    // E-mini S&P 500
    'ESH6': 0.25,    // E-mini S&P 500
    'M6E': 0.00001,  // Micro Euro
    'MES': 0.25,     // Micro E-mini S&P 500
    'MNQ': 0.25,     // Micro E-mini Nasdaq
  };

  const correctTickSize = KNOWN_TICK_SIZES[SYMBOL] || metadata.tickSize || 0.25;
  if (KNOWN_TICK_SIZES[SYMBOL] && metadata.tickSize !== correctTickSize) {
    log(`⚠️ TopStepX API returned wrong tick size for ${SYMBOL}: ${metadata.tickSize} (corrected to ${correctTickSize})`, 'warning');
  }

  executionManager = createExecutionManager(SYMBOL, contractId, CONTRACTS, LIVE_TRADING, {
    tickSize: correctTickSize,
    preferredAccountId: ACCOUNT_ID > 0 ? ACCOUNT_ID : undefined,
    enableNativeBrackets: process.env.TOPSTEPX_ENABLE_NATIVE_BRACKETS === 'true',
    requireNativeBrackets: process.env.TOPSTEPX_REQUIRE_NATIVE_BRACKETS !== 'false',
  });
  log(`⚙️ Execution manager initialized for ${SYMBOL} (${LIVE_TRADING ? 'LIVE TRADING' : 'SIM MODE'})`, LIVE_TRADING ? 'warning' : 'success');

  // Initialize trading account (selects account with balance < $40k)
  const accountInitialized = await executionManager.initializeTradingAccount();
  if (!accountInitialized) {
    log('⚠️ Failed to initialize trading account. Orders will not be submitted.', 'error');
  }

  // Initialize websocket position feed for real-time sync
  if (accountInitialized) {
    await executionManager.initializeAccountFeed();
  }

  if (accountInitialized) {
    const rehydrated = await executionManager.rehydrateActivePosition();
    if (rehydrated) {
      currentPosition = rehydrated;
      noteContractIdFromPosition(rehydrated);
      log(
        `♻️ Rehydrated existing ${rehydrated.side.toUpperCase()} position (${rehydrated.contracts} contracts) @ ${rehydrated.entryPrice.toFixed(2)} | SL ${rehydrated.stopLoss.toFixed(2)} | TP ${rehydrated.target.toFixed(2)}`,
        'warning'
      );
    }
  }

  // Subscribe to market data
  marketHub.invoke('SubscribeContractQuotes', contractId).catch(err =>
    log(`Failed to subscribe to quotes: ${err}`, 'error')
  );
  marketHub.invoke('SubscribeContractTrades', contractId).catch(err =>
    log(`Failed to subscribe to trades: ${err}`, 'error')
  );
  marketHub.invoke('SubscribeContractMarketDepth', contractId).catch(err =>
    log(`Failed to subscribe to market depth: ${err}`, 'error')
  );

  log(`Subscribed to ${SYMBOL} market data (contractId=${contractId})`, 'success');
}

// Get futures trading day start time (6pm ET yesterday)
function getTradingDayStart(referenceDate: Date = new Date()): Date {
  const now = new Date(referenceDate);

  // Convert to ET (UTC-5 or UTC-4 during DST)
  const etOffset = -5; // Standard time, adjust for DST if needed
  const etNow = new Date(now.getTime() + (etOffset * 60 * 60 * 1000));

  // Get current hour in ET
  const etHour = etNow.getUTCHours();

  // If before 5pm ET (17:00), trading day started 6pm ET yesterday
  // If after 5pm ET, trading day starts at 6pm ET today
  let tradingDayStart: Date;
  if (etHour < 17) {
    // Before 5pm ET - use 6pm yesterday
    tradingDayStart = new Date(etNow);
    tradingDayStart.setUTCDate(tradingDayStart.getUTCDate() - 1);
    tradingDayStart.setUTCHours(18, 0, 0, 0); // 6pm ET = 18:00 ET
  } else {
    // After 5pm ET - use 6pm today
    tradingDayStart = new Date(etNow);
    tradingDayStart.setUTCHours(18, 0, 0, 0);
  }

  // Convert back to UTC
  return new Date(tradingDayStart.getTime() - (etOffset * 60 * 60 * 1000));
}

function getSessionBars(): TopstepXFuturesBar[] {
  const startTime = getTradingDayStart().getTime();
  return bars
    .filter(bar => {
      const ts = new Date(bar.timestamp).getTime();
      return !Number.isNaN(ts) && ts >= startTime;
    })
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
}

// Load Historical Data
async function loadHistoricalData() {
  log('Loading historical data for full trading day (6pm ET - 5pm ET)...', 'info');

  try {
    // Ensure we have a resolved contractId
    if (!contractId) {
      const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
      if (!metadata) {
        log(`Failed to resolve TopstepX metadata for ${SYMBOL} (historical load)`, 'error');
        return;
      }
      contractId = metadata.id;
    }

    const endTime = new Date();
    const startTime = getTradingDayStart();

    log(`📅 Trading day: ${startTime.toISOString()} to ${endTime.toISOString()}`, 'info');

    // Use 10-second bars for accurate volume profile with full trading day coverage
    // Full session = ~18.5 hours = 6,660 10-second bars (fits in single request!)
    log('Fetching 10-second bars for full trading day...', 'info');

    const historicalBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
      unit: 1,        // Seconds
      unitNumber: 10, // 10-second bars (full day coverage, good granularity)
      limit: 20000,   // Max per request (should be enough for full day)
    });

    if (historicalBars && historicalBars.length > 0) {
      historicalSessionBars = historicalBars;  // Store in separate array
      bars = historicalBars.slice();  // Initialize bars with historical data
      volumeProfile = calculateVolumeProfile(historicalSessionBars);  // Use historical for profile
      marketStructure.state = detectMarketState();
      cvdMinuteBars = [];
      currentCvdBar = null;
      orderFlowData.cvd = 0;
      orderFlowData.cvdHistory = [];
      orderFlowData.bigPrints = [];
      orderFlowData.footprintImbalance = {};
      log(`✅ Loaded ${historicalSessionBars.length} historical 10-second bars from TopstepX`, 'success');
    } else {
      log('No historical bars returned from TopstepX', 'warning');
    }
    await refreshHigherTimeframes(true);
    await refreshRecentVolumeProfiles(true);
  } catch (error: any) {
    log(`Failed to load historical data: ${error.message}`, 'error');
  }
}

// Main Function
async function main() {
  // Add optional startup delay to stagger multiple agents (reduces rate limiting)
  const startupDelayMs = parseInt(process.env.STARTUP_DELAY_MS || '0', 10);
  if (startupDelayMs > 0) {
    console.log(`⏳ Startup delay: ${startupDelayMs}ms to avoid rate limiting...`);
    await new Promise(resolve => setTimeout(resolve, startupDelayMs));
  }

  console.log('\n' + '='.repeat(80));
  console.log('🧠 FABIO AGENT - VOLUME PROFILE & ORDER FLOW TRADING');
  console.log('='.repeat(80));
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Playbook: ${fabioPlaybook.philosophy}`);
  console.log(`Risk per trade: ${fabioPlaybook.riskRules.riskPerTradePctMin}-${fabioPlaybook.riskRules.riskPerTradePctMax}%`);
  console.log(`Dashboard: http://localhost:${DASHBOARD_PORT}`);
  console.log('='.repeat(80) + '\n');

  try {
    // Initialize components
    await initDashboard();

    // Fetch real account balance from TopStepX
    try {
      const accounts = await fetchTopstepXAccounts(true);
      const myAccount = accounts.find(acc => acc.id === ACCOUNT_ID);
      if (myAccount && myAccount.balance) {
        accountBalance = myAccount.balance;
        log(`✅ Fetched real account balance: $${accountBalance.toFixed(2)}`, 'success');
      } else {
        log(`⚠️  Could not find account ${ACCOUNT_ID}, using default balance: $${accountBalance}`, 'warning');
      }
    } catch (error: any) {
      log(`⚠️  Failed to fetch account balance: ${error.message}, using default: $${accountBalance}`, 'warning');
    }

    // Resolve contract ID FIRST before loading historical data
    const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
    if (!metadata) {
      throw new Error(`Failed to resolve contract for symbol ${SYMBOL}`);
    }
    contractId = metadata.id;
    log(`Resolved ${SYMBOL} to contractId=${contractId}`, 'info');

    await loadHistoricalData();

    await connectMarketData();

    // Create order manager
    const { orderManager: om } = await createProjectXRest();
    orderManager = om;

    log('🚀 FABIO AGENT FULLY INITIALIZED', 'success');
    log('Waiting for market conditions...', 'info');

  } catch (error: any) {
    log(`Failed to initialize: ${error.message}`, 'error');
    process.exit(1);
  }
}

// Graceful Shutdown
process.on('SIGINT', async () => {
  log('Shutting down gracefully...', 'warning');

  if (marketHub) await marketHub.stop();
  if (userHub) await userHub.stop();
  if (server) server.close();

  process.exit(0);
});

// Start the agent
main().catch(console.error);
