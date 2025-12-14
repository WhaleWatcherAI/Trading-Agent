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
import path from 'path';
import { writeFileSync, readFileSync, existsSync } from 'fs';
import fetch from 'node-fetch';
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
import { analyzeFuturesMarket, buildDecisionPromptPayload, FuturesMarketData, OpenAITradingDecision } from './lib/qwenTradingAgent';
import { cancelActiveRequest, resetOllamaOnStartup } from './lib/ollamaClient';
import { createExecutionManager } from './lib/executionManager';
import { belleCurveProfitProtection } from './lib/riskManagementAgent';
import { buildMlFeatureSnapshot } from './lib/mlFeatureExtractor';
import { predictMetaLabel } from './lib/mlMetaLabelService';

// Configuration - GOLD (GC) Instance
// GCG6 = Gold Feb 2026 (front month contract)
const SYMBOL = process.env.TOPSTEPX_SYMBOL || 'GCG6';
const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '0');
const DASHBOARD_PORT = parseInt(process.env.DASHBOARD_PORT || '3338'); // Different port for GC instance
const SOCKET_PATH = process.env.SOCKET_PATH || '';
const CONTRACTS = 1;
// REMOVED: Timer-based analysis - now using continuous sequential mode
// const ANALYSIS_INTERVAL_MS = 15_000;
const RISK_MGMT_INTERVAL_MS = 10_000; // Risk management checks every 10 seconds for active drawdown prevention
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:7b';

// DATA MODE CONFIGURATION
// - 'rest': REST only - no WebSocket, most stable but no L2/trade data
// - 'hybrid': REST for positions + Market WebSocket for L2/trades (recommended)
// - 'websocket': Full WebSocket mode - may get kicked if multiple sessions
const DATA_MODE = (process.env.DATA_MODE || 'hybrid') as 'rest' | 'hybrid' | 'websocket';
const REST_POLL_INTERVAL_MS = 3000; // Poll REST every 3 seconds for position data

// Legacy flag for backwards compatibility
const REST_ONLY_MODE = DATA_MODE === 'rest';

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
let userHubReconnecting = false; // Lock to prevent concurrent reconnection attempts
let userHubAccountId: number | null = null; // Track account ID for reconnection
let userHubReconnectAttempts = 0; // Exponential backoff counter
let userHubReconnectTimer: ReturnType<typeof setTimeout> | null = null; // Track pending reconnect timer
let marketHubReconnectAttempts = 0; // Exponential backoff counter
let marketHubReconnectTimer: ReturnType<typeof setTimeout> | null = null; // Track pending reconnect timer
let marketHubLastTokenRefresh = 0; // Track when token was last refreshed
const TOKEN_REFRESH_INTERVAL_MS = 30 * 60 * 1000; // Refresh token every 30 minutes
let orderManager: any = null;
let io: Server;
let app: express.Application;
let server: http.Server;
let contractId: string = ''; // Resolved TopstepX contract ID for SYMBOL

// Market Data Storage
let bars: TopstepXFuturesBar[] = [];
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
let oneSecondBars: TopstepXFuturesBar[] = [];  // 1-second bars for volume profile
let lastOneSecondBarRefresh = 0;
let cvdMinuteBars: CurrentCvdBar[] = [];

// 5-minute footprint candle tracking (per-price buy/sell volume)
interface FootprintLevel {
  price: number;
  buyVolume: number;
  sellVolume: number;
  delta: number;
  imbalance: 'buy' | 'sell' | 'neutral';
}
interface FootprintCandle {
  timestamp: string;
  levels: Map<number, { buyVolume: number; sellVolume: number }>;
}
let currentFootprintCandle: FootprintCandle | null = null;
let footprintCandles: Array<{
  timestamp: string;
  levels: FootprintLevel[];
}> = [];

let marketHubState: 'connected' | 'reconnecting' | 'disconnected' = 'disconnected';
let marketHubReconnecting = false; // Lock to prevent concurrent reconnection attempts
let lastMarketHubEventMs = 0;
let lastMarketHubDisconnectMs: number | null = null;
let marketHubWatchdog: NodeJS.Timeout | null = null;

function rehydrateCvdFromSnapshots(symbol: string): number {
  try {
    const file = path.resolve(__dirname, 'ml', 'data', 'snapshots.jsonl');
    if (!existsSync(file)) return 0;
    const lines = readFileSync(file, 'utf-8').trim().split('\n');
    for (let i = lines.length - 1; i >= 0; i -= 1) {
      const line = lines[i].trim();
      if (!line) continue;
      try {
        const parsed = JSON.parse(line) as { symbol?: string; features?: Record<string, any> };
        if (parsed.symbol !== symbol) continue;
        const cvdVal = parsed.features?.cvd_value;
        const num = typeof cvdVal === 'string' ? Number(cvdVal) : cvdVal;
        if (Number.isFinite(num)) {
          return Number(num);
        }
      } catch {
        continue;
      }
    }
  } catch (err) {
    console.warn('[CVD] Failed to rehydrate from snapshots:', (err as Error)?.message || err);
  }
  return 0;
}

// Position Management
let currentPosition: any = null;
let lastRiskMgmtTime = Date.now(); // Initialize to now to prevent immediate spam
let lastRiskMgmtDecision: any = null; // Store last Risk Management decision for dashboard
let accountBalance = 50000;

// OpenAI + Execution Integration
let executionManager: ExecutionManager | null = null;
let realizedPnL = 0; // Track realized P&L from closed positions

// Continuous Sequential Qwen Analysis - runs in dedicated loop, no timer
// One request at a time: when response received, immediately send next
let latestQwenDecision: OpenAITradingDecision | null = null; // Latest decision from continuous loop
let latestQwenDecisionTime = 0; // When the latest decision was received
let qwenAnalysisLoopRunning = false; // Is the continuous analysis loop active?
let qwenRequestCount = 0; // Track number of requests for logging

// Utility Functions
function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}][FABIO] ${message}`);

  if (io) {
    io.emit('log', { timestamp, message, type });
  }
}

async function emitR1Advisory(marketData: FuturesMarketData) {
  try {
    const { systemInstructions, decisionPrompt } = buildDecisionPromptPayload(marketData);
    const hardGuards = `
STRICT CONSTRAINTS (apply to the JSON you return):
- Use ONLY the provided auction/order-flow/profile fields. Do NOT mention EMA, SMA, MACD, RSI, generic TA indicators, or "broader market news".
- Keep the Fabio role: auction/location/flow/EV. No generic disclaimers.
- Output JSON ONLY per the schema. No prose, no markdown, no extra fields.
- If a value is unknown, set it to null; never invent data.`;
    const combinedPrompt = `${systemInstructions}\n\n${decisionPrompt}\n${hardGuards}\n\nRespond ONLY with valid JSON per the schema in the prompt. No extra text.`;
    log(`[R1] sending prompt (len=${combinedPrompt.length}) symbol=${marketData.symbol}`, 'info');
    const ollamaHost = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
    const url = `${ollamaHost.replace(/\/$/, '')}/api/generate`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: combinedPrompt,
        stream: false,
        temperature: 0.2,
        top_p: 0.9,
        repeat_penalty: 1.1,
      }),
    });
    const data: any = await res.json().catch(() => ({}));
    const responseText = data?.response || JSON.stringify(data);
    io?.emit('r1_decision', {
      timestamp: new Date().toISOString(),
      response: responseText,
      prompt: combinedPrompt,
      symbol: marketData.symbol,
    });
  } catch (err: any) {
    log(`[R1] advisory failed: ${err.message || err}`, 'warning');
  }
}

/**
 * CONTINUOUS SEQUENTIAL QWEN ANALYSIS LOOP
 * Runs independently - one request at a time with zero overlap and zero gaps
 * When one response is received, immediately starts the next request
 * This prevents Ollama queue backup that causes timeouts
 */
async function runContinuousQwenAnalysis() {
  if (qwenAnalysisLoopRunning) {
    log('🔄 [Qwen] Analysis loop already running, skipping duplicate start', 'warning');
    return;
  }

  qwenAnalysisLoopRunning = true;
  log('🚀 [Qwen] Starting CONTINUOUS SEQUENTIAL analysis loop (no timer, immediate next on completion)', 'success');

  while (qwenAnalysisLoopRunning) {
    // Check if we have an open position
    const activePos = executionManager?.getActivePosition();
    if (activePos) {
      // Position open - run BELLE CURVE profit protection (no LLM, instant!)
      // Get current price from latest bar
      const currentPrice = bars && bars.length > 0 ? bars[bars.length - 1].close : 0;

      if (currentPrice > 0) {
        // Run belle curve profit protection (pure algorithmic, no LLM delay)
        const riskDecision = belleCurveProfitProtection(activePos, currentPrice);

        // Apply the risk decision if it calls for bracket adjustments
        if (riskDecision.action !== 'HOLD_BRACKETS' && executionManager) {
          log(`🔔 [BelleCurve] Applying: ${riskDecision.action}`, 'info');
          if (riskDecision.newStopLoss !== null) {
            log(`   Stop: ${activePos.stopLoss.toFixed(2)} -> ${riskDecision.newStopLoss.toFixed(2)}`, 'info');
          }
          await executionManager.adjustActiveProtection(
            riskDecision.newStopLoss,
            riskDecision.newTarget,
            undefined, // positionVersion
            currentPrice // pass live market price for stop validation
          );

          // Emit to dashboard
          if (io) {
            io.emit('risk_decision', {
              timestamp: new Date().toISOString(),
              action: riskDecision.action,
              newStopLoss: riskDecision.newStopLoss,
              newTarget: riskDecision.newTarget,
              reasoning: riskDecision.reasoning,
              urgency: riskDecision.urgency,
              riskLevel: riskDecision.riskLevel,
            });
          }
        }
      }

      // Small delay before next check (belle curve is instant, don't spam)
      await new Promise(resolve => setTimeout(resolve, 500));
      continue; // Continue to next loop iteration
    }

    // Need bars data to analyze
    if (!bars || bars.length < 10) {
      await new Promise(resolve => setTimeout(resolve, 500));
      continue;
    }

    qwenRequestCount++;
    const requestNum = qwenRequestCount;
    const startTime = Date.now();

    try {
      // First 3 requests get extended timeout for cold start context loading
      const isWarmupPhase = requestNum <= 3;
      log(`🧠 [Qwen] #${requestNum} Starting analysis...${isWarmupPhase ? ' (extended 90s timeout for cold start)' : ''}`, 'info');

      // Build market data using the proper factory function (same as handleDataEvent uses)
      const marketData = buildFuturesMarketData(
        SYMBOL,
        bars,
        volumeProfile,
        orderFlowData,
        marketStructure,
        currentCvdBar,
        accountBalance,
        null, // currentPosition - null since we're looking for entries
        realizedPnL,
        higherTimeframeSnapshots,
        recentSessionProfiles,
        cvdMinuteBars,
        {
          marketHubState,
          lastMarketHubEventAgoSec: lastMarketHubEventMs ? Number(((Date.now() - lastMarketHubEventMs) / 1000).toFixed(1)) : null,
          lastMarketHubDisconnectAgoSec: lastMarketHubDisconnectMs ? Number(((Date.now() - lastMarketHubDisconnectMs) / 1000).toFixed(1)) : null,
        },
        footprintCandles
      );

      // Make the request - this will await until complete (success or failure)
      // First 3 requests get 90s timeout (cold start context loading), then 30s for warmed up requests
      const timeoutMs = requestNum <= 3 ? 90000 : 30000;
      const result = await analyzeFuturesMarket(marketData, timeoutMs);

      const elapsed = Date.now() - startTime;

      // Store the result for the main loop to use
      // IMPORTANT: Only overwrite if:
      // 1. No existing decision (null)
      // 2. New decision has higher confidence
      // 3. Existing decision has < 75% confidence (not actionable anyway)
      // This prevents high-confidence signals from being overwritten before the main loop processes them
      const existingConf = latestQwenDecision?.confidence ?? 0;
      const newConf = result.confidence ?? 0;
      const shouldReplace = !latestQwenDecision || newConf >= existingConf || existingConf < 75;

      if (shouldReplace) {
        latestQwenDecision = result;
        latestQwenDecisionTime = Date.now();
      } else {
        log(`🔒 [Qwen] #${requestNum} Keeping existing ${existingConf}% decision (new was ${newConf}%)`, 'info');
      }

      const clampedConfidence = Math.max(0, Math.min(100, result.confidence || 0));
      log(`✅ [Qwen] #${requestNum} Complete in ${elapsed}ms: ${result.decision} @ ${result.entryPrice?.toFixed(2) || 'null'} (${clampedConfidence}% conf)`, 'success');

      // Emit to dashboard
      if (io && result) {
        const llmPayload = {
          timestamp: new Date().toISOString(),
          decision: result.decision,
          reasoning: result.reasoning,
          confidence: clampedConfidence / 100,
          entryPrice: result.entryPrice,
          stopLoss: result.stopLoss,
          target: result.target,
          riskManagementReasoning: result.riskManagementReasoning || null,
          inferredRegime: result.inferredRegime,
          trade_decisions: result.decision !== 'HOLD' ? [result.decision] : [],
        };
        log(`📤 [Qwen] Emitting to dashboard: ${result.decision}`, 'info');
        io.emit('llm_decision', llmPayload);
      }

      // ========== IMMEDIATE EXECUTION for high-confidence signals ==========
      // Execute RIGHT HERE in the Qwen loop - don't wait for processMarketUpdate
      if (executionManager && result.decision !== 'HOLD' && clampedConfidence >= 75) {
        log(`🚀 [Qwen] HIGH CONFIDENCE ${clampedConfidence}% - EXECUTING IMMEDIATELY`, 'success');
        const currentPrice = bars.length > 0 ? bars[bars.length - 1].close : result.entryPrice || 0;
        try {
          const executionResult = await processOpenAIDecision(
            result,
            executionManager,
            currentPrice,
            SYMBOL,
            orderFlowData,
            volumeProfile,
            marketStructure
          );
          if (executionResult.executed) {
            log(`✅ [Qwen] Trade executed successfully!`, 'success');
            currentPosition = executionManager.getActivePosition();
          }
        } catch (execError: any) {
          log(`❌ [Qwen] Immediate execution failed: ${execError.message}`, 'error');
        }
      }

      // Immediately continue to next iteration - no wait!
      // The while loop starts the next request right away

    } catch (error: any) {
      const elapsed = Date.now() - startTime;
      log(`❌ [Qwen] #${requestNum} Failed after ${elapsed}ms: ${error.message}`, 'error');

      // On error, small delay before retry to avoid hammering
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }

  log('🛑 [Qwen] Continuous analysis loop stopped', 'warning');
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

async function refreshOneSecondBars() {
  // Refresh 1-second bars every 5 seconds for accurate volume profile
  const refreshIntervalMs = 5 * 1000;
  if (Date.now() - lastOneSecondBarRefresh < refreshIntervalMs) {
    return;
  }
  if (!contractId) return;

  try {
    const now = new Date();
    // Fetch last 30 minutes of 1-second bars (1800 bars max)
    const lookback = new Date(now.getTime() - 30 * 60 * 1000);

    const bars1s = await fetchTopstepXFuturesBars({
      contractId,
      startTime: lookback.toISOString(),
      endTime: now.toISOString(),
      unit: 1,        // 1 = Seconds
      unitNumber: 1,  // 1-second bars
      limit: 2000,
      includePartialBar: true,
    });

    oneSecondBars = bars1s || [];
    lastOneSecondBarRefresh = Date.now();
    log(`📊 Refreshed 1-second bars for volume profile (${oneSecondBars.length} bars)`, 'debug');
  } catch (error: any) {
    log(`⚠️ Failed to refresh 1-second bars: ${error.message}`, 'warning');
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

    // Use 1-second bars for current session volume profile (more accurate)
    // Fetch last 4 hours of 1-second bars for current session
    const currentSessionLookback = new Date(now.getTime() - 4 * 60 * 60 * 1000);
    const currentSessionBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: currentSessionLookback.toISOString(),
      endTime: now.toISOString(),
      unit: 1,        // 1 = Seconds
      unitNumber: 1,  // 1-second bars
      limit: 15000,   // 4 hours = 14400 seconds
      includePartialBar: true,
    });

    // Store 1-second bars for real-time volume profile
    oneSecondBars = currentSessionBars || [];
    log(`📊 Loaded ${oneSecondBars.length} 1-second bars for volume profile`, 'info');

    // Also fetch 5-min bars for historical sessions (last 6 days)
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
  // Use 1-second bars for volume profile if available, fallback to 1-minute bars
  const barsForProfile = oneSecondBars.length >= 60 ? oneSecondBars : bars;
  if (barsForProfile.length < 20) return 'balanced';

  const recentBars = barsForProfile.slice(-300);  // Use more bars for 1s data (5 min worth)
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
      return 'balanced_with_failed_breakout_above';
    }
    if (recentLow < profile.val && currentPrice > profile.val) {
      return 'balanced_with_failed_breakout_below';
    }
    return 'balanced';
  }

  // Out of balance if strong directional move
  const trend = recentBars.reduce((sum, bar, i) => {
    if (i === 0) return 0;
    return sum + (bar.close - recentBars[i - 1].close);
  }, 0);

  if (trend > priceRange * 0.3) {
    return 'out_of_balance_uptrend';
  } else if (trend < -priceRange * 0.3) {
    return 'out_of_balance_downtrend';
  }

  return 'balanced';
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

  if (state.includes('out_of_balance')) {
    model = 'trend_continuation';

    if (state === 'out_of_balance_uptrend' && location === 'at_lvn' && orderFlowConfirmed) {
      if (orderFlowData.cvd > 0 && orderFlowData.bigPrints.filter(p => p.side === 'buy').length > 0) {
        entrySide = 'long';
        reason = 'Trend continuation: Pullback to LVN with buy aggression confirmed';
        confidence = 80;
      }
    } else if (state === 'out_of_balance_downtrend' && location === 'at_lvn' && orderFlowConfirmed) {
      if (orderFlowData.cvd < 0 && orderFlowData.bigPrints.filter(p => p.side === 'sell').length > 0) {
        entrySide = 'short';
        reason = 'Trend continuation: Pullback to LVN with sell aggression confirmed';
        confidence = 80;
      }
    }
  } else if (state.includes('failed_breakout')) {
    model = 'mean_reversion';

    if (state === 'balanced_with_failed_breakout_above' && location === 'at_vah' && orderFlowConfirmed) {
      // Failed breakout above means buy exhaustion - buyers couldn't push through
      if (orderFlowData.exhaustion.buy > 0.6) {
        entrySide = 'short';
        reason = 'Mean reversion: Failed breakout above VAH with buy exhaustion';
        confidence = 75;
      }
    } else if (state === 'balanced_with_failed_breakout_below' && location === 'at_val' && orderFlowConfirmed) {
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

  if (executionManager) {
    await executionManager.syncWithBrokerState();
    currentPosition = executionManager.getActivePosition();
  }

  // Update volume profile using 1-second bars for accuracy (or fallback to 1-min bars)
  const tradingDayStart = getTradingDayStart();
  // Prefer 1-second bars if available, otherwise use 1-minute bars
  if (oneSecondBars.length >= 100) {
    // Use 1-second bars for more accurate volume profile
    const day1sBars = oneSecondBars.filter(bar => {
      const ts = new Date(bar.timestamp).getTime();
      return !Number.isNaN(ts) && ts >= tradingDayStart.getTime();
    });
    const profileSource = day1sBars.length > 0 ? day1sBars : oneSecondBars;
    volumeProfile = calculateVolumeProfile(profileSource);
  } else {
    // Fallback to 1-minute bars
    const dayBars = bars.filter(bar => {
      const ts = new Date(bar.timestamp).getTime();
      return !Number.isNaN(ts) && ts >= tradingDayStart.getTime();
    });
    const profileSource = dayBars.length > 0 ? dayBars : bars.slice(-50);
    volumeProfile = calculateVolumeProfile(profileSource);
  }

  // Update market structure (existing)
  marketStructure.state = detectMarketState();

  await refreshHigherTimeframes();
  await refreshRecentVolumeProfiles();
  await refreshOneSecondBars();  // Refresh 1-second bars for accurate volume profile

  // ========== CRITICAL: Sync position from ExecutionManager FIRST ==========
  // This MUST happen before any trading decisions to prevent duplicate entries
  if (executionManager) {
    const previousPosition = currentPosition;
    currentPosition = executionManager.getActivePosition();

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
    cvdMinuteBars,
    {
      marketHubState,
      lastMarketHubEventAgoSec: lastMarketHubEventMs ? Number(((Date.now() - lastMarketHubEventMs) / 1000).toFixed(1)) : null,
      lastMarketHubDisconnectAgoSec: lastMarketHubDisconnectMs ? Number(((Date.now() - lastMarketHubDisconnectMs) / 1000).toFixed(1)) : null,
    },
    footprintCandles
  );

  // Lightweight ML prefilter (probabilities only, no gating yet)
  try {
    const mlSnapshot = buildMlFeatureSnapshot(marketData);
    const mlScores = predictMetaLabel(mlSnapshot);
    if (mlScores) {
      marketData.mlScores = {
        ...mlScores,
        modelVersion: 'lightgbm-meta-label-v0',
      };
    }
  } catch (error: any) {
    console.warn('[ML] Meta-label scoring failed:', error?.message || error);
  }

  // ========== CONTINUOUS SEQUENTIAL QWEN ANALYSIS ==========
  // Analysis runs in a separate loop (runContinuousQwenAnalysis)
  // Here we just read the latest decision from that loop
  const nowMs = Date.now();
  const openaiDecision = latestQwenDecision;

  // Clear the decision after reading so we don't act on the same decision twice
  // The continuous loop will populate a new one soon
  if (openaiDecision) {
    latestQwenDecision = null;
  }

  // OLD RULE-BASED SYSTEM - DISABLED (legacy reasoner-only path)
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
      // Call belle curve immediately after entering position (instant, no LLM)
      if (currentPosition && bars && bars.length > 0) {
        log('🛡️ [BelleCurve] 🚀 NEW POSITION - Running IMMEDIATE profit protection check...', 'success');

        // CRITICAL: Set timer IMMEDIATELY to prevent regular 30s check from also firing
        lastRiskMgmtTime = nowMs;

        const currentPrice = bars[bars.length - 1].close;
        const riskDecision = belleCurveProfitProtection(currentPosition, currentPrice);

        log(`🛡️ [BelleCurve] Initial Decision: ${riskDecision.action} (${riskDecision.urgency} urgency)`, 'info');
        log(`🛡️ [BelleCurve] ${riskDecision.reasoning.substring(0, 200)}`, 'info');

        // Store for dashboard
        lastRiskMgmtDecision = {
          timestamp: new Date().toISOString(),
          action: riskDecision.action,
          urgency: riskDecision.urgency,
          reasoning: riskDecision.reasoning,
          newStopLoss: riskDecision.newStopLoss,
          newTarget: riskDecision.newTarget,
        };

        // Apply risk management decision (belle curve doesn't recommend CLOSE, just stop adjustments)
        if (riskDecision.action !== 'HOLD_BRACKETS') {
          // Adjust brackets immediately
          const adjusted = await executionManager.adjustActiveProtection(
            riskDecision.newStopLoss,
            riskDecision.newTarget,
            riskDecision.positionVersion,
            currentPrice // pass live market price for stop validation
          );
          if (adjusted) {
            log(`🛡️ [BelleCurve] ✅ Initial brackets set - Stop: ${riskDecision.newStopLoss?.toFixed(2) || 'unchanged'}`, 'success');
            currentPosition = executionManager.getActivePosition();
          }
        }
      }
    }
  }
  // OLD RULE-BASED FALLBACK - DISABLED (legacy reasoner-only path)
  // else if (ruleBasedDecision.entry.side && ruleBasedDecision.entry.confidence >= 70 && !currentPosition) {
  //   log(`📊 Rule-based Decision: ${ruleBasedDecision.entry.side.toUpperCase()} (Confidence: ${ruleBasedDecision.entry.confidence}%)`, 'info');
  //   executeEntry(ruleBasedDecision);
  // }

  // ========== NEW: Update position and check exits ==========
  if (executionManager && currentPosition) {
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
        // Ghost position detection happens at top of processMarketUpdate() via syncWithBrokerState()
        // If we reach here, currentPosition is already synced with broker state

        try {
          log(`🛡️ [BelleCurve] 🎯 ACTIVE POSITION - Running profit protection... (${Math.floor(timeSinceLastRiskMgmt / 1000)}s since last check)`, 'info');

          // CRITICAL: Update timer IMMEDIATELY to prevent duplicate calls during async operation
          lastRiskMgmtTime = Date.now();

          // Safety check: ensure we have bars before analyzing
          if (bars && bars.length > 0) {
            const currentPrice = bars[bars.length - 1].close;
            const riskDecision = belleCurveProfitProtection(currentPosition, currentPrice);

          log(`🛡️ [BelleCurve] Decision: ${riskDecision.action} (${riskDecision.urgency} urgency)`, 'info');
          log(`🛡️ [BelleCurve] ${riskDecision.reasoning.substring(0, 200)}`, 'info');

          // Store for dashboard broadcast
          lastRiskMgmtDecision = {
            timestamp: new Date().toISOString(),
            action: riskDecision.action,
            urgency: riskDecision.urgency,
            reasoning: riskDecision.reasoning,
            newStopLoss: riskDecision.newStopLoss,
            newTarget: riskDecision.newTarget,
          };

          // Apply risk management decision (belle curve doesn't recommend CLOSE, just stop adjustments)
          if (riskDecision.action !== 'HOLD_BRACKETS') {
            // Adjust brackets
            const adjusted = await executionManager.adjustActiveProtection(
              riskDecision.newStopLoss,
              riskDecision.newTarget,
              riskDecision.positionVersion,
              currentPrice // pass live market price for stop validation
            );

            if (adjusted) {
              log(`🛡️ [BelleCurve] ✅ Brackets adjusted - Stop: ${riskDecision.newStopLoss?.toFixed(2) || 'unchanged'}`, 'success');
              currentPosition = executionManager.getActivePosition(); // Refresh position
            } else {
              log('🛡️ [BelleCurve] ⚠️ Failed to adjust brackets', 'warning');
            }
          } else {
            log('🛡️ [BelleCurve] ✓ Holding current brackets - no adjustments needed', 'info');
          }
          } else {
            log('🛡️ [BelleCurve] ⚠️ No bars available yet, skipping profit protection', 'warn');
          }

        } catch (error: any) {
          log(`🛡️ [RiskMgmt] ❌ Error in risk management: ${error.message}`, 'error');
          // Timer already set at start of try block, don't reset
        }
      } else {
        // Log why we're not running (for debugging silent periods)
        if (currentPosition && timeSinceLastRiskMgmt < RISK_MGMT_INTERVAL_MS && timeSinceLastRiskMgmt > 0) {
          const secondsRemaining = Math.ceil((RISK_MGMT_INTERVAL_MS - timeSinceLastRiskMgmt) / 1000);
          log(`🛡️ [RiskMgmt] ⏳ Next check in ${secondsRemaining}s (${Math.floor(timeSinceLastRiskMgmt / 1000)}s/${RISK_MGMT_INTERVAL_MS / 1000}s elapsed)`, 'debug');
        }
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
  // OLD RULE-BASED SIGNAL EMISSION - DISABLED (legacy reasoner-only path)
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

  // Emit status event with enhanced position stats
  let positionStats = null;
  if (currentPosition && currentBar) {
    const currentPrice = currentBar.close;
    const profitLoss = currentPrice - currentPosition.entryPrice;
    const profitLossPoints = currentPosition.side === 'long' ? profitLoss : -profitLoss;
    const distanceToTarget = currentPosition.side === 'long'
      ? currentPosition.target - currentPosition.entryPrice
      : currentPosition.entryPrice - currentPosition.target;
    const distanceToStop = currentPosition.side === 'long'
      ? currentPosition.entryPrice - currentPosition.stopLoss
      : currentPosition.stopLoss - currentPosition.entryPrice;
    const percentToTarget = distanceToTarget > 0 ? (profitLossPoints / distanceToTarget) * 100 : 0;
    const pnlDollars = profitLossPoints * 100 * currentPosition.contracts; // GC = $100/point

    positionStats = {
      side: currentPosition.side,
      entry_price: currentPosition.entryPrice,
      current_price: currentPrice,
      contracts: currentPosition.contracts,
      pnl_points: profitLossPoints,
      pnl_dollars: pnlDollars,
      stop_loss: currentPosition.stopLoss,
      target: currentPosition.target,
      percent_to_target: percentToTarget,
      distance_to_target: distanceToTarget,
      distance_to_stop: distanceToStop,
      at_breakeven: profitLossPoints >= 0 && currentPosition.stopLoss >= currentPosition.entryPrice - 0.5,
    };
  }

  io.emit('status', {
    balance: accountBalance,
    position: positionStats,
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
    cors: { origin: '*', methods: ['GET', 'POST'] },
    path: SOCKET_PATH || undefined,
    pingTimeout: 60000,      // 60s before considering connection dead
    pingInterval: 10000,     // 10s keep-alive ping
    transports: ['websocket', 'polling'], // Allow fallback to polling
  });

  // Serve dashboard HTML
  app.get('/', (req, res) => {
    // Use GC-specific dashboard
    const dashboardPath = path.resolve(__dirname, 'public', 'fabio-agent-dashboard-mgc.html');
    res.sendFile(dashboardPath);
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

  server.listen(DASHBOARD_PORT, '0.0.0.0', () => {
    log(`Dashboard running on http://localhost:${DASHBOARD_PORT}`, 'success');
  });
}

// Connect to Market Data
async function connectMarketData() {
  // Prevent concurrent reconnection attempts
  if (marketHubReconnecting) {
    log('⚠️ Market hub reconnection already in progress, skipping', 'warning');
    return;
  }

  // Stop existing hub if present
  if (marketHub) {
    try {
      await marketHub.stop();
    } catch (e) {
      // Ignore stop errors
    }
    marketHub = null;
  }

  marketHubReconnecting = true;

  // Get FRESH token for connection - critical for long-running sessions
  // Token may have expired since last connection, always refresh on reconnect
  const token = await authenticate();
  marketHubLastTokenRefresh = Date.now();
  log(`🔐 Fresh token obtained for market hub (will refresh in ${TOKEN_REFRESH_INTERVAL_MS / 60000}min)`, 'info');

  // Rehydrate CVD baseline from last ML snapshot so restart doesn't zero out flow context
  const rehydratedCvd = rehydrateCvdFromSnapshots(SYMBOL);
  if (Number.isFinite(rehydratedCvd) && rehydratedCvd !== 0) {
    orderFlowData.cvd = rehydratedCvd;
    orderFlowData.cvdHistory = [{ timestamp: Date.now(), cvd: rehydratedCvd, delta: 0 }];
    log(`♻️ Rehydrated CVD baseline: ${rehydratedCvd.toFixed(2)}`, 'info');
  }

  // Create market hub - simplified config matching stable deepseek version
  // Simplified config matching stable deepseek version
  // REMOVED: accessTokenFactory (causes token conflicts during reconnect)
  // REMOVED: encodeURIComponent (may corrupt token)
  // KEPT: withServerTimeout (120s - generous timeout to survive any Ollama delays)
  log(`🔑 Token obtained (length: ${token?.length || 0})`, 'info');

  // DISABLED: withAutomaticReconnect - it uses stale token from URL
  // SignalR auto-reconnect reuses the original connection URL which has the old token
  // This causes immediate disconnect after reconnect (server rejects expired token)
  // Instead, we handle ALL reconnection in onclose which creates a fresh hub with fresh token

  marketHub = new HubConnectionBuilder()
    .withUrl(`https://rtc.topstepx.com/hubs/market?access_token=${token}`, {
      transport: HttpTransportType.WebSockets,
      skipNegotiation: true,
    })
    // NO withAutomaticReconnect - onclose handler will create fresh connection with fresh token
    .withServerTimeout(120000) // 2 minutes - generous to survive any Ollama/system load
    .configureLogging(LogLevel.Warning)
    .build();

  // NOTE: onreconnecting and onreconnected are NOT used since we disabled auto-reconnect
  // All reconnection is handled by onclose -> connectMarketData() which gets fresh token

  marketHub.onclose(error => {
    marketHubState = 'disconnected';
    lastMarketHubDisconnectMs = Date.now();
    const errorMsg = error?.message || 'no error provided';
    log(`❌ Market hub disconnected: ${errorMsg}`, 'error');

    // ALWAYS clear the reconnecting flag on close - prevents deadlock
    // The reconnect timer will set it again when it starts
    marketHubReconnecting = false;

    // Cancel any previous pending timer before scheduling a new one
    if (marketHubReconnectTimer) {
      clearTimeout(marketHubReconnectTimer);
      marketHubReconnectTimer = null;
    }

    // Check if this looks like a token expiry (immediate disconnect after connect)
    const timeSinceTokenRefresh = Date.now() - marketHubLastTokenRefresh;
    const isLikelyTokenExpiry = timeSinceTokenRefresh > TOKEN_REFRESH_INTERVAL_MS;

    // If token is old, try immediate reconnect with fresh token (no backoff)
    // Otherwise use exponential backoff
    let backoffDelay: number;
    if (isLikelyTokenExpiry && marketHubReconnectAttempts < 3) {
      // Token likely expired - fast retry with fresh token
      backoffDelay = 2000; // 2 second delay
      log(`🔐 Token is ${Math.round(timeSinceTokenRefresh / 60000)}min old - likely expired, fast retry with fresh token`, 'warning');
    } else {
      // Normal exponential backoff: 5s, 10s, 20s, 40s, max 60s (reduced from 120s)
      marketHubReconnectAttempts++;
      backoffDelay = Math.min(5000 * Math.pow(2, marketHubReconnectAttempts - 1), 60000);
    }

    log(`⏳ Will attempt market hub reconnection in ${Math.round(backoffDelay/1000)}s (attempt ${marketHubReconnectAttempts})`, 'info');

    marketHubReconnectTimer = setTimeout(async () => {
      marketHubReconnectTimer = null;
      if (marketHubState === 'disconnected') {
        log('🔄 Attempting manual reconnection after hub closure...', 'warning');
        try {
          await connectMarketData();
        } catch (reconnectErr: any) {
          log(`❌ Manual reconnection failed: ${reconnectErr?.message || reconnectErr}`, 'error');
          // Ensure flag is cleared even on error so next timer can try
          marketHubReconnecting = false;
        }
      }
    }, backoffDelay);
  });

  // Handle logout events from broker (session kicked, logged in elsewhere, etc.)
  marketHub.on('gatewaylogout', (data: any) => {
    log(`⚠️ [MarketHub] Received gatewaylogout: ${JSON.stringify(data)}`, 'warning');
    log(`💡 This usually means you're logged in elsewhere (web platform, another instance). Check and close other sessions.`, 'warning');
  });

  // Subscribe to market events
  marketHub.on('gatewayquote', (incomingContractId: string, quotes: any) => {
    if (incomingContractId !== contractId) return;
    lastMarketHubEventMs = Date.now();

    // Process Level 2 data
    if (quotes?.bids && quotes?.asks) {
      l2Data = [];
      for (let i = 0; i < Math.min(10, quotes.bids.length); i++) {
        l2Data.push({
          price: quotes.bids[i].price,
          bidSize: quotes.bids[i].size,
          askSize: quotes.asks[i]?.size || 0,
        });
      }
    }
    processMarketUpdate();
  });

  const handleDepthEvent = (incomingContractId: string, depth: any) => {
    if (incomingContractId !== contractId) return;
    lastMarketHubEventMs = Date.now();
    try {
      if (depth?.bids && depth?.asks) {
        l2Data = [];
        for (let i = 0; i < Math.min(10, depth.bids.length); i++) {
          l2Data.push({
            price: Number(depth.bids[i].price),
            bidSize: Number(depth.bids[i].size),
            askSize: Number(depth.asks[i]?.size ?? 0),
          });
        }
        processMarketUpdate();
      }
    } catch (err: any) {
      log(`⚠️ Failed to process depth event: ${err?.message || err}`, 'warning');
    }
  };

  // Depth stream (TopstepX may emit different casing)
  marketHub.on('gatewaydepth', handleDepthEvent);
  marketHub.on('gatewayDepth', handleDepthEvent);
  marketHub.on('GatewayDepth', handleDepthEvent);

  async function subscribeToMarketStreams(retryCount = 0): Promise<void> {
    const MAX_RETRIES = 5;
    const RETRY_DELAY_MS = 1000;

    if (!contractId) {
      log('⚠️ Cannot subscribe: contractId not set', 'warning');
      return;
    }

    if (!marketHub) {
      log('⚠️ Cannot subscribe: market hub not initialized', 'warning');
      return;
    }

    const hubState = marketHub.state as string;
    if (hubState !== 'Connected') {
      if (retryCount < MAX_RETRIES) {
        log(`⏳ Hub state is '${hubState}', waiting to subscribe (retry ${retryCount + 1}/${MAX_RETRIES})...`, 'info');
        setTimeout(() => subscribeToMarketStreams(retryCount + 1), RETRY_DELAY_MS);
      } else {
        log(`⚠️ Cannot subscribe: market hub state is '${hubState}' after ${MAX_RETRIES} retries`, 'warning');
      }
      return;
    }

    try {
      await Promise.all([
        marketHub.invoke('SubscribeContractQuotes', contractId),
        marketHub.invoke('SubscribeContractTrades', contractId),
        marketHub.invoke('SubscribeContractMarketDepth', contractId),
      ]);
      log(`✅ Subscribed to ${SYMBOL} market data (contractId=${contractId})`, 'success');
      // Only reset backoff AFTER subscription succeeds (not just start)
      // This prevents counter reset when connection briefly connects then drops
      marketHubReconnectAttempts = 0;
      log(`✅ Market hub fully connected and subscribed - backoff counter reset`, 'success');
    } catch (err: any) {
      log(`❌ Failed to subscribe to market streams: ${err?.message || err}`, 'error');
      if (retryCount < MAX_RETRIES) {
        log(`🔄 Retrying subscription in ${RETRY_DELAY_MS}ms (retry ${retryCount + 1}/${MAX_RETRIES})...`, 'info');
        setTimeout(() => subscribeToMarketStreams(retryCount + 1), RETRY_DELAY_MS);
      }
    }
  }

  // Execution / fills stream (if provided by TopstepX feed)
  // Note: Use execution events as fast invalidators for position version
  marketHub.on('gatewayexecution', async (id: string, exec: any) => {
    if (id !== contractId) return;
    lastMarketHubEventMs = Date.now();
    const status = (exec?.status || '').toString().toLowerCase();
    const filledQty = Number(exec?.filledSize ?? exec?.fillSize ?? exec?.lastQty ?? exec?.lastFillQty ?? 0);
    const isFill = status.includes('fill') || filledQty > 0;
    if (!isFill) return;
    if (executionManager) {
      executionManager.bumpVersionExternal();
    }
  });

  marketHub.on('gatewaytrade', (incomingContractId: string, trades: any[]) => {
    if (incomingContractId !== contractId) return;
    lastMarketHubEventMs = Date.now();
    log(`📊 Received ${trades.length} trade(s) for ${incomingContractId}`, 'info');
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

      // ========== 5-MINUTE FOOTPRINT CANDLE TRACKING ==========
      const fiveMinMs = 5 * 60 * 1000;
      const current5MinBoundary = Math.floor(now.getTime() / fiveMinMs) * fiveMinMs;
      const current5MinISO = new Date(current5MinBoundary).toISOString();

      // Check if we need to start a new 5-minute footprint candle
      if (!currentFootprintCandle || currentFootprintCandle.timestamp !== current5MinISO) {
        // Finalize previous candle if exists
        if (currentFootprintCandle) {
          const finalizedLevels: FootprintLevel[] = [];
          currentFootprintCandle.levels.forEach((vol, lvlPrice) => {
            const delta = vol.buyVolume - vol.sellVolume;
            const imbalance: 'buy' | 'sell' | 'neutral' =
              delta > vol.buyVolume * 0.3 ? 'buy' :
              delta < -vol.sellVolume * 0.3 ? 'sell' : 'neutral';
            finalizedLevels.push({
              price: lvlPrice,
              buyVolume: vol.buyVolume,
              sellVolume: vol.sellVolume,
              delta,
              imbalance,
            });
          });
          // Sort by price descending for better readability
          finalizedLevels.sort((a, b) => b.price - a.price);
          footprintCandles.push({
            timestamp: currentFootprintCandle.timestamp,
            levels: finalizedLevels,
          });
          // Keep last 12 candles (1 hour of 5-min candles)
          if (footprintCandles.length > 12) {
            footprintCandles.shift();
          }
          log(`📊 NEW 5-MIN FOOTPRINT: ${current5MinISO}, levels=${finalizedLevels.length}, total candles=${footprintCandles.length}`, 'info');
        }
        // Start new footprint candle
        currentFootprintCandle = {
          timestamp: current5MinISO,
          levels: new Map(),
        };
      }

      // Update current footprint candle with this trade
      const footprintPriceLevel = Math.round(price * 4) / 4; // Round to 0.25 tick
      if (!currentFootprintCandle.levels.has(footprintPriceLevel)) {
        currentFootprintCandle.levels.set(footprintPriceLevel, { buyVolume: 0, sellVolume: 0 });
      }
      const fpLevel = currentFootprintCandle.levels.get(footprintPriceLevel)!;
      if (side === 'Buy') {
        fpLevel.buyVolume += size;
      } else {
        fpLevel.sellVolume += size;
      }
      // ========== END FOOTPRINT CANDLE TRACKING ==========

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
  log(`Connected to market data hub (state: ${marketHub.state})`, 'success');
  marketHubState = 'connected';
  marketHubReconnecting = false; // Release lock on success
  // NOTE: marketHubReconnectAttempts is reset INSIDE subscribeToMarketStreams()
  // AFTER subscription succeeds - not here after start() - to prevent premature reset
  // when connection briefly connects then drops before subscription completes
  lastMarketHubEventMs = Date.now();
  lastMarketHubDisconnectMs = null;

  // Give the connection a moment to stabilize before proceeding
  await new Promise(resolve => setTimeout(resolve, 500));
  log(`Hub state after stabilization: ${marketHub.state}`, 'info');

  // If connection died during stabilization, onclose will handle reconnect
  // Just log the state and continue - subscription function will check state anyway
  if (marketHub.state !== 'Connected') {
    log(`⚠️ Market hub state changed to ${marketHub.state} during stabilization - onclose handler will reconnect`, 'warning');
    // Don't return - let the code continue to subscription which will retry
  }

  // Watchdog: health monitoring and proactive token refresh
  // 1. Proactive token refresh - reconnect BEFORE token expires (prevents immediate disconnect)
  // 2. Stale data detection - resubscribe if no data flowing
  if (marketHubWatchdog) clearInterval(marketHubWatchdog);
  marketHubWatchdog = setInterval(async () => {
    const now = Date.now();

    // PROACTIVE TOKEN REFRESH: Reconnect with fresh token before expiry
    // This prevents the "connect then immediate disconnect" loop
    const timeSinceTokenRefresh = now - marketHubLastTokenRefresh;
    if (timeSinceTokenRefresh > TOKEN_REFRESH_INTERVAL_MS && marketHub?.state === 'Connected') {
      log(`🔐 Token is ${Math.round(timeSinceTokenRefresh / 60000)}min old - proactively refreshing connection`, 'warning');
      try {
        await marketHub?.stop();
      } catch {}
      try {
        await connectMarketData();
        log(`✅ Proactive token refresh completed`, 'success');
      } catch (err: any) {
        log(`❌ Proactive token refresh failed: ${err?.message || err}`, 'error');
      }
      return; // Skip stale check since we just reconnected
    }

    // Only check staleness if hub thinks it's connected
    if (marketHub?.state !== 'Connected') {
      return; // SignalR auto-reconnect is handling this
    }

    const staleForMs = lastMarketHubEventMs ? now - lastMarketHubEventMs : 0;

    // If connected but no data for 60s, likely need to resubscribe
    if (staleForMs > 60_000) {
      log(`⚠️ Market hub connected but stale for ${Math.round(staleForMs / 1000)}s, resubscribing...`, 'warning');
      try {
        // Use correct TopStepX method names (same as initial subscription)
        if (contractId) {
          await Promise.all([
            marketHub.invoke('SubscribeContractQuotes', contractId),
            marketHub.invoke('SubscribeContractTrades', contractId),
            marketHub.invoke('SubscribeContractMarketDepth', contractId),
          ]);
          log(`✅ Resubscribed to ${contractId}`, 'success');
          lastMarketHubEventMs = Date.now(); // Reset timer
        }
      } catch (err: any) {
        log(`❌ Resubscribe failed: ${err?.message || err}, will retry connection`, 'error');
        try {
          await marketHub?.stop();
        } catch {}
        try {
          await connectMarketData();
        } catch (err2: any) {
          log(`❌ Failed to restart market hub: ${err2?.message || err2}`, 'error');
        }
      }
    }
  }, 30_000); // Check every 30s

  // Subscribe to contract
  const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
  if (!metadata) {
    log(`Failed to resolve TopstepX metadata for ${SYMBOL}`, 'error');
    return;
  }

  contractId = metadata.id;

  // Initialize execution manager with live trading capability
  const LIVE_TRADING = process.env.LIVE_TRADING === 'true';  // Set to 'true' to enable real order submission
  executionManager = createExecutionManager(SYMBOL, contractId, CONTRACTS, LIVE_TRADING, {
    tickSize: metadata.tickSize || 0.25,
    preferredAccountId: ACCOUNT_ID > 0 ? ACCOUNT_ID : undefined,
    enableNativeBrackets: process.env.TOPSTEPX_ENABLE_NATIVE_BRACKETS === 'true',
  });
  log(`⚙️ Execution manager initialized for ${SYMBOL} (${LIVE_TRADING ? 'LIVE TRADING' : 'SIM MODE'})`, LIVE_TRADING ? 'warning' : 'success');

  // Initialize trading account (selects account with balance < $40k)
  const accountInitialized = await executionManager.initializeTradingAccount();
  if (!accountInitialized) {
    log('⚠️ Failed to initialize trading account. Orders will not be submitted.', 'error');
  }

  if (accountInitialized) {
    const rehydrated = await executionManager.rehydrateActivePosition();
    if (rehydrated) {
      currentPosition = rehydrated;
      log(
        `♻️ Rehydrated existing ${rehydrated.side.toUpperCase()} position (${rehydrated.contracts} contracts) @ ${rehydrated.entryPrice.toFixed(2)} | SL ${rehydrated.stopLoss.toFixed(2)} | TP ${rehydrated.target.toFixed(2)}`,
        'warning'
      );
    }

    // Connect user hub for real-time position updates (replaces REST polling)
    // SKIP in hybrid mode - User Hub causes "multiple sessions" kicks
    // In hybrid mode, we use REST polling for position data instead
    if (DATA_MODE !== 'hybrid') {
      // Add delay to avoid overwhelming TopstepX server with simultaneous connections
      await new Promise(resolve => setTimeout(resolve, 2000));
      await connectUserHub(executionManager.getAccountId());
    } else {
      log('📡 [Hybrid] Skipping User Hub (using REST for positions)', 'info');
    }
  }

  // Subscribe to market data
  log(`📡 Attempting subscription (hub state: ${marketHub?.state}, marketHubState: ${marketHubState})`, 'info');
  subscribeToMarketStreams();
}

// Connect to User Hub for real-time position updates (replaces REST polling)
async function connectUserHub(accountId: number) {
  if (!accountId) {
    log('⚠️ Cannot connect user hub: no account ID', 'warning');
    return;
  }

  // Prevent concurrent reconnection attempts
  if (userHubReconnecting) {
    log('⚠️ User hub reconnection already in progress, skipping', 'warning');
    return;
  }

  // Stop existing hub if present
  if (userHub) {
    try {
      await userHub.stop();
    } catch (e) {
      // Ignore stop errors
    }
    userHub = null;
  }

  userHubReconnecting = true;
  userHubAccountId = accountId;

  try {
    const token = await authenticate();
    log(`🔌 Connecting to user hub for account ${accountId}...`, 'info');

    // DISABLED: withAutomaticReconnect for user hub (same reason as market hub)
    // SignalR auto-reconnect reuses the original connection URL which has the old token
    // This causes immediate disconnect after reconnect (server rejects expired token)
    // Instead, we handle ALL reconnection in onclose which creates a fresh hub with fresh token

    userHub = new HubConnectionBuilder()
      .withUrl(`https://rtc.topstepx.com/hubs/user?access_token=${token}`, {
        transport: HttpTransportType.WebSockets,
        skipNegotiation: true,
      })
      // NO withAutomaticReconnect - onclose handler will create fresh connection with fresh token
      .withServerTimeout(120000) // 2 minutes - generous to survive any Ollama/system load
      .configureLogging(LogLevel.Warning)
      .build();

    // Handle logout events from broker (session kicked, logged in elsewhere, etc.)
    userHub.on('gatewaylogout', (data: any) => {
      log(`⚠️ [UserHub] Received gatewaylogout: ${JSON.stringify(data)}`, 'warning');
      log(`💡 This usually means you're logged in elsewhere (web platform, another instance). Check and close other sessions.`, 'warning');
    });

    // Handle position updates from WebSocket
    userHub.on('GatewayUserPosition', (data: any) => {
      const netQty = Number(data.netQty ?? data.position ?? data.size ?? data.qty ?? 0);
      const avgPrice = Number(data.avgPrice ?? data.price ?? data.entryPrice ?? 0);
      const posContractId = data.contractId ?? data.instrumentId ?? undefined;

      // Only process positions for our contract
      if (posContractId && posContractId !== contractId) {
        return;
      }

      if (executionManager) {
        if (netQty !== 0) {
          executionManager.updatePositionFromWebSocket({
            netQty,
            avgPrice,
            contractId: posContractId,
          });
        } else {
          executionManager.updatePositionFromWebSocket(null);
        }
      }
    });

    // Handle order updates from WebSocket (fills, cancels, etc.)
    userHub.on('GatewayUserOrder', (data: any) => {
      const orderContractId = data.contractId ?? data.instrumentId;

      // Only process orders for our contract
      if (orderContractId && orderContractId !== contractId) {
        return;
      }

      log(`📋 [WS Order] ${data.side === 1 ? 'BUY' : 'SELL'} ${data.type === 2 ? 'MARKET' : data.type === 3 ? 'LIMIT' : 'STOP'} - Status: ${data.status} - ID: ${data.id}`, 'info');

      if (executionManager) {
        executionManager.updateOrderFromWebSocket({
          id: data.id,
          contractId: orderContractId,
          side: data.side, // 1=buy, 2=sell
          type: data.type, // 2=market, 3=limit, 4=stop
          status: data.status, // 1=working, 2=filled, 3=cancelled, etc.
          filledQty: data.filledQty ?? data.fillQty ?? 0,
          avgFillPrice: data.avgFillPrice ?? data.fillPrice ?? 0,
          price: data.price,
          stopPrice: data.stopPrice,
        });
      }
    });

    // Handle account updates (optional, for balance tracking)
    userHub.on('GatewayUserAccount', (data: any) => {
      if (data?.balance !== undefined) {
        accountBalance = Number(data.balance);
      }
    });

    const subscribeUserData = async (): Promise<boolean> => {
      if (!userHub) return false;
      try {
        await userHub.invoke('SubscribeAccounts');
        await userHub.invoke('SubscribePositions', accountId);
        await userHub.invoke('SubscribeOrders', accountId);
        log(`✅ User hub subscribed for account ${accountId}`, 'success');
        return true;
      } catch (err: any) {
        log(`❌ User hub subscription failed: ${err?.message || err}`, 'error');
        return false;
      }
    };

    userHub.onreconnected(() => {
      log('✅ User hub reconnected', 'success');
      subscribeUserData();
    });

    userHub.onclose(async (error) => {
      log(`❌ User hub disconnected: ${error?.message || 'unknown'}`, 'warning');

      // Don't schedule another reconnect if we're already reconnecting
      // This prevents duplicate timers from stacking up when onclose fires
      // during a failed reconnection attempt
      if (userHubReconnecting) {
        log(`⚠️ User hub onclose fired during reconnection - not scheduling duplicate`, 'info');
        return;
      }

      // Cancel any previous pending timer before scheduling a new one
      if (userHubReconnectTimer) {
        clearTimeout(userHubReconnectTimer);
        userHubReconnectTimer = null;
      }

      // Exponential backoff: 10s, 20s, 40s, 80s, max 120s
      userHubReconnectAttempts++;
      const backoffDelay = Math.min(10000 * Math.pow(2, userHubReconnectAttempts - 1), 120000);
      log(`⏳ Will attempt user hub reconnection in ${Math.round(backoffDelay/1000)}s (attempt ${userHubReconnectAttempts})`, 'info');

      userHubReconnectTimer = setTimeout(async () => {
        userHubReconnectTimer = null;
        if (userHubAccountId) {
          log('🔄 Attempting manual user hub reconnection...', 'info');
          try {
            await connectUserHub(userHubAccountId);
          } catch (reconnectErr: any) {
            log(`❌ Manual user hub reconnection failed: ${reconnectErr?.message}`, 'error');
          }
        }
      }, backoffDelay);
    });

    await userHub.start();
    log(`✅ User hub connected (state: ${userHub.state})`, 'success');

    // CRITICAL: Wait for connection to stabilize before subscribing
    // TopStepX broker appears to close connections immediately if we subscribe too fast
    // This gives the WebSocket handshake time to fully complete on the broker side
    await new Promise(r => setTimeout(r, 1000));

    // Verify connection is still alive after stabilization wait
    if (userHub.state !== 'Connected') {
      log(`⚠️ User hub state changed to ${userHub.state} during stabilization wait`, 'warning');
      userHubReconnecting = false;

      // Connection died during stabilization - schedule reconnect since onclose was suppressed
      userHubReconnectAttempts++;
      const backoffDelay = Math.min(10000 * Math.pow(2, userHubReconnectAttempts - 1), 120000);
      log(`⏳ Scheduling user hub reconnect after stabilization failure in ${Math.round(backoffDelay/1000)}s (attempt ${userHubReconnectAttempts})`, 'info');

      if (userHubReconnectTimer) clearTimeout(userHubReconnectTimer);
      userHubReconnectTimer = setTimeout(async () => {
        userHubReconnectTimer = null;
        if (userHubAccountId) {
          log('🔄 Attempting user hub reconnection after stabilization failure...', 'info');
          try {
            await connectUserHub(userHubAccountId);
          } catch (reconnectErr: any) {
            log(`❌ User hub reconnection after stabilization failed: ${reconnectErr?.message}`, 'error');
          }
        }
      }, backoffDelay);
      return;
    }

    userHubReconnecting = false; // Release lock on success
    const subscriptionSuccess = await subscribeUserData();
    // Only reset backoff AFTER subscription actually succeeds (not just start)
    // This prevents counter reset when connection briefly connects then drops
    if (subscriptionSuccess) {
      userHubReconnectAttempts = 0;
      log(`✅ User hub fully connected and subscribed - backoff counter reset`, 'success');
    } else {
      log(`⚠️ User hub connected but subscription failed - keeping backoff counter at ${userHubReconnectAttempts}`, 'warning');
    }

    // Enable WebSocket mode in ExecutionManager - disables REST polling
    if (executionManager) {
      executionManager.enableWebSocketMode();
    }

  } catch (error: any) {
    userHubReconnecting = false; // Release lock on failure
    log(`❌ Failed to connect user hub: ${error?.message || error}`, 'error');
  }
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

// Update latest bar with a new trade price/size to keep chart live
function upsertBarFromTrade(price: number, size: number, timestamp: number) {
  if (!Number.isFinite(price) || !Number.isFinite(timestamp)) return;
  const tsDate = new Date(timestamp);
  if (Number.isNaN(tsDate.getTime())) return;
  const currentBucket = new Date(tsDate);
  currentBucket.setSeconds(0, 0); // align to minute for 1-min bars
  const bucketIso = currentBucket.toISOString();
  const lastBar = bars[bars.length - 1];
  if (lastBar && lastBar.timestamp === bucketIso) {
    lastBar.close = price;
    lastBar.high = Math.max(lastBar.high, price);
    lastBar.low = Math.min(lastBar.low, price);
    lastBar.volume = (lastBar.volume || 0) + size;
  } else {
    bars.push({
      timestamp: bucketIso,
      open: price,
      high: price,
      low: price,
      close: price,
      volume: size,
    });
  }
  // keep last ~500 bars
  if (bars.length > 600) {
    bars = bars.slice(-600);
  }
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

    const historicalBars = await fetchTopstepXFuturesBars({
      contractId,
      startTime: startTime.toISOString(),
      endTime: endTime.toISOString(),
      unit: 2,        // Minutes
      unitNumber: 1,  // 1-minute bars
      limit: 500,     // ~23 hours * 12 bars/hour = 276 bars max
    });

    if (historicalBars && historicalBars.length > 0) {
      bars = historicalBars;
      volumeProfile = calculateVolumeProfile(bars);
      marketStructure.state = detectMarketState();
      cvdMinuteBars = [];
      currentCvdBar = null;
      orderFlowData.cvd = 0;
      orderFlowData.cvdHistory = [];
      orderFlowData.bigPrints = [];
      orderFlowData.footprintImbalance = {};
      log(`✅ Loaded ${bars.length} historical 1-min bars from TopstepX`, 'success');
    } else {
      log('No historical bars returned from TopstepX', 'warning');
    }
    await refreshHigherTimeframes(true);
    await refreshRecentVolumeProfiles(true);
    // Initial load of 1-second bars for accurate volume profile
    lastOneSecondBarRefresh = 0;  // Force refresh
    await refreshOneSecondBars();
  } catch (error: any) {
    log(`Failed to load historical data: ${error.message}`, 'error');
  }
}

// ============================================================================
// REST-ONLY POLLING MODE
// Polls REST API for market data and positions instead of WebSocket
// This avoids the "multiple sessions" kick from the broker
// ============================================================================
let restPollingActive = false;
let lastRestPollTime = 0;

async function startRestPolling() {
  if (restPollingActive) {
    log('⚠️ REST polling already active', 'warning');
    return;
  }

  restPollingActive = true;
  log('🔄 Starting REST-only polling mode (no WebSocket)', 'success');
  log(`   Poll interval: ${REST_POLL_INTERVAL_MS}ms`, 'info');

  const projectXRest = createProjectXRest();

  // REST polling loop
  while (restPollingActive) {
    try {
      const pollStart = Date.now();

      // 1. Fetch latest 1-minute bars (last 5 minutes for freshness)
      const endTime = new Date();
      const startTime = new Date(endTime.getTime() - 5 * 60 * 1000); // Last 5 min

      const recentBars = await fetchTopstepXFuturesBars({
        contractId,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString(),
        unit: 2, // Minutes
        unitNumber: 1, // 1-min bars
        limit: 10,
        includePartialBar: true, // CRITICAL: Get live/partial bar for real-time price updates
      });

      if (recentBars && recentBars.length > 0) {
        // Update bars array with fresh data
        const latestBar = recentBars[recentBars.length - 1];
        const existingIndex = bars.findIndex(b => b.timestamp === latestBar.timestamp);

        if (existingIndex >= 0) {
          // Update existing bar
          bars[existingIndex] = latestBar;
        } else {
          // Add new bar
          bars.push(latestBar);
          // Keep only last 500 bars
          if (bars.length > 500) bars.shift();

          log(`📊 [REST] New bar: ${latestBar.timestamp.slice(11, 19)} @ ${latestBar.close.toFixed(2)}`, 'info');
        }

        // Emit to dashboard
        if (io) {
          io.emit('bars', { bars: bars.slice(-100) });
        }
      }

      // 2. Poll position status via REST (if we have execution manager)
      if (executionManager) {
        const accountId = executionManager.getAccountId();
        if (accountId) {
          try {
            // Check open orders to track position state
            const openOrders = await projectXRest.searchOpenOrders({ accountId });
            const orders = openOrders?.orders || [];
            const ourOrders = orders.filter((o: any) => o.contractId === contractId);

            // Update execution manager with order status
            if (ourOrders.length > 0) {
              for (const order of ourOrders) {
                executionManager.updateOrderFromWebSocket({
                  id: order.id,
                  contractId: order.contractId,
                  side: order.side,
                  type: order.type,
                  status: order.status,
                  filledQty: order.filledQty || 0,
                  avgFillPrice: order.avgFillPrice || 0,
                  price: order.limitPrice,
                  stopPrice: order.stopPrice,
                });
              }
            }
          } catch (posErr: any) {
            // Don't spam errors, just log occasionally
            if (Date.now() - lastRestPollTime > 30000) {
              log(`⚠️ [REST] Position poll error: ${posErr.message}`, 'warning');
            }
          }
        }
      }

      lastRestPollTime = Date.now();
      const pollDuration = Date.now() - pollStart;

      // Wait for next poll interval
      const sleepTime = Math.max(100, REST_POLL_INTERVAL_MS - pollDuration);
      await new Promise(r => setTimeout(r, sleepTime));

    } catch (pollErr: any) {
      log(`❌ [REST] Polling error: ${pollErr.message}`, 'error');
      await new Promise(r => setTimeout(r, 5000)); // Wait 5s on error
    }
  }

  log('🛑 REST polling stopped', 'warning');
}

function stopRestPolling() {
  restPollingActive = false;
}

// Main Function
async function main() {
  console.log('\n' + '='.repeat(80));
  console.log('🧠 FABIO AGENT - VOLUME PROFILE & ORDER FLOW TRADING');
  console.log('='.repeat(80));
  console.log(`Symbol: ${SYMBOL}`);
  console.log(`Playbook: ${fabioPlaybook.philosophy}`);
  console.log(`Risk per trade: ${fabioPlaybook.riskRules.riskPerTradePctMin}-${fabioPlaybook.riskRules.riskPerTradePctMax}%`);
  console.log(`Dashboard: http://localhost:${DASHBOARD_PORT}`);
  console.log('='.repeat(80) + '\n');

  try {
    // CRITICAL: Reset Ollama model state on startup
    // This runs `ollama stop MODEL` to clear any stuck generations from previous sessions
    await resetOllamaOnStartup();

    // CRITICAL: Clear any stale Ollama requests from previous session
    // This prevents timeouts from queued requests backing up Ollama
    cancelActiveRequest();
    console.log('🧹 Cleared any stale Ollama requests from previous session');

    // WARMUP: Send a quick test request to Ollama /api/chat to verify model is ready
    console.log('🔥 Warming up Ollama...');
    try {
      const { chat } = await import('./lib/ollamaClient');
      const warmupStart = Date.now();
      await chat([{ role: 'user', content: 'hi' }], { timeoutMs: 60000 }); // Allow 60s for cold start
      const warmupTime = Date.now() - warmupStart;
      console.log(`✅ Ollama warmup complete in ${warmupTime}ms`);
      if (warmupTime > 5000) {
        console.log('⚠️  Warmup took >5s - model was cold loading');
      }
    } catch (warmupError: any) {
      console.warn(`⚠️  Ollama warmup failed (${warmupError.message}) - continuing anyway`);
    }

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

    // Choose data mode based on DATA_MODE setting
    log(`📡 Data mode: ${DATA_MODE.toUpperCase()}`, 'info');

    if (DATA_MODE === 'rest') {
      // REST-ONLY: No WebSocket at all
      log('📡 REST-ONLY MODE - No WebSocket connections', 'success');

      // Initialize execution manager for REST mode (normally done in connectMarketData)
      const LIVE_TRADING = process.env.LIVE_TRADING === 'true';
      executionManager = createExecutionManager(SYMBOL, contractId, CONTRACTS, LIVE_TRADING, {
        tickSize: metadata.tickSize || 0.25,
        preferredAccountId: ACCOUNT_ID > 0 ? ACCOUNT_ID : undefined,
        enableNativeBrackets: process.env.TOPSTEPX_ENABLE_NATIVE_BRACKETS === 'true',
      });
      log(`⚙️ Execution manager initialized for ${SYMBOL} (${LIVE_TRADING ? 'LIVE TRADING' : 'SIM MODE'})`, LIVE_TRADING ? 'warning' : 'success');

      // Initialize trading account
      const accountInitialized = await executionManager.initializeTradingAccount();
      if (!accountInitialized) {
        log('⚠️ Failed to initialize trading account. Orders will not be submitted.', 'error');
      }

      // Rehydrate any existing position
      if (accountInitialized) {
        const rehydrated = await executionManager.rehydrateActivePosition();
        if (rehydrated) {
          currentPosition = rehydrated;
          log(
            `♻️ Rehydrated existing ${rehydrated.side.toUpperCase()} position (${rehydrated.contracts} contracts) @ ${rehydrated.entryPrice.toFixed(2)} | SL ${rehydrated.stopLoss.toFixed(2)} | TP ${rehydrated.target.toFixed(2)}`,
            'warning'
          );
        }
      }

      startRestPolling().catch((err) => {
        log(`❌ REST polling crashed: ${err.message}`, 'error');
      });
    } else if (DATA_MODE === 'hybrid') {
      // HYBRID: Market WebSocket for L2/trades + REST for positions
      log('📡 HYBRID MODE - Market WebSocket for L2 + REST for positions', 'success');
      // Connect Market WebSocket for L2 and trade data
      await connectMarketData();
      // Start REST polling for position monitoring (no User Hub)
      startRestPolling().catch((err) => {
        log(`❌ REST polling crashed: ${err.message}`, 'error');
      });
      log('   ✅ Market WebSocket: L2, trades, order flow', 'info');
      log('   ✅ REST polling: positions, orders (no User Hub kicks)', 'info');
    } else {
      // WEBSOCKET: Full WebSocket mode (original behavior)
      log('📡 FULL WEBSOCKET MODE - May get kicked if multiple sessions', 'warning');
      await connectMarketData();
    }

    // Create order manager
    const { orderManager: om } = await createProjectXRest();
    orderManager = om;

    log('🚀 FABIO AGENT FULLY INITIALIZED', 'success');
    log('Waiting for market conditions...', 'info');

    // Start the continuous Qwen analysis loop (fire and forget - runs in background)
    // This loop sends requests sequentially: when one completes, immediately starts the next
    // No timer, no overlap, no gaps - prevents Ollama queue backup
    runContinuousQwenAnalysis().catch((err) => {
      log(`❌ [Qwen] Continuous analysis loop crashed: ${err.message}`, 'error');
    });

  } catch (error: any) {
    log(`Failed to initialize: ${error.message}`, 'error');
    process.exit(1);
  }
}

// Graceful Shutdown
process.on('SIGINT', async () => {
  log('Shutting down gracefully...', 'warning');

  // Stop the continuous Qwen analysis loop
  qwenAnalysisLoopRunning = false;

  // Stop REST polling if active
  stopRestPolling();

  // Stop WebSocket hubs if connected
  if (marketHub) await marketHub.stop();
  if (userHub) await userHub.stop();
  if (server) server.close();

  process.exit(0);
});

// Start the agent
main().catch(console.error);
