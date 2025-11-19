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
} from './lib/fabioOpenAIIntegration';
import { analyzeFuturesMarket } from './lib/openaiTradingAgent';
import { createExecutionManager } from './lib/executionManager';

// Configuration
const SYMBOL = process.env.TOPSTEPX_SYMBOL || 'NQZ5';
const ACCOUNT_ID = parseInt(process.env.TOPSTEPX_ACCOUNT_ID || '0');
const DASHBOARD_PORT = 3337;
const CONTRACTS = 1;

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
  absorption: { buy: number; sell: number }; // Actual absorption strength 0-1
  exhaustion: { buy: number; sell: number }; // Actual exhaustion strength 0-1
  cvdHistory: Array<{ timestamp: number; cvd: number; delta: number }>; // Track CVD momentum
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

// Position Management
let currentPosition: any = null;
let accountBalance = 50000;

// OpenAI + Execution Integration
let executionManager: ExecutionManager | null = null;
let realizedPnL = 0; // Track realized P&L from closed positions

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

  // Buy absorption: High buy volume + price holding (not falling despite sells)
  let buyAbsorptionStrength = 0;
  if (buyRatio > 0.4) { // Need at least 40% buy volume
    const buyDominance = (buyRatio - 0.5) * 2; // Scale 0.5-1.0 to 0-1
    buyAbsorptionStrength = Math.max(0, buyDominance) * priceStability;
  }

  // Sell absorption: High sell volume + price holding (not rising despite buys)
  let sellAbsorptionStrength = 0;
  if (sellRatio > 0.4) { // Need at least 40% sell volume
    const sellDominance = (sellRatio - 0.5) * 2; // Scale 0.5-1.0 to 0-1
    sellAbsorptionStrength = Math.max(0, sellDominance) * priceStability;
  }

  return {
    buy: buyAbsorptionStrength,
    sell: sellAbsorptionStrength,
  };
}

// Calculate Exhaustion: Detects when momentum is dying out
function calculateExhaustion(): { buy: number; sell: number } {
  const minHistoryWindow = 10; // Minimum data points needed

  if (orderFlowData.cvdHistory.length < minHistoryWindow) {
    return { buy: 0, sell: 0 };
  }

  const recent = orderFlowData.cvdHistory.slice(-Math.min(30, orderFlowData.cvdHistory.length));

  // Calculate delta momentum (rate of change in delta)
  const recentDeltas = recent.map(h => h.delta);
  const buyDeltas = recentDeltas.filter(d => d > 0);
  const sellDeltas = recentDeltas.filter(d => d < 0);

  // Calculate momentum decline
  let buyExhaustion = 0;
  let sellExhaustion = 0;

  if (buyDeltas.length >= 5) {
    // Check if buy deltas are decreasing (buy momentum dying)
    const firstHalf = buyDeltas.slice(0, Math.floor(buyDeltas.length / 2));
    const secondHalf = buyDeltas.slice(Math.floor(buyDeltas.length / 2));

    const firstHalfAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondHalfAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    if (firstHalfAvg > 0 && secondHalfAvg < firstHalfAvg) {
      // Buy momentum is declining - calculate exhaustion as percentage decline
      const declineRatio = (firstHalfAvg - secondHalfAvg) / firstHalfAvg;
      buyExhaustion = Math.min(1, declineRatio * 1.5); // Amplify for better visibility
    }
  }

  if (sellDeltas.length >= 5) {
    // Check if sell deltas are decreasing in magnitude (sell momentum dying)
    const firstHalf = sellDeltas.slice(0, Math.floor(sellDeltas.length / 2));
    const secondHalf = sellDeltas.slice(Math.floor(sellDeltas.length / 2));

    const firstHalfAvg = Math.abs(firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length);
    const secondHalfAvg = Math.abs(secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length);

    if (firstHalfAvg > 0 && secondHalfAvg < firstHalfAvg) {
      // Sell momentum is declining - calculate exhaustion as percentage decline
      const declineRatio = (firstHalfAvg - secondHalfAvg) / firstHalfAvg;
      sellExhaustion = Math.min(1, declineRatio * 1.5); // Amplify for better visibility
    }
  }

  return {
    buy: buyExhaustion,
    sell: sellExhaustion,
  };
}

// Emit LLM Decision
function emitLLMDecision(analysis: any) {
  const decision = {
    timestamp: new Date().toISOString(),
    marketState: analysis.marketState,
    model: analysis.model,
    location: analysis.location,
    orderFlow: analysis.orderFlow,
    decision: analysis.decision,
    reasoning: analysis.reasoning,
    confidence: analysis.confidence,
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
  if (bars.length < 20) return;

  // Update volume profile (existing)
  volumeProfile = calculateVolumeProfile(bars.slice(-50));

  // Update market structure (existing)
  marketStructure.state = detectMarketState();

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
    realizedPnL
  );

  // ========== NEW: Get OpenAI decision ==========
  let openaiDecision = null;
  try {
    openaiDecision = await analyzeFuturesMarket(marketData);
    if (openaiDecision) {
      log(`🤖 OpenAI Decision: ${openaiDecision.decision} @ ${openaiDecision.entryPrice} (Confidence: ${openaiDecision.confidence}%)`, 'info');
    }
  } catch (error) {
    console.error('[OpenAI] Analysis failed:', error);
    // Continue with rule-based only if OpenAI fails
  }

  // Get rule-based decision (existing)
  const ruleBasedDecision = makeDecision();

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
    }
  } else if (ruleBasedDecision.entry.side && ruleBasedDecision.entry.confidence >= 70 && !currentPosition) {
    // Fallback to rule-based if no OpenAI decision
    log(`📊 Rule-based Decision: ${ruleBasedDecision.entry.side.toUpperCase()} (Confidence: ${ruleBasedDecision.entry.confidence}%)`, 'info');
    executeEntry(ruleBasedDecision);
  }

  // ========== NEW: Update position and check exits ==========
  if (executionManager && executionManager.getActivePosition()) {
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
    } else {
      // Update position reference
      currentPosition = executionManager.getActivePosition();
    }
  }

  // Emit decision (existing, can be enhanced)
  if (openaiDecision && openaiDecision.confidence >= 70) {
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
      confidence: openaiDecision.confidence,
    });
  } else if (ruleBasedDecision.entry.confidence > 60) {
    emitLLMDecision({
      marketState: ruleBasedDecision.marketState,
      model: ruleBasedDecision.model,
      location: ruleBasedDecision.location,
      orderFlow: {
        cvd: orderFlowData.cvd,
        bigPrints: orderFlowData.bigPrints.length,
        confirmed: ruleBasedDecision.orderFlowConfirmation,
      },
      decision: ruleBasedDecision.entry.side || 'hold',
      reasoning: ruleBasedDecision.entry.reason,
      confidence: ruleBasedDecision.entry.confidence,
    });
  }

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
    } : null,
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
    big_prints: orderFlowData.bigPrints.slice(-10),
  };
  log(`📡 Broadcasting CVD: ${orderFlowData.cvd.toFixed(2)}, trend=${cvdData.cvd_trend}, OHLC=[O:${currentCvdBar?.open.toFixed(1)}, H:${currentCvdBar?.high.toFixed(1)}, L:${currentCvdBar?.low.toFixed(1)}, C:${currentCvdBar?.close.toFixed(1)}], buy_abs=${(cvdData.buy_absorption * 100).toFixed(1)}%, sell_abs=${(cvdData.sell_absorption * 100).toFixed(1)}%`, 'info');
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
    res.sendFile('/Users/coreycosta/trading-agent/public/fabio-topstep-dashboard.html');
  });

  // Socket.IO connection
  io.on('connection', (socket) => {
    log('Dashboard client connected', 'info');
    broadcastDashboardUpdate();

    socket.on('request_chart_history', () => {
      socket.emit('chart_history', bars.slice(-100));
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
  marketHub.on('gatewayquote', (contractId: string, quotes: any) => {
    // Process Level 2 data
    if (quotes.bids && quotes.asks) {
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

  marketHub.on('gatewaytrade', (contractId: string, trades: any[]) => {
    log(`📊 Received ${trades.length} trade(s) for ${contractId}`, 'info');
    trades.forEach((trade, idx) => {
      const price = parseFloat(trade.price);
      const size = parseFloat(trade.volume); // TopStepX uses "volume" not "size"
      const type = trade.type; // 0 or 1
      const side = type === 1 ? 'Buy' : 'Sell'; // type: 1=Buy, 0=Sell

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
      const currentFiveMinute = Math.floor(now.getTime() / (5 * 60 * 1000)) * (5 * 60 * 1000);
      const currentFiveMinuteISO = new Date(currentFiveMinute).toISOString();
      let currentBar = bars[bars.length - 1];

      if (!currentBar || new Date(currentBar.timestamp).getTime() < currentFiveMinute) {
        // New bar - use 5-minute boundary timestamp
        const newBar: TopstepXFuturesBar = {
          timestamp: currentFiveMinuteISO,
          open: price,
          high: price,
          low: price,
          close: price,
          volume: size,
        };
        bars.push(newBar);
        log(`📊 NEW 5-MIN CANDLE: ${currentFiveMinuteISO}, O=${price}, bars=${bars.length}`, 'info');
        // Keep last 1000 bars
        if (bars.length > 1000) bars.shift();
      } else {
        // Update existing bar
        currentBar.high = Math.max(currentBar.high, price);
        currentBar.low = Math.min(currentBar.low, price);
        currentBar.close = price;
        currentBar.volume = (currentBar.volume || 0) + size;
      }

      // Update CVD
      const delta = trade.side === 'Buy' ? size : -size;
      const previousCvd = orderFlowData.cvd;
      orderFlowData.cvd += delta;

      // Track CVD OHLC for current 5-minute bar
      if (!currentCvdBar || currentCvdBar.timestamp !== currentFiveMinuteISO) {
        // New CVD bar - initialize with current CVD value
        currentCvdBar = {
          timestamp: currentFiveMinuteISO,
          open: orderFlowData.cvd,
          high: orderFlowData.cvd,
          low: orderFlowData.cvd,
          close: orderFlowData.cvd,
        };
        log(`📊 NEW CVD BAR: ${currentFiveMinuteISO}, CVD=${orderFlowData.cvd.toFixed(2)}`, 'info');
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
      if (trade.side === 'Buy') {
        orderFlowData.volumeAtPrice[priceLevel].buy += size;
      } else {
        orderFlowData.volumeAtPrice[priceLevel].sell += size;
      }
      orderFlowData.volumeAtPrice[priceLevel].timestamp = Date.now();

      // Check for big prints
      if (size >= 10) { // 10+ contracts is big
        orderFlowData.bigPrints.push({
          price,
          size,
          side: trade.side === 'Buy' ? 'buy' : 'sell',
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

  // Initialize execution manager for OpenAI integration
  executionManager = createExecutionManager(SYMBOL, CONTRACTS);
  log(`⚙️ Execution manager initialized for ${SYMBOL}`, 'success');

  // Subscribe to contract
  const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
  if (!metadata) {
    log(`Failed to resolve TopstepX metadata for ${SYMBOL}`, 'error');
    return;
  }

  contractId = metadata.id;

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
function getTradingDayStart(): Date {
  const now = new Date();

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
      unitNumber: 5,  // 5-minute bars
      limit: 500,     // ~23 hours * 12 bars/hour = 276 bars max
    });

    if (historicalBars && historicalBars.length > 0) {
      bars = historicalBars;
      volumeProfile = calculateVolumeProfile(bars);
      marketStructure.state = detectMarketState();
      log(`✅ Loaded ${bars.length} historical 5-min bars from TopstepX`, 'success');
    } else {
      log('No historical bars returned from TopstepX', 'warning');
    }
  } catch (error: any) {
    log(`Failed to load historical data: ${error.message}`, 'error');
  }
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
    // Initialize components
    await initDashboard();

    // Resolve contract ID FIRST before loading historical data
    const metadata = await fetchTopstepXFuturesMetadata(SYMBOL);
    if (!metadata) {
      throw new Error(`Failed to resolve contract for symbol ${SYMBOL}`);
    }
    contractId = metadata.id;
    log(`Resolved ${SYMBOL} to contractId=${contractId}`, 'info');

    // Load historical data for chart and volume profile initialization
    try {
      log('Loading historical data from TopStepX...', 'info');
      const historicalBars = await fetchTopstepXFuturesBars({
        contractId,
        startTime: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        endTime: new Date().toISOString(),
        unit: 2, // Minutes
        unitNumber: 5, // 5-minute bars
        limit: 100,
        live: false,
      });

      if (historicalBars.length > 0) {
        bars.push(...historicalBars);
        log(`✅ Loaded ${historicalBars.length} historical bars from TopStepX`, 'success');
      } else {
        log('⚠️  No historical data returned - will build from live market data', 'warning');
      }
    } catch (error: any) {
      log(`⚠️  Failed to load historical data: ${error.message} - will build from live market data`, 'warning');
    }

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
