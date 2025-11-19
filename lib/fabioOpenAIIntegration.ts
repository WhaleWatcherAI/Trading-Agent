/**
 * Fabio Agent + OpenAI Integration
 * Bridges existing Fabio calculations with OpenAI decision making
 * and execution system with self-learning database
 */

import { analyzeFuturesMarket, FuturesMarketData, OpenAITradingDecision } from './openaiTradingAgent';
import { ExecutionManager } from './executionManager';
import { tradingDB } from './tradingDatabase';
import { MarketState } from './fabioPlaybook';

// Type imports from live-fabio-agent-playbook
export interface TopstepXFuturesBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface VolumeProfile {
  nodes: any[];
  poc: number;
  vah: number;
  val: number;
  lvns: number[];
}

export interface OrderFlowData {
  bigPrints: Array<{ price: number; size: number; side: 'buy' | 'sell'; timestamp: number }>;
  cvd: number;
  footprintImbalance: { [price: number]: number };
  absorption: { buy: number; sell: number };
  exhaustion: { buy: number; sell: number };
  cvdHistory: Array<{ timestamp: number; cvd: number; delta: number }>;
  volumeAtPrice: { [price: number]: { buy: number; sell: number; timestamp: number } };
}

export interface MarketStructure {
  state: MarketState;
  impulseLegs: any[];
  balanceAreas: any[];
  failedBreakouts: any[];
}

export interface CurrentCvdBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

/**
 * Build FuturesMarketData from Fabio agent's existing calculations
 * This is the bridge between Fabio's analysis and OpenAI's decision making
 */
export function buildFuturesMarketData(
  symbol: string,
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile | null,
  orderFlowData: OrderFlowData,
  marketStructure: MarketStructure,
  currentCvdBar: CurrentCvdBar | null,
  accountBalance: number,
  currentPosition: any | null,
  realizedPnL: number = 0
): FuturesMarketData {
  // Get current price from latest bar
  const currentPrice = bars.length > 0 ? bars[bars.length - 1].close : 0;

  // Get last 5 candles (25 minutes for 5-min bars)
  const recentCandles = bars.slice(-5);

  // Calculate CVD trend from history
  let cvdTrend: 'up' | 'down' | 'neutral' = 'neutral';
  if (orderFlowData.cvd > 100) {
    cvdTrend = 'up';
  } else if (orderFlowData.cvd < -100) {
    cvdTrend = 'down';
  }

  // Get buy/sell control from CVD
  const totalCVD = Math.abs(orderFlowData.cvd);
  const buyersControl = totalCVD > 0 ? (orderFlowData.cvd + totalCVD) / (2 * totalCVD) : 0.5;
  const sellersControl = 1 - buyersControl;

  // Determine if order flow is confirmed (alignment of multiple signals)
  const orderFlowConfirmed =
    orderFlowData.bigPrints.length > 0 &&
    Math.abs(orderFlowData.cvd) > 50 &&
    (orderFlowData.absorption.buy > 0.5 || orderFlowData.absorption.sell > 0.5);

  // Build the market data object
  const marketData: FuturesMarketData = {
    symbol,
    timestamp: new Date().toISOString(),
    currentPrice,

    // Price candles (last 5 for 25-minute window)
    candles: recentCandles,

    // CVD data with OHLC candlestick
    cvd: {
      value: orderFlowData.cvd,
      trend: cvdTrend,
      ohlc: currentCvdBar || {
        timestamp: new Date().toISOString(),
        open: 0,
        high: 0,
        low: 0,
        close: 0,
      },
    },

    // Order flow metrics (absorption/exhaustion)
    orderFlow: {
      buyAbsorption: orderFlowData.absorption.buy,
      sellAbsorption: orderFlowData.absorption.sell,
      buyExhaustion: orderFlowData.exhaustion.buy,
      sellExhaustion: orderFlowData.exhaustion.sell,
      bigPrints: orderFlowData.bigPrints.slice(-10), // Last 10 big prints
    },

    // Volume profile structure
    volumeProfile: volumeProfile || {
      poc: 0,
      vah: 0,
      val: 0,
      lvns: [],
      sessionHigh: currentPrice,
      sessionLow: currentPrice,
    },

    // Market state detection
    marketState: {
      state: marketStructure.state,
      buyersControl,
      sellersControl,
    },

    // Order flow confirmation (all 3 layers aligned)
    orderFlowConfirmed,

    // Account information
    account: {
      balance: accountBalance,
      position: currentPosition ? 1 : 0,
      unrealizedPnL: currentPosition?.unrealizedPnL || 0,
      realizedPnL,
    },
  };

  return marketData;
}

/**
 * Process OpenAI decision and execute if criteria met
 */
export async function processOpenAIDecision(
  openaiDecision: OpenAITradingDecision | null,
  executionManager: ExecutionManager,
  currentPrice: number,
  symbol: string,
  orderFlowData: OrderFlowData,
  volumeProfile: VolumeProfile | null,
  marketStructure: MarketStructure
): Promise<{ executed: boolean; decisionId?: string }> {
  if (!openaiDecision) {
    return { executed: false };
  }

  // Only execute if decision is BUY/SELL and confidence is high
  if (openaiDecision.decision === 'HOLD') {
    console.log('[OpenAI] HOLD - Not executing');
    return { executed: false };
  }

  // Minimum confidence threshold
  if (openaiDecision.confidence < 70) {
    console.log(`[OpenAI] Confidence ${openaiDecision.confidence}% below 70% threshold - Not executing`);
    return { executed: false };
  }

  // Check if already in position (one position at a time)
  const activePosition = executionManager.getActivePosition();
  if (activePosition) {
    console.log('[OpenAI] Already in position - Not executing new entry');
    return { executed: false };
  }

  // Execute the decision
  const order = await executionManager.executeDecision(openaiDecision, currentPrice);

  if (!order) {
    return { executed: false };
  }

  // Get the active position to update with additional context
  const position = executionManager.getActivePosition();
  if (position) {
    position.stopLoss = openaiDecision.stopLoss || currentPrice - 30;
    position.target = openaiDecision.target || currentPrice + 30;

    // Record in database with full Fabio context
    const decision = tradingDB.recordDecision({
      symbol,
      marketState: openaiDecision.marketState,
      location: openaiDecision.location,
      setupModel: openaiDecision.setupModel,
      decision: openaiDecision.decision as 'BUY' | 'SELL' | 'HOLD',
      confidence: openaiDecision.confidence,
      entryPrice: currentPrice,
      stopLoss: openaiDecision.stopLoss || currentPrice - 30,
      target: openaiDecision.target || currentPrice + 30,
      riskPercent: openaiDecision.riskPercent,
      source: 'openai',
      reasoning: openaiDecision.reasoning,
      cvd: orderFlowData.cvd,
      cvdTrend: openaiDecision.decision === 'BUY' ? 'up' : 'down',
      currentPrice,
      buyAbsorption: orderFlowData.absorption.buy,
      sellAbsorption: orderFlowData.absorption.sell,
    });

    console.log(
      `[OpenAI] ✅ Executed ${openaiDecision.decision} @ ${currentPrice} | Entry: ${openaiDecision.entryPrice} | SL: ${openaiDecision.stopLoss} | TP: ${openaiDecision.target} | Confidence: ${openaiDecision.confidence}%`
    );

    return { executed: true, decisionId: decision.id };
  }

  return { executed: false };
}

/**
 * Update active position and check for exits
 */
export async function updatePositionAndCheckExits(
  executionManager: ExecutionManager,
  currentPrice: number,
  bars: TopstepXFuturesBar[]
): Promise<{ exited: boolean; closedDecisionId?: string; reason?: string }> {
  const activePosition = executionManager.getActivePosition();

  if (!activePosition) {
    return { exited: false };
  }

  // Update position with current price
  executionManager.updatePositionPrice(activePosition.decisionId, currentPrice);

  // Check for exit conditions
  const closedDecisionId = await executionManager.checkExits(currentPrice);

  if (closedDecisionId) {
    const outcome = tradingDB.getOutcome(closedDecisionId);
    console.log(
      `[Position] ✅ Closed: ${outcome?.reason} | P&L: ${outcome?.profitLoss > 0 ? '+' : ''}$${outcome?.profitLoss.toFixed(2)} (${outcome?.profitLossPercent.toFixed(2)}%)`
    );
    return { exited: true, closedDecisionId, reason: outcome?.reason };
  }

  return { exited: false };
}

/**
 * Get trading statistics for learning
 */
export function getTradeStats(symbol: string) {
  return tradingDB.calculateStats(symbol);
}

/**
 * Log statistics to console for monitoring
 */
export function logTradeStats(symbol: string) {
  const stats = tradingDB.calculateStats(symbol);

  if (stats.totalOutcomes === 0) {
    console.log(`[Stats] No completed trades yet for ${symbol}`);
    return;
  }

  console.log(`
    📊 Trading Statistics for ${symbol}:
    ├─ Total Decisions: ${stats.totalDecisions}
    ├─ Filled Orders: ${stats.totalFilled}
    ├─ Completed Trades: ${stats.totalOutcomes}
    ├─ Win Rate: ${stats.winRate.toFixed(1)}%
    ├─ Avg Win: $${stats.avgWin.toFixed(2)}
    ├─ Avg Loss: $${stats.avgLoss.toFixed(2)}
    ├─ Profit Factor: ${stats.profitFactor.toFixed(2)}
    │
    ├─ By Source:
    │  ├─ OpenAI: ${stats.bySource['openai']?.count || 0} trades (${((stats.bySource['openai']?.wins || 0) / (stats.bySource['openai']?.count || 1) * 100).toFixed(1)}% win rate)
    │  └─ Rule-based: ${stats.bySource['rule_based']?.count || 0} trades (${((stats.bySource['rule_based']?.wins || 0) / (stats.bySource['rule_based']?.count || 1) * 100).toFixed(1)}% win rate)
    │
    └─ By Setup Model:
       ├─ Trend Continuation: ${stats.bySetupModel['trend_continuation']?.count || 0} trades (${((stats.bySetupModel['trend_continuation']?.wins || 0) / (stats.bySetupModel['trend_continuation']?.count || 1) * 100).toFixed(1)}% win rate)
       └─ Mean Reversion: ${stats.bySetupModel['mean_reversion']?.count || 0} trades (${((stats.bySetupModel['mean_reversion']?.wins || 0) / (stats.bySetupModel['mean_reversion']?.count || 1) * 100).toFixed(1)}% win rate)
  `);
}

/**
 * Export all trading data for external analysis
 */
export function exportTradingData(symbol: string) {
  return {
    symbol,
    timestamp: new Date().toISOString(),
    data: tradingDB.exportData(),
  };
}

/**
 * Get recent high-confidence decisions for analysis
 */
export function getHighConfidenceDecisions(symbol: string, minConfidence: number = 75) {
  const decisions = tradingDB.getDecisionsBySymbol(symbol);
  return decisions
    .filter((d) => d.confidence >= minConfidence)
    .map((d) => ({
      id: d.id,
      timestamp: d.timestamp,
      decision: d.decision,
      confidence: d.confidence,
      outcome: tradingDB.getOutcome(d.id),
    }));
}

/**
 * Analyze which confidence levels actually work
 */
export function analyzeConfidenceCalibration(symbol: string) {
  const decisions = tradingDB.getDecisionsBySymbol(symbol);
  const groups = tradingDB.getDecisionsByConfidence(symbol);

  const analysis: { [key: string]: { count: number; wins: number; winRate: number } } = {};

  Object.entries(groups).forEach(([group, decisions]) => {
    const outcomes = decisions.map((d) => tradingDB.getOutcome(d.id)).filter((o) => !!o);
    const wins = outcomes.filter((o) => o?.profitLoss! > 0).length;

    analysis[group] = {
      count: decisions.length,
      wins,
      winRate: decisions.length > 0 ? (wins / decisions.length) * 100 : 0,
    };
  });

  console.log(`
    📈 Confidence Calibration Analysis for ${symbol}:
    ├─ Very High (80%+): ${analysis['very_high']?.count || 0} trades, ${analysis['very_high']?.winRate.toFixed(1) || 0}% win rate
    ├─ High (60-79%): ${analysis['high']?.count || 0} trades, ${analysis['high']?.winRate.toFixed(1) || 0}% win rate
    ├─ Medium (40-59%): ${analysis['medium']?.count || 0} trades, ${analysis['medium']?.winRate.toFixed(1) || 0}% win rate
    └─ Low (<40%): ${analysis['low']?.count || 0} trades, ${analysis['low']?.winRate.toFixed(1) || 0}% win rate
  `);

  return analysis;
}

export { ExecutionManager };
