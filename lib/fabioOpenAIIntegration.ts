/**
 * Fabio Agent + OpenAI Integration
 * Bridges existing Fabio calculations with OpenAI decision making
 * and execution system with self-learning database
 */

import { analyzeFuturesMarket, FuturesMarketData, OpenAITradingDecision } from './openaiTradingAgent';
import { ExecutionManager } from './executionManager';
import { tradingDB } from './tradingDatabase';
import { MarketState } from './fabioPlaybook';
import {
  POCCrossTracker,
  MarketStatsCalculator,
  PerformanceTracker,
  HistoricalNotesManager,
} from './enhancedFeatures';

// Global enhanced trackers for self-learning system
const pocCrossTracker = new POCCrossTracker();
const marketStatsCalc = new MarketStatsCalculator();
const performanceTracker = new PerformanceTracker();
const notesManager = new HistoricalNotesManager();

function isSelfLearningEnabled(): boolean {
  const flag = process.env.SELF_LEARNING_DISABLED?.toLowerCase();
  return flag !== 'true' && flag !== '1' && flag !== 'yes';
}

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

export interface HigherTimeframeSnapshot {
  timeframe: string;
  candles: Array<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
  }>;
}

export interface SessionVolumeProfileSummary {
  sessionStart: string;
  sessionEnd: string;
  poc: number;
  vah: number;
  val: number;
  lvns: number[];
  sessionHigh: number;
  sessionLow: number;
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
  realizedPnL: number = 0,
  higherTimeframes: HigherTimeframeSnapshot[] = [],
  recentVolumeProfiles: SessionVolumeProfileSummary[] = [],
  cvdCandles: CurrentCvdBar[] = []
): FuturesMarketData {
  const selfLearning = isSelfLearningEnabled();
  // Get current price from latest bar
  const currentPrice = bars.length > 0 ? bars[bars.length - 1].close : 0;

  // Get last 5 candles (25 minutes for 5-min bars)
  const recentCandles = bars.slice(-20);

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
    Math.abs(orderFlowData.cvd) > 50;

  // Get session high/low for tracking
  const sessionHigh = bars.reduce((max, bar) => Math.max(max, bar.high), 0);
  const sessionLow = bars.reduce((min, bar) => (min === Infinity ? bar.low : Math.min(min, bar.low)), Infinity);

  // Update enhanced trackers
  const poc = volumeProfile?.poc || currentPrice;
  const vah = volumeProfile?.vah || currentPrice;
  const val = volumeProfile?.val || currentPrice;

  // Update POC cross tracking
  const pocCrossStats = pocCrossTracker.update(currentPrice, poc);

  // Update market statistics
  marketStatsCalc.updateSession(sessionHigh, sessionLow);
  marketStatsCalc.updateTimeInValue(currentPrice, vah, val);
  marketStatsCalc.updateCVD(orderFlowData.cvd);

  const marketStats = marketStatsCalc.calculate(currentPrice, poc, vah, val, 0.25);

  // Get performance metrics
  const performance = selfLearning ? performanceTracker.getMetrics() : null;

  // Get historical notes (last 10)
  const historicalNotes = selfLearning ? notesManager.getRecentNotes(10) : [];

  // Build microstructure snapshot from order flow
  const microstructure = buildMicrostructureFromOrderFlow(orderFlowData);

  // Build macrostructure (session/multi-hour context) from bars + profile
  const macrostructure = buildMacrostructureFromBars(
    bars,
    volumeProfile,
    sessionHigh,
    sessionLow,
    higherTimeframes,
    recentVolumeProfiles
  );

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
      bigTrades: orderFlowData.bigPrints.slice(-25).map(print => ({
        price: print.price,
        size: print.size,
        side: print.side,
        timestamp: new Date(print.timestamp).toISOString(),
      })),
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
    position: currentPosition
      ? (currentPosition.side === 'long' ? currentPosition.contracts : -currentPosition.contracts)
      : 0,
    unrealizedPnL: currentPosition?.unrealizedPnL || 0,
    realizedPnL,
  },
  openPosition: currentPosition
    ? {
        decisionId: currentPosition.decisionId,
        side: currentPosition.side,
        contracts: currentPosition.contracts,
        entryPrice: currentPosition.entryPrice,
        entryTime: currentPosition.entryTime,
        stopLoss: currentPosition.stopLoss,
        target: currentPosition.target,
        unrealizedPnL: currentPosition.unrealizedPnL,
        stopOrderId: currentPosition.stopOrderId,
        targetOrderId: currentPosition.targetOrderId,
        distanceToStopPoints: Number((currentPosition.side === 'long'
          ? currentPrice - currentPosition.stopLoss
          : currentPosition.stopLoss - currentPrice).toFixed(2)),
        distanceToTargetPoints: Number((currentPosition.side === 'long'
          ? currentPosition.target - currentPrice
          : currentPrice - currentPosition.target).toFixed(2)),
        positionAgeSeconds: Math.max(0, Math.round((Date.now() - new Date(currentPosition.entryTime).getTime()) / 1000)),
      }
    : null,

    // === ENHANCED FEATURES ===
    pocCrossStats,
    marketStats,
    cvdCandles,
    performance,
    historicalNotes,
    microstructure,
    macrostructure,
  };

  return marketData;
}

function buildMicrostructureFromOrderFlow(
  orderFlowData: OrderFlowData,
): FuturesMarketData['microstructure'] {
  const largeWhaleTrades = orderFlowData.bigPrints
    .slice(-25)
    .map(t => ({
      price: t.price,
      size: t.size,
      side: t.side,
      timestamp: new Date(t.timestamp).toISOString(),
    }));

  const restingLimitOrders = Object.entries(orderFlowData.volumeAtPrice || {})
    .map(([priceStr, vol]) => ({
      price: Number(priceStr),
      restingBid: Number(vol.buy ?? 0),
      restingAsk: Number(vol.sell ?? 0),
      total: Number(vol.buy ?? 0) + Number(vol.sell ?? 0),
      lastSeen: new Date(vol.timestamp).toISOString(),
    }))
    .filter(entry => !Number.isNaN(entry.price))
    .sort((a, b) => b.total - a.total)
    .slice(0, 10);

  return {
    largeWhaleTrades,
    restingLimitOrders,
  };
}

function buildMacrostructureFromBars(
  bars: TopstepXFuturesBar[],
  volumeProfile: VolumeProfile | null,
  sessionHigh: number,
  sessionLow: number,
  additionalTimeframes: HigherTimeframeSnapshot[] = [],
  recentProfiles: SessionVolumeProfileSummary[] = [],
): FuturesMarketData['macrostructure'] {
  if (!bars || bars.length === 0) {
    return undefined;
  }

  // Multi-session / session profile (approximate 24h trading day)
  const multiDayProfile = volumeProfile
    ? {
        lookbackHours: 24,
        poc: volumeProfile.poc,
        vah: volumeProfile.vah,
        val: volumeProfile.val,
        high: sessionHigh,
        low: sessionLow,
      }
    : undefined;

  // Aggregate 5-minute bars into 60-minute candles
  const oneHourMs = 60 * 60 * 1000;
  const bucketMap = new Map<number, {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>();

  for (const bar of bars) {
    const ts = new Date(bar.timestamp).getTime();
    if (Number.isNaN(ts)) {
      continue;
    }
    const bucketKey = Math.floor(ts / oneHourMs) * oneHourMs;
    const existing = bucketMap.get(bucketKey);
    if (!existing) {
      bucketMap.set(bucketKey, {
        timestamp: new Date(bucketKey).toISOString(),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        volume: bar.volume ?? 0,
      });
    } else {
      existing.high = Math.max(existing.high, bar.high);
      existing.low = Math.min(existing.low, bar.low);
      existing.close = bar.close;
      existing.volume += bar.volume ?? 0;
    }
  }

  const hourlyCandles = Array.from(bucketMap.values())
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  const recentHourly = hourlyCandles.slice(-12); // last ~12 hours

  const higherTimeframes: HigherTimeframeSnapshot[] = [];
  if (recentHourly.length > 0) {
    higherTimeframes.push({
      timeframe: '60m',
      candles: recentHourly,
    });
  }

  additionalTimeframes
    .filter(tf => tf && tf.candles && tf.candles.length > 0)
    .forEach(tf => {
      higherTimeframes.push({
        timeframe: tf.timeframe,
        candles: tf.candles,
      });
    });

  return {
    multiDayProfile,
    higherTimeframes: higherTimeframes.length > 0 ? higherTimeframes : undefined,
    recentVolumeProfiles: recentProfiles.length > 0 ? recentProfiles : undefined,
  };
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
    await maybeAdjustExistingPosition(executionManager, openaiDecision);
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
    // Check if signal is opposite to current position
    const isOppositeSignal =
      (activePosition.side === 'long' && openaiDecision.decision === 'SELL') ||
      (activePosition.side === 'short' && openaiDecision.decision === 'BUY');

    if (isOppositeSignal) {
      console.log(`[OpenAI] ⚠️  OPPOSITE SIGNAL DETECTED: Current position is ${activePosition.side.toUpperCase()} but got ${openaiDecision.decision} signal`);
      console.log(`[OpenAI] This would auto-reverse the position. Skipping to prevent unintended reversal.`);
      console.log(`[OpenAI] To close position, use explicit exit logic or let stop-loss/target handle it.`);
      return { executed: false };
    }

    await maybeAdjustExistingPosition(executionManager, openaiDecision);

    console.log('[OpenAI] Already in position - Not executing new entry');
    return { executed: false };
  }

  // Execute the decision
  const order = await executionManager.executeDecision(
    openaiDecision,
    currentPrice,
    {
      entryPrice: openaiDecision.entryPrice ?? null,
      stopLoss: openaiDecision.stopLoss ?? null,
      takeProfit: openaiDecision.target ?? null,
    }
  );

  if (!order) {
    // Even if no new order, allow position management adjustments
    await maybeAdjustExistingPosition(executionManager, openaiDecision);
    return { executed: false };
  }

  // Get the active position to update with additional context
  const position = executionManager.getActivePosition();
  if (position) {
    // Record in database with full Fabio context
    const decision = tradingDB.recordDecision({
      symbol,
      marketState: openaiDecision.marketState,
      location: openaiDecision.location,
      setupModel: openaiDecision.setupModel,
      decision: openaiDecision.decision as 'BUY' | 'SELL' | 'HOLD',
      confidence: openaiDecision.confidence,
      entryPrice: currentPrice,
      stopLoss: openaiDecision.stopLoss || position.stopLoss || currentPrice - 30,
      target: openaiDecision.target || position.target || currentPrice + 30,
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

    await maybeAdjustExistingPosition(executionManager, openaiDecision);

    return { executed: true, decisionId: decision.id };
  }

  return { executed: false };
}

async function maybeAdjustExistingPosition(
  executionManager: ExecutionManager,
  openaiDecision: OpenAITradingDecision
) {
  const activePosition = executionManager.getActivePosition();
  if (!activePosition) {
    return;
  }

  const desiredStop = typeof openaiDecision.stopLoss === 'number' ? openaiDecision.stopLoss : undefined;
  const desiredTarget = typeof openaiDecision.target === 'number' ? openaiDecision.target : undefined;

  if (desiredStop == null && desiredTarget == null) {
    console.log('[OpenAI] Active position detected but no stop/target guidance provided in JSON.');
    return;
  }

  const matchingSide =
    (activePosition.side === 'long' && openaiDecision.decision === 'BUY') ||
    (activePosition.side === 'short' && openaiDecision.decision === 'SELL') ||
    openaiDecision.decision === 'HOLD';

  if (!matchingSide) {
    return;
  }

  console.log(
    `[OpenAI] Monitoring active ${activePosition.side.toUpperCase()} — Proposed stop: ${desiredStop ?? 'unchanged'}, target: ${desiredTarget ?? 'unchanged'}`
  );

  const adjusted = await executionManager.adjustActiveProtection(desiredStop, desiredTarget);
  if (adjusted) {
    console.log('[OpenAI] 🔧 Updated protective orders based on latest plan.');
  } else {
    console.log('[OpenAI] Protective orders unchanged (levels already aligned or missing order IDs).');
  }
}

/**
 * Update active position and check for exits
 */
export async function updatePositionAndCheckExits(
  executionManager: ExecutionManager,
  currentPrice: number,
  bars: TopstepXFuturesBar[],
  openaiDecision?: OpenAITradingDecision,
  marketStructure?: MarketStructure
): Promise<{ exited: boolean; closedDecisionId?: string; reason?: string }> {
  const selfLearning = isSelfLearningEnabled();
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

    if (outcome) {
      if (selfLearning) {
        // Record performance for self-learning
        performanceTracker.recordTrade(outcome.profitLoss);

        // Add note if significant loss (> $200)
        if (outcome.profitLoss < -200) {
          const decision = tradingDB.getDecision(closedDecisionId);
          if (decision) {
            notesManager.addNote(
              `Large loss (${outcome.profitLoss.toFixed(2)}) on ${decision.setupModel || 'unknown'} setup - review entry conditions`,
              marketStructure?.state || 'unknown'
            );
          }
        }

        // Add note from OpenAI if provided
        if (openaiDecision?.noteForFuture) {
          notesManager.addNote(
            openaiDecision.noteForFuture,
            marketStructure?.state || 'unknown'
          );
        }
      }

      console.log(
        `[Position] ✅ Closed: ${outcome.reason} | P&L: ${outcome.profitLoss > 0 ? '+' : ''}$${outcome.profitLoss.toFixed(2)} (${outcome.profitLossPercent.toFixed(2)}%)`
      );
    }

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
