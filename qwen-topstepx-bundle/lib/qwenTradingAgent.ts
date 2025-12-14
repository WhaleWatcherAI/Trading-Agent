/**
 * Qwen Trading Agent - Local LLM version
 * Uses Qwen2.5:7b via Ollama for fast trading decisions
 * Drop-in replacement for openaiTradingAgent.ts
 */

import { jsonrepair } from 'jsonrepair';
import { fabioPlaybook, MarketState, SetupModel } from './fabioPlaybook';
import { POCCrossStats, MarketStatistics, PerformanceMetrics, HistoricalNote } from './enhancedFeatures';
import { chatCancellable, generate, checkHealth, OllamaChatMessage, getBacklogTimeoutMs, cancelActiveRequest } from './ollamaClient';
import { shouldProceedWithAnalysis, ChronosFilterResult } from './chronosFilter';
import * as fs from 'fs';
import * as path from 'path';

const DEFAULT_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:7b';

// Re-export all the interfaces from the original file
export interface FlowSignals {
  deltaLast1m?: number;
  deltaLast5m?: number;
  cvdSlopeShort?: number;
  cvdSlopeLong?: number;
  cvdDivergence?: 'none' | 'weak' | 'strong';
}

export interface AbsorptionSignal {
  levelName: string;
  price: number;
  side: 'bid' | 'ask';
  strength: number;
  durationSec: number;
  confirmedByCvd: boolean;
}

export interface ProfileSummary {
  poc: number;
  vah: number;
  val: number;
  hvns?: number[];
  lvns?: number[];
}

export interface TradeLegProfile extends ProfileSummary {
  valueMigration?: 'up' | 'down' | 'flat';
  acceptanceStrengthDir?: number;
  hvnDir?: number;
  lvnDir?: number;
  airPocketAhead?: boolean;
}

export interface PullbackProfile extends ProfileSummary {
  acceptanceState?: 'accepting' | 'rejecting' | 'rotating';
  active: boolean;
}

export interface WatchZoneProfile {
  name: string;
  low: number;
  high: number;
  poc?: number;
  acceptanceState?: 'accepting' | 'rejecting' | 'rotating';
  acceptanceStrength?: number;
  breakthroughScore?: number;
  absorptionScore?: number;
}

export interface FuturesMarketData {
  symbol: string;
  timestamp: string;
  currentPrice: number;
  connectionHealth?: {
    marketHubState: 'connected' | 'reconnecting' | 'disconnected';
    lastMarketHubEventAgoSec?: number | null;
    lastMarketHubDisconnectAgoSec?: number | null;
  };
  candles: {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
  cvd: {
    value: number;
    trend: 'up' | 'down' | 'neutral';
    ohlc: {
      timestamp: string;
      open: number;
      high: number;
      low: number;
      close: number;
    };
  };
  cvdCandles: Array<{
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
  }>;
  cvdMomentum?: {
    deltaEma20: number;           // 20 EMA of 1-minute delta (close - open)
    trend: 'up' | 'down';         // Direction based on deltaEma20 sign (no neutral - like ATAS)
    trendStrength: 'strong' | 'moderate' | 'weak';  // Based on magnitude
    minutesSinceTrendSwitch: number | null;  // null if >3 mins or insufficient data
    trendSwitchContext: string | null;  // "trend just switched" / "trend switched 1 minute ago" etc
  };
  orderFlow: {
    bigTrades: Array<{
      price: number;
      size: number;
      side: 'buy' | 'sell';
      timestamp: string;
    }>;
  };
  volumeProfile: {
    poc: number;
    vah: number;
    val: number;
    lvns: number[];
    sessionHigh: number;
    sessionLow: number;
  };
  marketState: MarketState;
  orderFlowConfirmed: boolean;
  account: {
    balance: number;
    position: number;
    unrealizedPnL: number;
    realizedPnL: number;
  };
  openPosition?: {
    decisionId?: string;
    side: 'long' | 'short';
    contracts: number;
    entryPrice: number;
    entryTime: string;
    stopLoss: number;
    target: number;
    unrealizedPnL: number;
    stopOrderId?: string | number;
    targetOrderId?: string | number;
    distanceToStopPoints: number;
    distanceToTargetPoints: number;
    positionAgeSeconds: number;
    positionVersion?: number;
  } | null;
  microstructure?: {
    largeWhaleTrades: Array<{
      price: number;
      size: number;
      side: 'buy' | 'sell';
      timestamp: string;
    }>;
    restingLimitOrders: Array<{
      price: number;
      restingBid: number;
      restingAsk: number;
      total: number;
      lastSeen: string;
    }>;
    nearestRestingWallInDirection?: {
      side: 'bid' | 'ask';
      price: number;
      size: number;
      distance: number;
    };
    liquidityPullDetected?: boolean;
    weakWallDetected?: boolean;
  };
  macrostructure?: {
    multiDayProfile?: {
      lookbackHours: number;
      poc: number;
      vah: number;
      val: number;
      high: number;
      low: number;
    };
    higherTimeframes?: Array<{
      timeframe: string;
      candles: Array<{
        timestamp: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume?: number;
      }>;
    }>;
    recentVolumeProfiles?: Array<{
      sessionStart: string;
      sessionEnd: string;
      poc: number;
      vah: number;
      val: number;
      lvns: number[];
      sessionHigh: number;
      sessionLow: number;
    }>;
  };
  pocCrossStats: POCCrossStats;
  marketStats: MarketStatistics;
  mlScores?: {
    p_win_5m?: number;
    p_win_30m?: number;
    modelVersion?: string;
  };
  performance: PerformanceMetrics | null;
  historicalNotes: HistoricalNote[];
  flowSignals?: FlowSignals;
  absorption?: AbsorptionSignal[];
  exhaustion?: AbsorptionSignal[];
  tradeLegProfile?: TradeLegProfile;
  pullbackProfile?: PullbackProfile;
  watchZoneProfiles?: WatchZoneProfile[];
  reversalScores?: {
    long: number;
    short: number;
  };
  footprintCandles?: Array<{
    timestamp: string;
    levels: Array<{
      price: number;
      buyVolume: number;
      sellVolume: number;
      delta: number;
      imbalance: 'buy' | 'sell' | 'neutral';
    }>;
  }>;
  brokenWalls?: Array<{
    price: number;
    side: 'bid' | 'ask';
    size: number;
    brokenAt: string;
    direction: 'up' | 'down';
  }>;
}

export interface OpenAITradingDecision {
  decision: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  marketState: MarketState;
  location: string;
  setupModel: SetupModel | null;
  entryPrice: number | null;
  stopLoss: number | null;
  target: number | null;
  reasoning: string;
  plan: string;
  timingPlan?: string;
  reEntryPlan?: string;
  riskManagementReasoning?: string;
  riskRewardRatio: number | null;
  riskPercent: number;
  inferredRegime?: 'trend' | 'range' | 'chop';
  regimeConfidence?: number;
  noteForFuture?: string;
  positionVersion?: number;
}

// System prompt - optimized for order flow trading with priority hierarchy
const SYSTEM_INSTRUCTIONS = `You are Fabio, a futures scalper/day trader. Make trading decisions based on ORDER FLOW first, technicals second.

ANALYSIS PRIORITY (strictly follow this order):
1. CVD EMA TREND (most important) - This tells you WHO is in control
   - CVD EMA BULLISH (deltaEma20 > 0) = Buyers winning → favor LONG
   - CVD EMA BEARISH (deltaEma20 < 0) = Sellers winning → favor SHORT
   - NEVER trade against the CVD EMA trend unless you see clear exhaustion/reversal

2. HTF BIAS (4H/Daily direction)
   - Check 4H and 1D candles - are they making higher highs or lower lows?
   - Trade WITH the higher timeframe direction, not against it
   - HTF bullish + LTF pullback = BUY opportunity
   - HTF bearish + LTF bounce = SELL opportunity

3. FOOTPRINT DELTA & IMBALANCES
   - Look at buy vs sell volume at each price level
   - Imbalance = aggressive buyers/sellers stepping in
   - Delta positive = more buying → supports LONG
   - Delta negative = more selling → supports SHORT
   - Absorption = price stalls while volume absorbed (reversal signal)

4. BID/ASK WALLS (L2 order book)
   - BID WALL = large resting buy orders = SUPPORT (price likely to bounce)
   - ASK WALL = large resting sell orders = RESISTANCE (price likely to reject)
   - Wall BROKEN = continuation signal in direction of break
   - Use walls for stop placement (stop BEYOND the wall that protects you)

5. TECHNICALS (lower priority)
   - Candle patterns, swing highs/lows
   - Only use to fine-tune entry after flow confirms direction

DO NOT rely heavily on Volume Profile (POC/VAH/VAL). These are reference only, not primary signals.

ENTRY CRITERIA (all must align):
✅ CVD EMA confirms direction (bullish for long, bearish for short)
✅ HTF bias supports trade direction
✅ Footprint shows delta/imbalance in your favor
✅ L2 wall provides protection for your stop

SHORT SETUPS:
- CVD EMA bearish (deltaEma20 < 0, trend = down)
- HTF showing lower highs/lower lows
- Footprint delta negative, sell imbalances
- ASK wall above holding as resistance OR BID wall broke down

LONG SETUPS:
- CVD EMA bullish (deltaEma20 > 0, trend = up)
- HTF showing higher lows/higher highs
- Footprint delta positive, buy imbalances
- BID wall below holding as support OR ASK wall broke up

HOLD if:
- CVD EMA is weak (close to zero but still directional)
- HTF and LTF disagree (e.g., CVD bearish but 4H bullish)
- No clear footprint imbalance
- No protective wall structure

Respond ONLY with valid JSON. No markdown.`;

/**
 * Build the analysis prompt from market data
 * Priority: CVD EMA → HTF Bias → Footprint → L2 Walls → Technicals
 */
function buildAnalysisPrompt(
  marketData: FuturesMarketData,
  recentCandles: any[],
  currentPrice: number
): string {
  const vp = marketData.volumeProfile;
  const flow = marketData.flowSignals;
  const micro = marketData.microstructure;
  const macro = marketData.macrostructure;
  const cvdMom = marketData.cvdMomentum;

  // === 1. CVD EMA TREND (HIGHEST PRIORITY) ===
  const cvdEmaTrend = cvdMom
    ? `${cvdMom.trend.toUpperCase()} (deltaEma20: ${cvdMom.deltaEma20.toFixed(1)}, strength: ${cvdMom.trendStrength})`
    : 'N/A';
  const cvdTrendSwitch = cvdMom?.trendSwitchContext || 'stable';
  // No neutral - CVD EMA is always bullish or bearish like ATAS
  const cvdBias = cvdMom?.trend === 'up' ? '→ FAVOR LONG (buyers in control)' : '→ FAVOR SHORT (sellers in control)';

  // === 2. HTF BIAS (4H/Daily) ===
  const getHTFBias = () => {
    const htf4h = macro?.higherTimeframes?.find(h => h.timeframe === '4h');
    const htf1d = macro?.higherTimeframes?.find(h => h.timeframe === '1d');

    let bias4h = 'neutral';
    let bias1d = 'neutral';

    if (htf4h?.candles && htf4h.candles.length >= 2) {
      const last = htf4h.candles[htf4h.candles.length - 1];
      const prev = htf4h.candles[htf4h.candles.length - 2];
      if (last.close > prev.close && last.low > prev.low) bias4h = 'bullish';
      else if (last.close < prev.close && last.high < prev.high) bias4h = 'bearish';
    }

    if (htf1d?.candles && htf1d.candles.length >= 2) {
      const last = htf1d.candles[htf1d.candles.length - 1];
      const prev = htf1d.candles[htf1d.candles.length - 2];
      if (last.close > prev.close && last.low > prev.low) bias1d = 'bullish';
      else if (last.close < prev.close && last.high < prev.high) bias1d = 'bearish';
    }

    return { bias4h, bias1d };
  };

  const htfBias = getHTFBias();
  const htfSummary = `4H: ${htfBias.bias4h.toUpperCase()} | Daily: ${htfBias.bias1d.toUpperCase()}`;
  const htfDirection = htfBias.bias4h === 'bullish' && htfBias.bias1d !== 'bearish' ? '→ HTF supports LONG'
    : htfBias.bias4h === 'bearish' && htfBias.bias1d !== 'bullish' ? '→ HTF supports SHORT'
    : '→ HTF mixed/neutral';

  // === 3. FOOTPRINT DELTA & IMBALANCES ===
  let footprintSummary = 'N/A';
  let footprintDelta = 0;
  let footprintBias = 'neutral';

  if (marketData.footprintCandles?.length) {
    const lastFP = marketData.footprintCandles[marketData.footprintCandles.length - 1];
    const levels = lastFP.levels.slice(0, 8);

    let totalBuy = 0, totalSell = 0;
    const imbalances: string[] = [];

    levels.forEach(l => {
      totalBuy += l.buyVolume;
      totalSell += l.sellVolume;
      if (l.imbalance !== 'neutral') {
        imbalances.push(`${l.price}:${l.imbalance.toUpperCase()}`);
      }
    });

    footprintDelta = totalBuy - totalSell;
    footprintBias = footprintDelta > 0 ? 'bullish' : footprintDelta < 0 ? 'bearish' : 'neutral';

    footprintSummary = `Delta: ${footprintDelta > 0 ? '+' : ''}${footprintDelta} (${footprintBias.toUpperCase()})`;
    if (imbalances.length > 0) {
      footprintSummary += ` | Imbalances: ${imbalances.slice(0, 4).join(', ')}`;
    }
  }

  // === 4. BID/ASK WALLS ===
  const l2Summary = micro?.restingLimitOrders
    ?.slice(0, 6)
    .map(o => `${o.price}:B${o.restingBid}/A${o.restingAsk}`)
    .join(' | ') || 'N/A';

  const nearestWall = micro?.nearestRestingWallInDirection;
  const wallInfo = nearestWall
    ? `${nearestWall.side.toUpperCase()} wall ${nearestWall.size}@${nearestWall.price} (${nearestWall.distance} ticks)`
    : 'None';

  const brokenWallsSummary = marketData.brokenWalls?.slice(-3)
    .map(w => `${w.side.toUpperCase()} broke ${w.direction}@${w.price}`)
    .join(' | ') || 'None';

  // === 5. TECHNICALS (lower priority) ===
  const candles1m = recentCandles.slice(-5).map(c =>
    `${c.close.toFixed(1)}(${c.close > c.open ? '+' : '-'})`
  ).join(' ');

  const formatHTFCandles = (tf: string, count: number) => {
    const htfData = macro?.higherTimeframes?.find(h => h.timeframe === tf);
    if (!htfData || !htfData.candles.length) return 'N/A';
    return htfData.candles.slice(-count).map(c =>
      `${c.close.toFixed(1)}(${c.close > c.open ? '+' : '-'})`
    ).join(' ');
  };

  const candles4h = formatHTFCandles('4h', 3);
  const candles1d = formatHTFCandles('1d', 3);

  // Whale prints
  const whales = micro?.largeWhaleTrades?.slice(-3)
    .map(t => `${t.side.toUpperCase()[0]}${t.size}@${t.price}`)
    .join(' ') || 'None';

  // Build prompt with new priority order
  let prompt = `${marketData.symbol} @ ${currentPrice} | ${marketData.timestamp.slice(11, 19)}

=== 1. CVD EMA TREND (Primary Signal) ===
TREND: ${cvdEmaTrend}
STATUS: ${cvdTrendSwitch}
${cvdBias}

=== 2. HTF BIAS (4H/Daily Direction) ===
${htfSummary}
${htfDirection}

=== 3. FOOTPRINT (Delta & Imbalances) ===
${footprintSummary}

=== 4. BID/ASK WALLS (L2) ===
NEAREST: ${wallInfo}
BOOK: ${l2Summary}
BROKEN: ${brokenWallsSummary}
WHALES: ${whales}

=== 5. TECHNICALS (Reference) ===
1m: ${candles1m}
4h: ${candles4h}
1D: ${candles1d}
Buyers: ${marketData.marketState.buyersControl?.toFixed(2) ?? 'N/A'} | Sellers: ${marketData.marketState.sellersControl?.toFixed(2) ?? 'N/A'}`;

  if (marketData.openPosition) {
    const pos = marketData.openPosition;
    prompt += `

OPEN POSITION - MANAGE THIS:
${pos.side.toUpperCase()} ${pos.contracts}x @ ${pos.entryPrice} | PnL: $${pos.unrealizedPnL.toFixed(2)}
Stop: ${pos.stopLoss} (${pos.distanceToStopPoints.toFixed(1)} pts) | Target: ${pos.target} (${pos.distanceToTargetPoints.toFixed(1)} pts)
Age: ${(pos.positionAgeSeconds / 60).toFixed(1)} min

TASK: Manage position. Adjust stop/target or exit. Set decision=HOLD to keep, decision=${pos.side === 'long' ? 'SELL' : 'BUY'} to exit.`;
  } else {
    prompt += `

POSITION: FLAT

TASK: Find entry or HOLD. If edge unclear, HOLD with trigger conditions.`;
  }

  prompt += `

Respond with JSON only:
{"decision":"BUY|SELL|HOLD","confidence":0-100,"entryPrice":null|number,"stopLoss":null|number,"target":null|number,"riskRewardRatio":null|number,"riskPercent":0.25-0.5,"reasoning":"brief explanation","plan":"action plan","riskManagementReasoning":"why these SL/TP levels"}`;

  return prompt;
}

/**
 * Parse the LLM response into a trading decision
 */
function parseQwenResponse(content: string): OpenAITradingDecision {
  try {
    let jsonStr: string | null = null;
    let parsed: any = null;

    // Try extracting JSON from markdown blocks
    const markdownMatch = content.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (markdownMatch) {
      jsonStr = markdownMatch[1].trim();
    }

    // Try finding JSON object with brace matching
    if (!jsonStr) {
      const firstBrace = content.indexOf('{');
      if (firstBrace !== -1) {
        let braceCount = 0;
        let lastBrace = -1;
        for (let i = firstBrace; i < content.length; i++) {
          if (content[i] === '{') braceCount++;
          if (content[i] === '}') {
            braceCount--;
            if (braceCount === 0) {
              lastBrace = i;
              break;
            }
          }
        }
        if (lastBrace !== -1) {
          jsonStr = content.substring(firstBrace, lastBrace + 1);
        }
      }
    }

    // Greedy regex fallback
    if (!jsonStr) {
      const greedyMatch = content.match(/\{[\s\S]*\}/);
      if (greedyMatch) {
        jsonStr = greedyMatch[0];
      }
    }

    if (!jsonStr) {
      console.error('[Qwen] No JSON found in response');
      throw new Error('No JSON found');
    }

    try {
      parsed = JSON.parse(jsonStr);
    } catch {
      parsed = JSON.parse(jsonrepair(jsonStr));
    }

    const decision = typeof parsed.decision === 'string'
      ? parsed.decision.toUpperCase()
      : 'HOLD';

    return {
      decision,
      confidence: Math.min(100, Math.max(0, parsed.confidence || 50)),
      marketState: parsed.marketState || 'balanced',
      location: parsed.location || 'at_poc',
      setupModel: parsed.setupModel || null,
      entryPrice: parsed.entryPrice || null,
      stopLoss: parsed.stopLoss || null,
      target: parsed.target || null,
      reasoning: parsed.reasoning || 'No reasoning provided',
      plan: parsed.plan || 'No plan provided',
      timingPlan: parsed.timingPlan,
      reEntryPlan: parsed.reEntryPlan,
      riskManagementReasoning: parsed.riskManagementReasoning,
      riskRewardRatio: parsed.riskRewardRatio || null,
      riskPercent: Math.min(0.5, Math.max(0.25, parsed.riskPercent || 0.35)),
      inferredRegime: parsed.inferredRegime,
      regimeConfidence: parsed.regimeConfidence,
      noteForFuture: parsed.noteForFuture,
    };
  } catch (error: any) {
    console.error('[Qwen] Parse error:', error.message);
    console.error('[Qwen] Raw content:', content.substring(0, 500));

    return {
      decision: 'HOLD',
      confidence: 0,
      marketState: 'balanced',
      location: 'at_poc',
      setupModel: null,
      entryPrice: null,
      stopLoss: null,
      target: null,
      reasoning: `Parse error: ${error.message}`,
      plan: 'Waiting for next analysis',
      riskRewardRatio: null,
      riskPercent: 0.35,
    };
  }
}

/**
 * Main analysis function - uses local Qwen via Ollama
 * @param marketData - Market data for analysis
 * @param timeoutMs - Optional timeout override (use shorter timeout when backed up)
 */
export async function analyzeFuturesMarket(
  marketData: FuturesMarketData,
  timeoutMs?: number
): Promise<OpenAITradingDecision> {
  try {
    const recentCandles = marketData.candles.slice(-5);
    const currentPrice = marketData.candles[marketData.candles.length - 1]?.close || 0;

    const prompt = buildAnalysisPrompt(marketData, recentCandles, currentPrice);

    const effectiveTimeout = timeoutMs ?? 30000;
    console.log(`📊 [Qwen] Analyzing ${marketData.symbol} at ${currentPrice} (timeout: ${effectiveTimeout}ms)`);

    const messages: OllamaChatMessage[] = [
      { role: 'system', content: SYSTEM_INSTRUCTIONS },
      { role: 'user', content: prompt },
    ];

    const startTime = Date.now();
    // Use chatCancellable to automatically cancel any previous stale request
    // This prevents Ollama queue backup when requests take too long
    const response = await chatCancellable(messages, {
      model: DEFAULT_MODEL,
      temperature: 0.3,
      timeoutMs: effectiveTimeout,
    });
    const elapsed = Date.now() - startTime;

    const content = response.message?.content || '';

    console.log(`✅ [Qwen] Response in ${elapsed}ms`);

    const decision = parseQwenResponse(content);

    console.log(`📍 [Qwen] ${decision.decision} | Conf: ${decision.confidence}% | Entry: ${decision.entryPrice} | SL: ${decision.stopLoss} | TP: ${decision.target}`);

    // Run Chronos shadow mode (non-blocking, just logs for analysis)
    runChronosShadow(marketData, decision).catch(() => {}); // Fire and forget

    return decision;
  } catch (error: any) {
    console.error('❌ [Qwen] Analysis failed:', error.message);
    // NOTE: Don't call checkHealth() here - it uses the mutex and can cause deadlock!
    // The error message itself is sufficient diagnostics
    throw error;
  }
}

/**
 * Build decision prompt payload (for external use)
 */
export function buildDecisionPromptPayload(
  marketData: FuturesMarketData
): { systemInstructions: string; decisionPrompt: string } {
  const recentCandles = marketData.candles.slice(-5);
  const currentPrice = marketData.candles[marketData.candles.length - 1]?.close || 0;
  const prompt = buildAnalysisPrompt(marketData, recentCandles, currentPrice);

  return {
    systemInstructions: SYSTEM_INSTRUCTIONS,
    decisionPrompt: prompt,
  };
}

// Chronos shadow mode - runs alongside Qwen for data collection (no filtering)
const CHRONOS_SHADOW_ENABLED = process.env.CHRONOS_SHADOW_ENABLED === 'true'; // Disabled by default
const CHRONOS_PRICE_HISTORY = parseInt(process.env.CHRONOS_PRICE_HISTORY || '60', 10);
const CHRONOS_LOG_FILE = process.env.CHRONOS_LOG_FILE || path.join(__dirname, '..', 'ml', 'data', 'chronos_live_log.jsonl');

// Log Chronos prediction for later analysis
interface ChronosLogEntry {
  timestamp: string;
  symbol: string;
  currentPrice: number;
  chronosDirection: string;
  chronosConfidence: number;
  chronosExpectedMove: number | null;
  qwenDecision: string;
  qwenConfidence: number;
  entryPrice: number | null;
  stopLoss: number | null;
  target: number | null;
  priceHistory: number[]; // Last N prices for replay
}

function logChronosPrediction(entry: ChronosLogEntry) {
  try {
    const line = JSON.stringify(entry) + '\n';
    fs.appendFileSync(CHRONOS_LOG_FILE, line);
  } catch (err) {
    console.error('[Chronos] Failed to log prediction:', err);
  }
}

/**
 * Run Chronos in shadow mode - logs prediction alongside Qwen decision
 * Does NOT filter or block any trades
 */
async function runChronosShadow(
  marketData: FuturesMarketData,
  qwenDecision: OpenAITradingDecision
): Promise<void> {
  if (!CHRONOS_SHADOW_ENABLED) return;

  try {
    const priceHistory = marketData.candles
      .slice(-CHRONOS_PRICE_HISTORY)
      .map(c => c.close);

    if (priceHistory.length < 10) return;

    const currentPrice = priceHistory[priceHistory.length - 1];

    // Run Chronos prediction (don't await blocking - fire and forget with timeout)
    const filterResult = await shouldProceedWithAnalysis(
      priceHistory,
      marketData.symbol,
      0 // No threshold - we want all predictions logged
    );

    // Log for later analysis
    logChronosPrediction({
      timestamp: marketData.timestamp,
      symbol: marketData.symbol,
      currentPrice,
      chronosDirection: filterResult.chronosResult.direction,
      chronosConfidence: filterResult.chronosResult.confidence,
      chronosExpectedMove: filterResult.chronosResult.expected_move ?? null,
      qwenDecision: qwenDecision.decision,
      qwenConfidence: qwenDecision.confidence,
      entryPrice: qwenDecision.entryPrice,
      stopLoss: qwenDecision.stopLoss,
      target: qwenDecision.target,
      priceHistory: priceHistory.slice(-60),
    });

    // Log alignment
    const chronosDir = filterResult.chronosResult.direction === 'up' ? 'long' : filterResult.chronosResult.direction === 'down' ? 'short' : null;
    const qwenDir = qwenDecision.decision === 'BUY' ? 'long' : qwenDecision.decision === 'SELL' ? 'short' : null;

    if (chronosDir && qwenDir) {
      const aligned = chronosDir === qwenDir;
      console.log(
        `📊 [Chronos Shadow] ${filterResult.chronosResult.direction.toUpperCase()} ` +
        `(${(filterResult.chronosResult.confidence * 100).toFixed(0)}%) ` +
        `${aligned ? '✓ ALIGNED' : '✗ DIVERGE'} with Qwen ${qwenDecision.decision}`
      );
    } else {
      console.log(
        `📊 [Chronos Shadow] ${filterResult.chronosResult.direction.toUpperCase()} ` +
        `(${(filterResult.chronosResult.confidence * 100).toFixed(0)}%) | Qwen: ${qwenDecision.decision}`
      );
    }
  } catch (err) {
    console.error('[Chronos Shadow] Error:', err);
  }
}

// Note: Filter mode removed - using shadow mode only for now
// To re-enable filtering, set CHRONOS_FILTER_ENABLED=true and restore analyzeFuturesMarketWithFilter

/**
 * Continuous analysis loop
 */
export async function startContinuousAnalysis(
  getMarketData: () => Promise<FuturesMarketData>,
  onDecision: (decision: OpenAITradingDecision) => void,
  intervalMs: number = 30000 // Faster interval since Qwen is quick
) {
  console.log(`🤖 Starting Qwen futures analysis loop (every ${intervalMs}ms)`);

  // Check Ollama health first
  const health = await checkHealth();
  if (!health.ok) {
    console.error(`❌ Ollama not available: ${health.error}`);
    console.error('Start Ollama with: ollama serve');
    throw new Error('Ollama not running');
  }
  console.log(`✅ Ollama ready with model: ${health.model}`);

  let isAnalyzing = false;

  const analysisLoop = setInterval(async () => {
    if (isAnalyzing) {
      console.log('⏳ Previous analysis still running, skipping...');
      return;
    }

    try {
      isAnalyzing = true;
      const marketData = await getMarketData();
      const decision = await analyzeFuturesMarket(marketData);
      onDecision(decision);
    } catch (error) {
      console.error('Analysis loop error:', error);
    } finally {
      isAnalyzing = false;
    }
  }, intervalMs);

  return () => {
    clearInterval(analysisLoop);
    console.log('🛑 Stopped Qwen analysis loop');
  };
}
