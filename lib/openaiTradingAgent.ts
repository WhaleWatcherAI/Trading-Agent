import OpenAI from 'openai';
import type { ReasoningEffort } from 'openai/resources/shared';
import { jsonrepair } from 'jsonrepair';
import { fabioPlaybook, MarketState, SetupModel } from './fabioPlaybook';
import { POCCrossStats, MarketStatistics, PerformanceMetrics, HistoricalNote } from './enhancedFeatures';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: resolveOpenAIBaseURL(),
  timeout: 180000, // Give DeepSeek Reasoner enough time for chain-of-thought
});

const DEFAULT_OPENAI_MODEL = 'deepseek-reasoner';

const VALID_REASONING_EFFORTS: ReasoningEffort[] = ['minimal', 'low', 'medium', 'high'];

// Hysteresis: require consecutive scan hits to promote to reasoner (per symbol)
const promoteStreakBySymbol: Record<string, number> = {};

type ResponseOutputItem = OpenAIResponse['output'][number];

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
  strength: number; // 0-1
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
  acceptanceStrengthDir?: number; // 0-1
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

/**
 * Real-time futures market data from TopStepX
 * This is the actual data being streamed from the Fabio agent
 * ENHANCED with POC cross tracking, market statistics, and self-learning feedback
 */
export interface FuturesMarketData {
  symbol: string;
  timestamp: string;
  currentPrice: number;

  // Price candles (5-minute bars)
  candles: {
    timestamp: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];

  // CVD (Cumulative Volume Delta) - order flow strength
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

  // Order flow absorption/exhaustion
  orderFlow: {
    bigTrades: Array<{
      price: number;
      size: number;
      side: 'buy' | 'sell';
      timestamp: string;
    }>;
  };

  // Volume Profile structure
  volumeProfile: {
    poc: number;  // Point of Control
    vah: number;  // Value Area High
    val: number;  // Value Area Low
    lvns: number[]; // Low Volume Nodes
    sessionHigh: number;
    sessionLow: number;
  };

  // Market state from rule-based analysis
  marketState: MarketState;

  // Order flow confirmation
  orderFlowConfirmed: boolean;

  // Account info
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
  } | null;

  // Microstructure snapshot derived from order flow
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

  // Higher-timeframe / multi-session structure
  macrostructure?: {
    // Approximate session / multi-hour profile
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

  // === ENHANCED FEATURES ===

  // POC Cross Statistics (Critical for regime detection)
  pocCrossStats: POCCrossStats;

  // Raw Market Statistics (for LLM regime inference)
  marketStats: MarketStatistics;

  // Performance Metrics (self-learning feedback)
  performance: PerformanceMetrics | null;

  // Historical Notes (lessons learned)
  historicalNotes: HistoricalNote[];

  // Flow signals (delta/CVD derivatives)
  flowSignals?: FlowSignals;

  // Absorption/exhaustion proxies around key levels
  absorption?: AbsorptionSignal[];
  exhaustion?: AbsorptionSignal[];

  // Event/zone profiles
  tradeLegProfile?: TradeLegProfile;
  pullbackProfile?: PullbackProfile;
  watchZoneProfiles?: WatchZoneProfile[];

  reversalScores?: {
    long: number;
    short: number;
  };
}

/**
 * OpenAI Decision based on Fabio's 3 decision layers:
 * 1. Market State (balanced vs imbalanced)
 * 2. Location (price relative to volume profile)
 * 3. Order Flow Aggression (CVD, absorption, exhaustion)
 * ENHANCED with regime inference and self-learning
 */
export interface OpenAITradingDecision {
  decision: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0-100

  // Fabio's 3 decision layers
  marketState: MarketState;
  location: string; // 'at_poc' | 'at_vah' | 'at_val' | 'at_lvn' | 'above_poc' | 'below_poc'
  setupModel: SetupModel | null; // 'trend_continuation' | 'mean_reversion'

  entryPrice: number | null;
  stopLoss: number | null;
  target: number | null;

  reasoning: string;
  plan: string; // REQUIRED: Always provide a plan for next actions (entry zones, stop placement, contingencies)
  timingPlan?: string;
  reEntryPlan?: string;
  riskManagementReasoning?: string; // Specific explanation for SL/TP placement (e.g., "SL below recent swing low at 25128.5, TP at volume shelf at 25162")
  riskRewardRatio: number | null;
  riskPercent: number; // 0.25-0.5 per Fabio rules

  // === ENHANCED SELF-LEARNING FIELDS ===
  inferredRegime?: 'trend' | 'range' | 'chop'; // LLM infers from raw stats
  regimeConfidence?: number; // 0-100 confidence in regime assessment
  noteForFuture?: string; // Note to add to historical notes
}

/**
 * Analyze real futures market data with OpenAI
 * Uses GPT-4 to interpret order flow, volume profile, and candlestick patterns
 */
export async function analyzeFuturesMarket(
  marketData: FuturesMarketData
): Promise<OpenAITradingDecision> {
  try {
    const recentCandles = marketData.candles.slice(-5); // Last 5 candles (25 minutes)
    const currentPrice = marketData.candles[marketData.candles.length - 1]?.close || 0;

    let prompt: string;
    try {
      prompt = buildAnalysisPrompt(marketData, recentCandles, currentPrice);
      console.log(`[DEBUG] Built prompt successfully, size: ${prompt.length} characters`);
    } catch (error) {
      console.error('[ERROR] Failed to build analysis prompt:', error);
      throw new Error(`Prompt building failed: ${error}`);
    }

    console.log(`📊 [OpenAI] Analyzing ${marketData.symbol} at ${currentPrice}`);
    // Debug prompt logging disabled to reduce output
    // console.log('='.repeat(80));
    // console.log('[DEBUG] EXACT PROMPT SENT TO DEEPSEEK:');
    // console.log('='.repeat(80));
    // console.log(prompt);
    // console.log('='.repeat(80));

    const systemInstructions = `You are Fabio, a multi-time Robbins World Cup champion trading NQ/MNQ futures with elite auction-market skill, order-flow intuition, and hedge-fund level risk discipline. You are a probabilistic trader, not a rules bot. Your job is to produce high-EV trades and precise triggers based on the current tape.

Mindset
- Intraday, scalp-focused: prioritize high-probability, tight-stop intraday setups over swing holds.
- Favor tight, structure-based stops sized to current volatility; exits are fast if edge erodes.
- ALWAYS pick a direction (BUY or SELL) - never HOLD. Your job is to read the tape and have an opinion on the most likely next move.
- Express your conviction through CONFIDENCE (0-100%). Low confidence (50-60%) = weak edge, choppy. High confidence (75-95%) = strong edge, clear setup.
- The system will filter your recommendations by confidence threshold. You just provide your best read on direction and how confident you are.
- REVERSALS ARE VALID: Don't be afraid to fade extremes when structure + location align, even if flow hasn't confirmed yet. The best reversals often happen BEFORE flow flips.
- NEUTRAL FLOW IS NOT A LOW CONFIDENCE REASON: If balance/trend + location provide edge, neutral/absent flow just means slightly lower confidence (65-70% instead of 80%+), not 50%. Flow is a booster, not a requirement.

MULTI-TIMEFRAME ANALYSIS (Critical Framework)
- HIGHER TIMEFRAMES (4H, Daily) = NARRATIVE & DIRECTION: Use to establish the dominant trend, key levels, and overall bias. Trade WITH the higher timeframe trend unless clear reversal evidence.
- LOWER TIMEFRAMES (1-min, 15-min) = ENTRY, STOP LOSS, TARGET POSITIONING: Use for precise entry timing, stop placement behind structure, and target placement before resistance/support.
- The HTF tells you WHAT to trade (direction bias), the LTF tells you WHERE and WHEN to trade (execution).

How You Think (in this order)
1) HTF Narrative: Check 4H and Daily candles first. What's the dominant trend? Where are major swing highs/lows? Is price at a significant HTF level?
2) LTF Structure: Use 1-min and 15-min for local swings, recent highs/lows, and immediate support/resistance for entry/exit positioning.
3) Flow Evidence (CONFIDENCE BOOSTER ONLY): CVD trend, delta impulses, whale prints, L2 liquidity walls. Flow confirmation INCREASES CONFIDENCE but is NOT MANDATORY for entry.
4) Expected Value (EV): continuation vs reversal probability; only trade when R/R and balance/location alignment create positive EV.
5) Risk/Reward Ratio (MINIMUM 1:3 REQUIRED): Target distance must be AT LEAST 3x the stop distance. This is NON-NEGOTIABLE for intraday/scalp trades.
6) Triggers & Contingencies: "If price does X → I do Y" with exact entry/stop/target numbers.

LIQUIDITY WALL TRACKING (Evidence, Not Religion)
- WALL HELD (price rejected): Consider it support/resistance. For reversals, place stop BEHIND the wall. Wall becomes your protection.
- WALL BROKEN (price sliced through): Indicates continuation momentum. Don't fade the break - flow WITH it. Broken walls often become support/resistance on retest.
- RECENT BREAK vs HELD: The data shows if walls were recently broken or held. Use this context for trade direction.

Toolbox (evidence, not religion)
- Structure: swings, breaks, failures, pullback depth, trend strength from candle analysis.
- Order Flow / L2: CVD trend, delta (buying vs selling pressure), aggressive vs passive prints, stacked liquidity walls.
- Liquidity Walls: Track if recently broken (continuation) or held (reversal potential).
- Volatility Regime: size stops/targets to the regime; wider in strong trends, tighter in chop.

STOP LOSS PLACEMENT (Critical for Protection):
- Place stops BEHIND key support/resistance levels (1-2 ticks beyond invalidation point)
- For LONGS: Stop below the nearest support level or bid wall - NOT at it
- For SHORTS: Stop above the nearest resistance level or ask wall - NOT at it
- Use bid/ask walls as protection: place stops BEHIND where large resting orders provide support
- Never place stops at obvious round numbers or exact technical levels (stop hunts target these)

TAKE PROFIT PLACEMENT (Critical for Fills):
- Place targets BEFORE key resistance/support levels (1-2 ticks before to ensure fills)
- For LONGS: Target 1-2 ticks BELOW the next resistance or ask wall
- For SHORTS: Target 1-2 ticks ABOVE the next support or bid wall
- Don't be greedy - get filled BEFORE the level, not rejected at it
- Factor in resting liquidity walls that may absorb momentum before your target

Execution Discipline
- Your JSON is the literal instruction to execute. If reasoning says BUY/SELL, JSON must say BUY/SELL with concrete prices.
- Never mismatch text vs JSON (no BUY/SELL in text while JSON is HOLD, or vice versa).
- Stops just beyond invalidation (structure/liquidity); never widen after entry.
- Assume MARKET entries only. If the setup is not active right now, HOLD and give the exact trigger for a future market entry.
- Always state whether the setup is a continuation or a reversal and the evidence for that choice.
- Always explain in riskManagementReasoning WHY your stop is placed behind a specific level and your target before another level.

Precision & Validity
- All price outputs must be valid tick increments for the symbol.
- Distances to levels are signed (currentPrice − level); negative = below, positive = above.
- If any microstructure fields are missing, treat them as unknown, not zero.
- Every decision must reference current numeric evidence from the snapshot.

Respond ONLY in JSON with the required fields for execution.`;

    // ---------- DECISION MODE ONLY (reasoner every interval) ----------
    const messages = [
      { role: 'system' as const, content: systemInstructions },
      { role: 'user' as const, content: prompt },
    ];

    const model = resolveOpenAIModel();
    const useReasoner = shouldUseReasoningModel(model);
    // IMPORTANT: For DeepSeek, don't use json_object format because it wastes
    // reasoning tokens on "how to format JSON" instead of market analysis.
    // We extract JSON from the natural response instead.
    const responseFormat = (model.includes('deepseek'))
      ? undefined
      : { type: 'json_object' as const };
    let content: string | null = null;
    const reasoningEffort = resolveReasoningEffort();

    if (useReasoner) {
        const reasonerAttempts: Array<{ stream: boolean; maxTokens: number; temperature: number }> = [
          { stream: false, maxTokens: 2000, temperature: 0.35 },
          { stream: false, maxTokens: 1600, temperature: 0.35 },
          { stream: false, maxTokens: 1200, temperature: 0.35 },
        ];

        for (let i = 0; i < reasonerAttempts.length; i += 1) {
          const attempt = reasonerAttempts[i];
          try {
            if (attempt.stream) {
              const stream = await openai.chat.completions.create({
                model,
                messages: messages,
                max_tokens: attempt.maxTokens,
                temperature: attempt.temperature,
                reasoning_effort: reasoningEffort,
                ...(responseFormat ? { response_format: responseFormat } : {}),
                stream: true,
              });

              let streamedContent = '';
              let reasoningLog = '';

              for await (const chunk of stream) {
                const choice = chunk.choices?.[0];
                if (!choice) continue;
                const delta: any = choice.delta || {};

                if (Array.isArray(delta.content)) {
                  delta.content.forEach((part: any) => {
                    const text = extractTextPart(part);
                    if (text) streamedContent += text;
                  });
                } else if (typeof delta.content === 'string') {
                  streamedContent += delta.content;
                }

                if (Array.isArray(delta.reasoning_content)) {
                  delta.reasoning_content.forEach((part: any) => {
                    const text = extractTextPart(part);
                    if (text) reasoningLog += text;
                  });
                }
              }

              if (reasoningLog.trim()) {
                console.log('='.repeat(80));
                console.log('[DEBUG] DEEPSEEK REASONER THINKING:');
                console.log('='.repeat(80));
                console.log(reasoningLog.trim());
                console.log('='.repeat(80));
              }

              content = streamedContent.trim();
            } else {
              const completion = await openai.chat.completions.create({
                model,
                messages: messages,
                max_tokens: attempt.maxTokens,
                temperature: attempt.temperature,
                reasoning_effort: reasoningEffort,
                ...(responseFormat ? { response_format: responseFormat } : {}),
              });

              const message: any = completion.choices?.[0]?.message;
              const reasoning = extractReasoningFromMessage(message);
              if (reasoning) {
                console.log('='.repeat(80));
                console.log('[DEBUG] DEEPSEEK REASONER THINKING:');
                console.log('='.repeat(80));
                console.log(reasoning);
                console.log('='.repeat(80));
              }

              content = extractMessageText(message);

              // Debug logging for DeepSeek responses
              console.log('[DEBUG] DeepSeek Response Analysis:');
              console.log('- Has content:', !!message.content);
              console.log('- Has reasoning_content:', !!message.reasoning_content);
              if (message.content) {
                console.log('- Content type:', typeof message.content);
                console.log('- Content preview (first 200 chars):',
                  typeof message.content === 'string'
                    ? message.content.substring(0, 200)
                    : JSON.stringify(message.content).substring(0, 200));
              }
              if (content) {
                console.log('- Extracted content preview (first 200 chars):', content.substring(0, 200));
              }
            }

            if (content && content.trim()) {
              break;
            }
            throw new Error('Reasoner returned empty response');
          } catch (error: any) {
            console.warn(`[OpenAI] Reasoner attempt ${i + 1} failed: ${error?.message || error}`);
            const shouldRetry = shouldRetryReasoner(error) && i < reasonerAttempts.length - 1;
            if (!shouldRetry) {
              // If all reasoner attempts failed, fall back to deepseek-chat or GPT-4
              console.warn('[OpenAI] All reasoner attempts failed. Falling back to deepseek-chat/GPT-4...');
              try {
                const fallbackModel = model.includes('deepseek') ? 'deepseek-chat' : 'gpt-4o';
                console.log(`[OpenAI] Using fallback model: ${fallbackModel}`);

                const fallbackResponse = await openai.chat.completions.create({
                  model: fallbackModel,
                  messages: messages,
                  max_tokens: 700,
                  temperature: 0.2,
                  response_format: responseFormat,
                });

                content = extractMessageText(fallbackResponse.choices[0]?.message);

                if (content && content.trim()) {
                  console.log('[OpenAI] Fallback model succeeded');
                  break;
                } else {
                  throw new Error(`Fallback model ${fallbackModel} also returned empty response`);
                }
              } catch (fallbackError: any) {
                console.error('[OpenAI] Fallback also failed:', fallbackError?.message || fallbackError);
                throw error; // Throw original error
              }
            }
            console.warn('[OpenAI] Retrying reasoner with alternate delivery (non-stream).');
          }
        }
      } else {
        const response = await openai.chat.completions.create({
          model,
          temperature: 0.7,
          max_tokens: 1000,
          messages: messages,
          ...(responseFormat ? { response_format: responseFormat } : {}),
        });
        content = extractMessageText(response.choices[0]?.message);
      }

    if (!content) {
      throw new Error('No content generated by OpenAI response');
    }

    const decision = parseOpenAIResponse(content);

    console.log(`✅ [OpenAI:DECISION] ${decision.decision} @ ${decision.entryPrice} | Confidence: ${decision.confidence}%`);

    return decision;
  } catch (error) {
    console.error('❌ [OpenAI] Analysis failed:', error);
    throw error;
  }
}

/**
 * Aggregate 5-minute candles into higher timeframes
 * @param candles - Array of 5-minute candles
 * @param targetMinutes - Target timeframe in minutes (must be multiple of 5)
 * @returns Aggregated candles
 */
function aggregateCandlesToTimeframe(
  candles: any[],
  targetMinutes: number,
  baseMinutes: number = 5
): any[] {
  if (candles.length === 0 || targetMinutes <= baseMinutes) {
    return candles;
  }

  const barsPerCandle = targetMinutes / baseMinutes;
  if (!Number.isInteger(barsPerCandle)) {
    return candles;
  }
  const aggregated: any[] = [];

  for (let i = 0; i < candles.length; i += barsPerCandle) {
    const chunk = candles.slice(i, i + barsPerCandle);
    if (chunk.length === 0) continue;

    aggregated.push({
      timestamp: chunk[0].timestamp,
      open: chunk[0].open,
      high: Math.max(...chunk.map(c => c.high)),
      low: Math.min(...chunk.map(c => c.low)),
      close: chunk[chunk.length - 1].close,
      volume: chunk.reduce((sum, c) => sum + (c.volume || 0), 0),
    });
  }

  return aggregated;
}

/**
 * Build a detailed market analysis prompt for OpenAI
 * Teaches GPT-4 Fabio's 3-layer decision framework
 * ENHANCED: Multi-timeframe analysis (5-min, 15-min, 60-min)
 */
function buildAnalysisPrompt(
  marketData: FuturesMarketData,
  recentCandles: any[],
  currentPrice: number
): string {
  const cvdTrend = marketData.cvd.trend === 'up' ? '🟢 BULLISH' :
                   marketData.cvd.trend === 'down' ? '🔴 BEARISH' : '⚪ NEUTRAL';

  const distToPoc = (marketData.volumeProfile && typeof marketData.volumeProfile.poc === 'number')
    ? Number((marketData.currentPrice - marketData.volumeProfile.poc).toFixed(2))
    : undefined;
  const distToVah = (marketData.volumeProfile && typeof marketData.volumeProfile.vah === 'number')
    ? Number((marketData.currentPrice - marketData.volumeProfile.vah).toFixed(2))
    : undefined;
  const distToVal = (marketData.volumeProfile && typeof marketData.volumeProfile.val === 'number')
    ? Number((marketData.currentPrice - marketData.volumeProfile.val).toFixed(2))
    : undefined;

  const nearestWall = marketData.microstructure?.nearestRestingWallInDirection;
  const currentBarRange = marketData.marketStats?.currentRangeTicks ?? null;
  const atr = marketData.marketStats?.atr5m ?? null;
  const stateSummary = `MARKET SNAPSHOT:
- location: distToPOC/VAH/VAL = ${distToPoc ?? 'n/a'} / ${distToVah ?? 'n/a'} / ${distToVal ?? 'n/a'} ticks
- flow: delta1m/5m ${marketData.flowSignals?.deltaLast1m ?? 'n/a'} / ${marketData.flowSignals?.deltaLast5m ?? 'n/a'}, CVD ${marketData.cvd?.value ?? 'n/a'}
- liquidity: nearest wall ${nearestWall ? `${nearestWall.side}@${nearestWall.price.toFixed(2)} dist=${nearestWall.distance}` : 'n/a'}
- volatility: current bar ${currentBarRange?.toFixed(1) ?? 'n/a'} ticks, ATR(14) ${atr?.toFixed(1) ?? 'n/a'} ticks`;

  // Build multi-timeframe candle summaries from minute base data
  const allCandles = marketData.candles;

  // 1-minute candles (last 20 candles = 20 minutes) - for precise entry/exit
  const oneMinCandles = allCandles.slice(-20);
  const oneMinSummary = oneMinCandles
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
    .join('\n  ');

  // 15-minute candles (last 8 candles = 2 hours) - for local structure
  const fifteenMinCandles = aggregateCandlesToTimeframe(allCandles, 15, 1);
  const fifteenMinSummary = fifteenMinCandles.slice(-8)
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
    .join('\n  ');

  // 4-hour candles (last 6 candles = 24 hours) - for intermediate trend/narrative
  const fourHourCandles = aggregateCandlesToTimeframe(allCandles, 240, 1);
  const fourHourSummary = fourHourCandles.slice(-6)
    .map((c) => `${new Date(c.timestamp).toLocaleDateString()} ${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
    .join('\n  ');

  // Daily candles (last 6 candles = 6 days) - for major trend/narrative
  const dailyCandles = aggregateCandlesToTimeframe(allCandles, 1440, 1);
  const dailySummary = dailyCandles.slice(-6)
    .map((c) => `${new Date(c.timestamp).toLocaleDateString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
    .join('\n  ');

  const cvdCandlesSummary = marketData.cvdCandles && marketData.cvdCandles.length > 0
    ? marketData.cvdCandles
        .slice(-10)  // Limit to last 10 candles to prevent huge prompts
        .map(c => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(1)}, H=${c.high.toFixed(1)}, L=${c.low.toFixed(1)}, C=${c.close.toFixed(1)}`)
        .join('\n  ')
    : 'Not provided';
  // Limit CVD candles JSON to last 20 to prevent prompt overflow
  const cvdCandlesJson = marketData.cvdCandles
    ? JSON.stringify(marketData.cvdCandles.slice(-20))
    : '[]';

  const bigPrints = marketData.orderFlow?.bigTrades
    ? marketData.orderFlow.bigTrades
        .slice(-5)
        .map((p) => `${p.side.toUpperCase()} ${p.size} @ ${p.price}`)
        .join(', ') || 'None'
    : 'None';

  const priceVsProfile = (marketData.volumeProfile?.poc !== undefined && currentPrice !== undefined)
    ? (currentPrice > marketData.volumeProfile.poc ? 'above POC' : 'below POC')
    : 'POC not available';

  // Microstructure summary
  const micro = marketData.microstructure;
  const whaleTradeSummary = micro && micro.largeWhaleTrades.length > 0
    ? micro.largeWhaleTrades
        .slice(-10)
        .map(t => `${new Date(t.timestamp).toLocaleTimeString()} ${t.side.toUpperCase()} ${t.size} @ ${t.price.toFixed(2)}`)
        .join('; ')
    : 'No large trades recorded';

  const restingLimitSummary = micro && micro.restingLimitOrders.length > 0
    ? micro.restingLimitOrders
        .slice(0, 10)
        .map(l => `P=${l.price.toFixed(2)} | Bid=${l.restingBid.toFixed(0)} | Ask=${l.restingAsk.toFixed(0)} | Seen=${new Date(l.lastSeen).toLocaleTimeString()}`)
        .join('; ')
    : 'No resting liquidity clusters detected';

  // Wall break/hold tracking for continuation vs reversal context
  const wallBreaks = micro?.recentWallBreaks || [];
  const wallHolds = micro?.recentWallHolds || [];
  const wallBreakSummary = wallBreaks.length > 0
    ? wallBreaks.map((w: any) => `${w.side.toUpperCase()} wall @ ${w.price.toFixed(2)} BROKEN (${w.brokenAgo})`).join('; ')
    : 'No recent wall breaks';
  const wallHoldSummary = wallHolds.length > 0
    ? wallHolds.map((w: any) => `${w.side.toUpperCase()} wall @ ${w.price.toFixed(2)} HELD (${w.heldAgo})`).join('; ')
    : 'No recent wall holds';

  // Macrostructure summary
  const macro = marketData.macrostructure;
  const multiDay = macro?.multiDayProfile;
  const higherTFs = macro?.higherTimeframes || [];
  // Only show raw delta values, not computed derivatives
  const flowSignals = marketData.flowSignals;
  const flowSignalsSummary = flowSignals
    ? `Delta 1m/5m: ${flowSignals.deltaLast1m ?? 'n/a'} / ${flowSignals.deltaLast5m ?? 'n/a'}`
    : 'Not provided';

  const multiDaySummary = multiDay
    ? `Lookback: ${multiDay.lookbackHours}h | POC=${multiDay.poc.toFixed(2)} VAH=${multiDay.vah.toFixed(2)} VAL=${multiDay.val.toFixed(2)} | High=${multiDay.high.toFixed(2)} Low=${multiDay.low.toFixed(2)}`
    : 'Not provided';

  const higherTfSummary = higherTFs.length > 0
    ? higherTFs
        .map(tf => {
          const recent = tf.candles.slice(-6);
          if (recent.length === 0) return `${tf.timeframe}: Not provided`;
          const candleSummary = recent
            .map(c => `${new Date(c.timestamp).toLocaleTimeString()}: ${c.open.toFixed(2)}/${c.high.toFixed(2)}/${c.low.toFixed(2)}/${c.close.toFixed(2)}`)
            .join(' | ');
          return `${tf.timeframe} (${recent.length} bars): ${candleSummary}`;
        })
        .join('\n  ')
    : 'Not provided';

  const recentProfiles = macro?.recentVolumeProfiles || [];
  const volumeProfileHistory = recentProfiles.length > 0
    ? recentProfiles
        .slice(-5)
        .map(profile => {
          const start = new Date(profile.sessionStart).toLocaleString();
          return `${start} → POC=${profile.poc.toFixed(2)} | VAH=${profile.vah.toFixed(2)} | VAL=${profile.val.toFixed(2)} | Range=${profile.sessionLow.toFixed(2)}-${profile.sessionHigh.toFixed(2)}`;
        })
        .join('\n  ')
    : 'Not provided';

return `
ROBBINS WORLD CUP SNAPSHOT
==========================

Symbol: ${marketData.symbol}
Time: ${marketData.timestamp}
Current Price: ${currentPrice}

${stateSummary}

=== HIGHER TIMEFRAME CANDLES (NARRATIVE & DIRECTION) ===

DAILY CANDLES (Last 6 days - Major Trend):
  ${dailySummary || 'Not enough data'}

4-HOUR CANDLES (Last 6 bars ≈ 24 hours - Intermediate Trend):
  ${fourHourSummary || 'Not enough data'}

=== LOWER TIMEFRAME CANDLES (ENTRY, SL, TP POSITIONING) ===

15-MINUTE CANDLES (Last 8 bars ≈ 2 hours - Local Structure):
  ${fifteenMinSummary || 'Not available'}

1-MINUTE CANDLES (Last 20 bars ≈ 20 minutes - Precise Entry/Exit):
  ${oneMinSummary}

=== ORDER FLOW & LIQUIDITY ===

Footprint Readings:
- CVD: ${marketData.cvd?.value !== undefined ? `${marketData.cvd.value.toFixed(2)} (${cvdTrend})` : 'n/a'}
- Delta: ${flowSignalsSummary}
- Whale Trades: ${whaleTradeSummary}
- Recent Large Prints: ${bigPrints}

LIQUIDITY WALLS (L2 - for SL/TP placement):
${restingLimitSummary}

WALL BREAK/HOLD TRACKING (Evidence for continuation vs reversal):
- Recent BREAKS (continuation momentum): ${wallBreakSummary}
- Recent HOLDS (reversal/support): ${wallHoldSummary}

CVD CANDLES (Session Flow):
${cvdCandlesSummary}

=== SESSION CONTEXT ===
Session High / Low: ${marketData.volumeProfile?.sessionHigh?.toFixed(2) ?? 'n/a'} / ${marketData.volumeProfile?.sessionLow?.toFixed(2) ?? 'n/a'}
Session Range: ${marketData.marketStats?.session_range_ticks?.toFixed(1) ?? 'n/a'} ticks

${marketData.watchZoneProfiles && marketData.watchZoneProfiles.length > 0 ? `WATCH-ZONE PROFILES (key levels to watch):
${marketData.watchZoneProfiles.slice(0, 3).map(z =>
  `- ${z.name} ${z.low.toFixed(2)}-${z.high.toFixed(2)} | zonePOC ${z.poc?.toFixed(2) ?? 'n/a'}`
).join('\n')}` : 'WATCH-ZONE PROFILES: none'}

=== ACCOUNT & PERFORMANCE ===
Balance: $${marketData.account.balance.toFixed(2)} | Position: ${marketData.account.position === 0 ? 'FLAT' : `${marketData.account.position > 0 ? 'LONG' : 'SHORT'} ${Math.abs(marketData.account.position)} contracts`} | Unrealized P&L: $${marketData.account.unrealizedPnL.toFixed(2)}
${marketData.performance ? `Recent Win Rate: ${(marketData.performance.win_rate * 100).toFixed(1)}%
Average P&L: $${marketData.performance.avg_pnl.toFixed(2)}
Trade Count: ${marketData.performance.trade_count}
Profit Factor: ${marketData.performance.profit_factor.toFixed(2)}
Avg Win: $${marketData.performance.avg_win.toFixed(2)}
Avg Loss: $${marketData.performance.avg_loss.toFixed(2)}` : 'No performance history yet'}

${marketData.openPosition
  ? `OPEN POSITION (MANAGE THIS FIRST):
  - Side: ${marketData.openPosition.side.toUpperCase()} ${marketData.openPosition.contracts} contract(s)
  - Entry: ${marketData.openPosition.entryPrice.toFixed(2)} @ ${marketData.openPosition.entryTime}
  - Stop: ${marketData.openPosition.stopLoss.toFixed(2)} | Target: ${marketData.openPosition.target.toFixed(2)}
  - Unrealized P&L: $${marketData.openPosition.unrealizedPnL.toFixed(2)}
  - Broker IDs → Stop: ${marketData.openPosition.stopOrderId ?? 'unknown'} | Target: ${marketData.openPosition.targetOrderId ?? 'unknown'}
  - Distance to Stop: ${marketData.openPosition.distanceToStopPoints.toFixed(2)} pts | Distance to Target: ${marketData.openPosition.distanceToTargetPoints.toFixed(2)} pts
  - Time in Trade: ${(marketData.openPosition.positionAgeSeconds / 60).toFixed(1)} min`
  : 'OPEN POSITION: None (flat).'}

=== HISTORICAL NOTES (Lessons Learned) ===
${marketData.historicalNotes.length > 0
  ? marketData.historicalNotes.map(n => `- [${n.context}] ${n.note}`).join('\n')
  : 'No historical notes yet'}

${marketData.openPosition ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ACTIVE POSITION MANAGEMENT MODE ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOUR ONLY JOB RIGHT NOW: MANAGE THE OPEN ${marketData.openPosition.side.toUpperCase()} POSITION

PRIMARY ANALYSIS FOCUS:
1. **STOP LOSS MANAGEMENT**: Should the stop be tightened/trailed based on new support/resistance, order flow, or profit protection?
   - Current stop: ${marketData.openPosition.stopLoss.toFixed(2)} (${marketData.openPosition.distanceToStopPoints.toFixed(2)} pts away)
   - Consider: Break-even moves, trailing to higher lows (long) or lower highs (short), defending against reversals

2. **TARGET MANAGEMENT**: Should the target be extended/reduced based on momentum, absorption, or structure?
   - Current target: ${marketData.openPosition.target.toFixed(2)} (${marketData.openPosition.distanceToTargetPoints.toFixed(2)} pts away)
   - Consider: Taking partials at resistance/support, extending if strong momentum, scaling out

3. **EXIT DECISION**: Should we exit now (set decision to opposite of position) or hold?
   - P&L: $${marketData.openPosition.unrealizedPnL.toFixed(2)}
   - Consider: Reversal signals, target hit, major resistance/support breach, adverse order flow

4. **TIMING & MONITORING**: What price levels or order flow conditions trigger the next adjustment?

CRITICAL RULES:
- Set \`decision\` to "HOLD" to keep position and manage brackets
- Set \`decision\` to "${marketData.openPosition.side === 'long' ? 'SELL' : 'BUY'}" ONLY to close/exit the position
- ALWAYS provide \`stopLoss\` and \`target\` values (even if unchanged) to reflect your desired bracket levels
- Explain ALL bracket changes in \`riskManagementReasoning\`
- DO NOT analyze new entry opportunities - focus 100% on managing this position
- Use \`plan\` to describe your monitoring approach for the next update

RESPOND WITH YOUR POSITION MANAGEMENT DECISION:
` : `
ANALYSIS REQUEST (NO OPEN POSITION):
- ALWAYS pick a direction (BUY or SELL) based on your best read of the tape. Express uncertainty through CONFIDENCE, not by avoiding a decision.
- Low confidence (50-60%): Choppy, uncertain, mixed signals. You still pick the most likely direction, but with low conviction.
- High confidence (75-95%): Clear setup, strong edge, aligned factors. You pick direction with high conviction.
- Provide entry price/zone, stop, target, and timing/invalidation details.
- Defend the stop & target placement using structure/order-flow context in riskManagementReasoning.
- Include add-on or re-entry logic if price overshoots or structure resets.
- Multiple trades per day are expected when the tape cooperates.
`}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 CRITICAL OUTPUT FORMAT REQUIREMENT 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU MUST OUTPUT **ONLY** VALID JSON. NO PLAIN TEXT. NO MARKDOWN OUTSIDE JSON. ONLY JSON.

DO NOT output format like this (WRONG ❌):
Trade Decision: BUY
Confidence: 60%

ONLY output valid JSON format like this (CORRECT ✅):
{
  "decision": "BUY",
  "confidence": 60,
  "stopLoss": 24905.25,
  "target": 24875.00,
  "riskRewardRatio": 1.5,
  "riskPercent": 0.35,
  "plan": "Enter BUY at VAL support with tight stop",
  "timingPlan": "Immediate market entry",
  "reEntryPlan": "Re-enter on bounce if stopped at VAL",
  "reasoning": "Strong bid absorption at VAL with POC magnet above",
  "riskManagementReasoning": "Stop at 24905.25 (below VAL structure), target at POC",
  "noteForFuture": null
}

REQUIRED JSON SCHEMA (all fields mandatory):
{
  "decision": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "stopLoss": null or number,
  "target": null or number,
  "riskRewardRatio": null or number,
  "riskPercent": number,
  "plan": "High-level summary of entry/stop/target/if-then contingencies",
  "timingPlan": "Describe when/how you intend to execute (immediate, on break, on pullback, etc.)",
  "reEntryPlan": "Describe add-on or re-entry logic if stopped/missed",
  "reasoning": "Tie together price action, footprint cues, volume profile, and lessons/performance",
  "riskManagementReasoning": "Specific explanation for WHY these SL/TP levels make sense (structure, liquidity, high-volume node, etc.). Required for all BUY/SELL decisions.",
  "noteForFuture": "Optional reminder or null"
}

Always include an actionable plan with specific entry/stop/target prices and timing details.

REMEMBER: OUTPUT ONLY THE JSON OBJECT. START WITH { AND END WITH }. NOTHING ELSE. 
`;
}

/**
 * Parse OpenAI's JSON response
 * Extracts all Fabio decision layers from the response
 */
function parseOpenAIResponse(content: string): OpenAITradingDecision {
  try {
    // Try multiple JSON extraction strategies
    let jsonStr: string | null = null;
    let parsed: any = null;

    // Strategy 1: Try extracting from markdown code blocks first (most specific)
    const markdownMatch = content.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (markdownMatch) {
      console.log('[parseOpenAIResponse] Found JSON in markdown code block');
      jsonStr = markdownMatch[1].trim();
    }

    // Strategy 2: Try finding JSON object (first { to matching })
    if (!jsonStr) {
      // Find the first { and then find its matching }
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
          console.log('[parseOpenAIResponse] Extracted JSON using brace matching');
        }
      }
    }

    // Strategy 3: Try greedy regex as last resort
    if (!jsonStr) {
      const greedyMatch = content.match(/\{[\s\S]*\}/);
      if (greedyMatch) {
        jsonStr = greedyMatch[0];
        console.log('[parseOpenAIResponse] Using greedy regex match');
      }
    }

    if (!jsonStr) {
      console.error('[parseOpenAIResponse] ❌ NO JSON FOUND IN RESPONSE');
      console.error('[parseOpenAIResponse] Raw content preview (first 500 chars):');
      console.error(content.substring(0, 500));
      console.error('[parseOpenAIResponse] Raw content preview (last 500 chars):');
      console.error(content.substring(Math.max(0, content.length - 500)));
      throw new Error('No JSON found in response');
    }

    // Extract reasoning text (everything before the JSON)
    let reasoningText = '';
    const jsonStartIndex = content.indexOf(jsonStr);
    if (jsonStartIndex > 0) {
      reasoningText = content.substring(0, jsonStartIndex).trim();
      // Clean up the reasoning text - remove template markers and markdown
      reasoningText = reasoningText
        .replace(/```json\s*$/gm, '')  // Remove trailing ```json markers
        .replace(/```\s*$/gm, '')       // Remove trailing ``` markers
        .replace(/DECISION:.*$/gm, '')
        .replace(/CONFIDENCE:.*$/gm, '')
        .replace(/ENTRY:.*$/gm, '')
        .replace(/STOP:.*$/gm, '')
        .replace(/TARGET:.*$/gm, '')
        .replace(/Trade Decision:.*$/gm, '')  // Remove "Trade Decision:" lines
        .replace(/Confidence:.*$/gm, '')       // Remove "Confidence:" lines
        .replace(/Then provide valid JSON.*$/gm, '')
        .trim();
      console.log('[parseOpenAIResponse] Extracted reasoning text (' + reasoningText.length + ' chars)');
    }

    // Try parsing the extracted JSON
    try {
      parsed = JSON.parse(jsonStr);
      console.log('[parseOpenAIResponse] ✅ Successfully parsed JSON');
    } catch (parseError: any) {
      console.error('[parseOpenAIResponse] ❌ JSON.parse failed:', parseError.message);
      console.error('[parseOpenAIResponse] Attempted to parse:');
      console.error(jsonStr);

      try {
        parsed = JSON.parse(jsonrepair(jsonStr));
        console.log('[parseOpenAIResponse] ✅ JSON repaired via jsonrepair()');
      } catch (repairError: any) {
        console.error('[parseOpenAIResponse] ❌ jsonrepair() failed:', repairError?.message || repairError);
        throw new Error(`JSON parse error: ${parseError.message}`);
      }
    }
    const decision = typeof parsed.decision === 'string'
      ? parsed.decision.toUpperCase()
      : 'HOLD';

    // Validate and parse market state
    const validMarketStates = [
      'balanced',
      'out_of_balance_uptrend',
      'out_of_balance_downtrend',
      'balanced_with_failed_breakout_above',
      'balanced_with_failed_breakout_below'
    ];
    const marketState = validMarketStates.includes(parsed.marketState)
      ? (parsed.marketState as MarketState)
      : 'balanced';

    // Validate and parse location
    const validLocations = ['at_poc', 'at_vah', 'at_val', 'at_lvn', 'above_poc', 'below_poc'];
    const location = validLocations.includes(parsed.location)
      ? parsed.location
      : 'at_poc';

    // Validate and parse setup model
    const validSetupModels = ['trend_continuation', 'mean_reversion'];
    const setupModel = parsed.setupModel && validSetupModels.includes(parsed.setupModel)
      ? (parsed.setupModel as SetupModel)
      : null;

    // Parse risk percent (0.25-0.5 per Fabio rules)
    const riskPercent = Math.min(0.5, Math.max(0.25, parsed.riskPercent || 0.35));

    // Parse regime inference
    const validRegimes = ['trend', 'range', 'chop'];
    const inferredRegime = parsed.inferredRegime && validRegimes.includes(parsed.inferredRegime)
      ? (parsed.inferredRegime as 'trend' | 'range' | 'chop')
      : undefined;

    const regimeConfidence = parsed.regimeConfidence
      ? Math.min(100, Math.max(0, parsed.regimeConfidence))
      : undefined;

    const noteForFuture = parsed.noteForFuture || undefined;

    const timingPlan = parsed.timingPlan || '';
    const reEntryPlan = parsed.reEntryPlan || '';
    const stitchedPlan = parsed.plan
      || [timingPlan ? `Timing: ${timingPlan}` : '', reEntryPlan ? `Re-entry: ${reEntryPlan}` : '']
          .filter(Boolean)
          .join(' | ')
      || 'No plan provided';

    const rawRiskReasoning = typeof parsed.riskManagementReasoning === 'string'
      ? parsed.riskManagementReasoning.trim()
      : '';

    if ((decision === 'BUY' || decision === 'SELL') && !rawRiskReasoning) {
      throw new Error('riskManagementReasoning is required for BUY/SELL decisions');
    }

    // Use extracted reasoning text from the response, fall back to JSON reasoning field, then to default
    let finalReasoning = reasoningText || parsed.reasoning || '';
    if (!finalReasoning) {
      finalReasoning = decision === 'HOLD'
        ? 'Market conditions unclear - no high-probability setup identified. Waiting for better risk/reward opportunity.'
        : 'No reasoning provided';
    }

    // Ensure confidence is a valid number (default to 0 for HOLD, 50 for trades)
    const defaultConfidence = decision === 'HOLD' ? 0 : 50;
    const confidence = typeof parsed.confidence === 'number'
      ? Math.min(100, Math.max(0, parsed.confidence))
      : defaultConfidence;

    return {
      decision,
      confidence,
      marketState,
      location,
      setupModel,
      entryPrice: parsed.entryPrice || null,
      stopLoss: parsed.stopLoss || null,
      target: parsed.target || null,
      reasoning: finalReasoning,
      plan: stitchedPlan,
      timingPlan: timingPlan || undefined,
      reEntryPlan: reEntryPlan || undefined,
      riskRewardRatio: parsed.riskRewardRatio || null,
      riskPercent,
      riskManagementReasoning: rawRiskReasoning || undefined,
      inferredRegime,
      regimeConfidence,
      noteForFuture,
    };
  } catch (error: any) {
    console.error('='.repeat(80));
    console.error('[parseOpenAIResponse] ❌ CRITICAL: Failed to parse OpenAI response');
    console.error('='.repeat(80));
    console.error('[parseOpenAIResponse] Error:', error?.message || error);
    console.error('[parseOpenAIResponse] Full error stack:', error?.stack || 'No stack trace');
    console.error('[parseOpenAIResponse] Raw content length:', content?.length || 0);
    console.error('[parseOpenAIResponse] Raw content preview (first 1000 chars):');
    console.error(content?.substring(0, 1000) || 'No content');
    console.error('[parseOpenAIResponse] Raw content preview (last 1000 chars):');
    console.error(content?.substring(Math.max(0, (content?.length || 0) - 1000)) || 'No content');
    console.error('='.repeat(80));
    console.error('[parseOpenAIResponse] Returning fallback HOLD decision with confidence 0%');
    console.error('='.repeat(80));

    return {
      decision: 'HOLD',
      confidence: 0,
      marketState: 'balanced',
      location: 'at_poc',
      setupModel: null,
      entryPrice: null,
      stopLoss: null,
      target: null,
      reasoning: `Failed to parse AI response: ${error?.message || 'Unknown error'}`,
      plan: 'Unable to parse response - waiting for next analysis',
      riskRewardRatio: null,
      riskPercent: 0.35,
    };
  }
}

function resolveOpenAIModel(): string {
  return process.env.OPENAI_MODEL?.trim() || DEFAULT_OPENAI_MODEL;
}

function resolveOpenAIBaseURL(): string {
  return process.env.OPENAI_BASE_URL?.trim() || 'https://api.deepseek.com';
}

function shouldUseReasoningModel(model: string): boolean {
  return model.toLowerCase().includes('reasoner');
}

function resolveReasoningEffort(): ReasoningEffort {
  const env = process.env.OPENAI_REASONING_EFFORT?.toLowerCase() as ReasoningEffort | undefined;
  if (env && VALID_REASONING_EFFORTS.includes(env)) {
    return env;
  }
  return 'medium';
}

function shouldRetryReasoner(error: any): boolean {
  if (!error) return false;
  if (isSocketTerminationError(error)) {
    return true;
  }
  const message = typeof error === 'string' ? error : error.message || '';
  return /Reasoner returned empty response/i.test(message);
}

function isSocketTerminationError(error: any): boolean {
  if (!error) return false;
  const message = typeof error === 'string' ? error : error.message || '';
  const code = (error as any)?.code || (error as any)?.cause?.code;
  return /UND_ERR_SOCKET/i.test(message) || /socket hang up/i.test(message) || code === 'UND_ERR_SOCKET';
}

function extractTextPart(part: any): string {
  if (!part) return '';
  if (typeof part === 'string') return part;
  if (typeof part.text === 'string') return part.text;
  if (Array.isArray(part)) {
    return part.map(chunk => extractTextPart(chunk)).join('');
  }
  if (typeof part === 'object') {
    return Object.values(part)
      .map(value => (typeof value === 'string' ? value : ''))
      .join('');
  }
  return '';
}
function extractMessageText(message: any): string {
  if (!message) return '';
  const { content } = message;
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    return content.map(part => extractTextPart(part)).join('');
  }
  return '';
}

function extractReasoningFromMessage(message: any): string {
  if (!message || !Array.isArray(message.reasoning_content)) {
    return '';
  }
  return message.reasoning_content.map((part: any) => extractTextPart(part)).join('').trim();
}

/**
 * Continuous analysis loop - analyze market every N seconds
 */
export async function startContinuousAnalysis(
  getMarketData: () => Promise<FuturesMarketData>,
  onDecision: (decision: OpenAITradingDecision) => void,
  intervalMs: number = 60000 // Analyze every 60 seconds
) {
  console.log(`🤖 Starting OpenAI futures analysis loop (every ${intervalMs}ms)`);

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
    console.log('🛑 Stopped OpenAI analysis loop');
  };
}
