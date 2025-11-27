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

    const systemInstructions = `You are Fabio, a multi-time Robbins World Cup champion trading NQ/MNQ futures. You read the tape, identify direction from higher timeframes, and execute with precision using order flow confirmation.

Core Framework
1. DIRECTION FROM HIGHER TIMEFRAMES: Look at DAILY and 4-HOUR candles first. Where does price WANT to go? Higher highs/lows = bullish. Lower highs/lows = bearish. This sets your directional bias. Then use 15-min for timing.

2. WHO'S IN CONTROL? Buyers or sellers?
   - CVD trending up = buyers in control
   - CVD trending down = sellers in control
   - Delta surges show aggressive participation
   - Whale prints show institutional activity
   - ABSORPTION: Buyers/sellers defending a level (potential reversal zone)
   - EXHAUSTION: Buyers/sellers running out of steam (trend may pause/reverse)

3. L2 LIQUIDITY WALLS (Critical for entries/stops):
   - Large resting bid walls = potential support (reversals: enter long, stop below wall)
   - Large resting ask walls = potential resistance (reversals: enter short, stop above wall)
   - Wall BREAK = continuation signal (price absorbed the wall and pushed through)
   - Set stops BEHIND the wall you're trading against

4. EXECUTION:
   - Reversals: Trade at walls with stop behind the wall
   - Continuations: Trade wall breaks with stop at the broken wall
   - Tight stops, let winners run to next wall/structure

Mindset
- ALWAYS pick a direction (BUY or SELL). Express uncertainty through confidence level (50-65% = weak edge, 70-85% = good edge, 85%+ = strong edge).
- Higher timeframe direction + order flow alignment + L2 wall = high confidence
- Counter-trend trades need strong absorption + L2 wall support

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
 * Build a simplified market analysis prompt focused on:
 * 1. Higher timeframe direction (ACTUAL 4H and 1D candles)
 * 2. Buyer/seller control (CVD, delta, absorption, exhaustion)
 * 3. L2 liquidity walls
 * 4. Whale prints
 */
function buildAnalysisPrompt(
  marketData: FuturesMarketData,
  recentCandles: any[],
  currentPrice: number
): string {
  const cvdTrend = marketData.cvd.trend === 'up' ? '🟢 BUYERS IN CONTROL' :
                   marketData.cvd.trend === 'down' ? '🔴 SELLERS IN CONTROL' : '⚪ NEUTRAL/BALANCED';

  // Build multi-timeframe candle summaries
  const allCandles = marketData.candles;

  // 1-minute candles (last 15 candles) - IMMEDIATE
  const oneMinCandles = allCandles.slice(-15);
  const oneMinSummary = oneMinCandles
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
    .join('\n  ');

  // Use ACTUAL higher timeframe data from macrostructure (4H and 1D candles)
  const higherTFs = marketData.macrostructure?.higherTimeframes || [];

  // Find 4H candles (240m timeframe)
  const fourHourTF = higherTFs.find(tf => tf.timeframe === '240m');
  const fourHourSummary = fourHourTF && fourHourTF.candles.length > 0
    ? fourHourTF.candles.slice(-8)
        .map((c) => `${new Date(c.timestamp).toLocaleString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
        .join('\n  ')
    : 'Not available';

  // Find Daily candles (1d timeframe)
  const dailyTF = higherTFs.find(tf => tf.timeframe === '1d');
  const dailySummary = dailyTF && dailyTF.candles.length > 0
    ? dailyTF.candles.slice(-10)
        .map((c) => `${new Date(c.timestamp).toLocaleDateString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
        .join('\n  ')
    : 'Not available';

  // 15-minute candles (aggregated from 1-min) - INTERMEDIATE
  const fifteenMinCandles = aggregateCandlesToTimeframe(allCandles, 15, 1);
  const fifteenMinSummary = fifteenMinCandles.slice(-8)
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}, C=${c.close.toFixed(2)}`)
    .join('\n  ');

  // Whale prints - institutional activity
  const micro = marketData.microstructure;
  const whaleTradeSummary = micro && micro.largeWhaleTrades.length > 0
    ? micro.largeWhaleTrades
        .slice(-10)
        .map(t => `${new Date(t.timestamp).toLocaleTimeString()} ${t.side.toUpperCase()} ${t.size} @ ${t.price.toFixed(2)}`)
        .join('\n  ')
    : 'No whale prints detected';

  // L2 Liquidity Walls - CRITICAL for entries/stops
  const restingLimitSummary = micro && micro.restingLimitOrders.length > 0
    ? micro.restingLimitOrders
        .slice(0, 10)
        .map(l => {
          const wallType = l.restingBid > l.restingAsk ? '🟢 BID WALL (support)' :
                          l.restingAsk > l.restingBid ? '🔴 ASK WALL (resistance)' : '⚪ BALANCED';
          return `${l.price.toFixed(2)} | Bid: ${l.restingBid.toFixed(0)} | Ask: ${l.restingAsk.toFixed(0)} → ${wallType}`;
        })
        .join('\n  ')
    : 'No significant liquidity walls detected';

  // Nearest wall in trade direction
  const nearestWall = micro?.nearestRestingWallInDirection;
  const nearestWallInfo = nearestWall
    ? `${nearestWall.side.toUpperCase()} wall @ ${nearestWall.price.toFixed(2)} (${nearestWall.size} contracts, ${nearestWall.distance} ticks away)`
    : 'No significant wall in direction';

  // Wall break detection
  const wallBreakInfo = micro?.liquidityPullDetected
    ? '⚠️ LIQUIDITY PULL DETECTED - wall being absorbed'
    : micro?.weakWallDetected
    ? '⚠️ WEAK WALL DETECTED - may break soon'
    : 'No wall break signals';

  // Flow signals
  const flowSignals = marketData.flowSignals;
  const deltaInfo = flowSignals
    ? `1min: ${flowSignals.deltaLast1m ?? 'n/a'} | 5min: ${flowSignals.deltaLast5m ?? 'n/a'}`
    : 'Delta not available';

  // Session structure
  const sessionHigh = marketData.volumeProfile?.sessionHigh ?? 0;
  const sessionLow = marketData.volumeProfile?.sessionLow ?? 0;
  const poc = marketData.volumeProfile?.poc ?? 0;

  // Absorption & Exhaustion signals
  const absorptionSignals = marketData.absorption || [];
  const exhaustionSignals = marketData.exhaustion || [];

  const absorptionSummary = absorptionSignals.length > 0
    ? absorptionSignals.slice(0, 5).map(a =>
        `${a.side.toUpperCase()} absorption at ${a.levelName} (${a.price.toFixed(2)}) - strength: ${(a.strength * 100).toFixed(0)}%${a.confirmedByCvd ? ' ✓CVD confirmed' : ''}`
      ).join('\n  ')
    : 'No absorption signals detected';

  const exhaustionSummary = exhaustionSignals.length > 0
    ? exhaustionSignals.slice(0, 5).map(e =>
        `${e.side.toUpperCase()} exhaustion at ${e.levelName} (${e.price.toFixed(2)}) - strength: ${(e.strength * 100).toFixed(0)}%`
      ).join('\n  ')
    : 'No exhaustion signals detected';

return `
TAPE READING SNAPSHOT
=====================

Symbol: ${marketData.symbol}
Time: ${marketData.timestamp}
Current Price: ${currentPrice}

=== 1. HIGHER TIMEFRAME DIRECTION (Where does price want to go?) ===

DAILY CANDLES (Last 10 days) - MACRO TREND:
  ${dailySummary}

4-HOUR CANDLES (Last 8 bars = 32 hours) - SWING:
  ${fourHourSummary}

15-MINUTE CANDLES (Last 8 bars = 2 hours) - INTERMEDIATE:
  ${fifteenMinSummary || 'Not available'}

1-MINUTE CANDLES (Last 15 bars) - IMMEDIATE:
  ${oneMinSummary}

Session High: ${sessionHigh.toFixed(2)} | Session Low: ${sessionLow.toFixed(2)} | POC: ${poc.toFixed(2)}

=== 2. WHO'S IN CONTROL? (Buyers or Sellers) ===

CVD (Cumulative Volume Delta): ${marketData.cvd?.value !== undefined ? marketData.cvd.value.toFixed(2) : 'n/a'} → ${cvdTrend}
Delta: ${deltaInfo}

ABSORPTION (Buyers/Sellers holding levels):
  ${absorptionSummary}

EXHAUSTION (Buyers/Sellers running out of steam):
  ${exhaustionSummary}

Whale Prints (Large Institutional Trades):
  ${whaleTradeSummary}

=== 3. L2 LIQUIDITY WALLS (Support/Resistance for entries & stops) ===

Nearest Wall in Direction: ${nearestWallInfo}
Wall Break Status: ${wallBreakInfo}

Resting Liquidity by Price Level:
  ${restingLimitSummary}

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
1. **STOP LOSS MANAGEMENT**: Should the stop be tightened/trailed based on L2 walls or order flow?
   - Current stop: ${marketData.openPosition.stopLoss.toFixed(2)} (${marketData.openPosition.distanceToStopPoints.toFixed(2)} pts away)
   - Move stop behind new bid walls (long) or ask walls (short) as they form

2. **TARGET MANAGEMENT**: Should target be extended based on momentum toward next wall?
   - Current target: ${marketData.openPosition.target.toFixed(2)} (${marketData.openPosition.distanceToTargetPoints.toFixed(2)} pts away)
   - Extend if wall breaks and next wall is further

3. **EXIT DECISION**: Should we exit now?
   - P&L: $${marketData.openPosition.unrealizedPnL.toFixed(2)}
   - Exit if CVD flips against position or major wall breaks against us

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
1. CHECK HTF DIRECTION: Look at 60-min and 15-min candles. Higher highs/lows = bullish. Lower highs/lows = bearish.
2. CHECK WHO'S IN CONTROL: CVD trend up = buyers. CVD trend down = sellers. Delta surges confirm.
3. FIND L2 WALL FOR ENTRY:
   - For LONG: Find bid wall (support) to enter at, set stop BELOW the wall
   - For SHORT: Find ask wall (resistance) to enter at, set stop ABOVE the wall
4. SET TARGET: Next significant wall in trade direction
5. WALL BREAKS = CONTINUATION: If price absorbed a wall and pushed through, trade continuation with stop at broken wall.

Express uncertainty through CONFIDENCE (50-60% weak, 75-95% strong). ALWAYS pick a direction.
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
  "confidence": 75,
  "stopLoss": 24905.25,
  "target": 24965.00,
  "riskRewardRatio": 1.5,
  "riskPercent": 0.35,
  "plan": "Buy at bid wall support, stop below wall, target next ask wall",
  "timingPlan": "Immediate market entry",
  "reEntryPlan": "Re-enter if price retests bid wall and holds",
  "reasoning": "Daily/4H showing higher lows, CVD bullish with BID absorption at 24910, large bid wall providing support",
  "riskManagementReasoning": "Stop at 24905.25 (below bid wall), target at ask wall 24965",
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
  "reasoning": "Explain HTF direction + who's in control (CVD/delta) + L2 wall setup",
  "riskManagementReasoning": "Explain stop placement relative to L2 walls and target based on next wall. Required for BUY/SELL.",
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
