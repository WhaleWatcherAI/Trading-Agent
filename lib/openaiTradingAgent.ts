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

type ResponseOutputItem = OpenAIResponse['output'][number];

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

    const prompt = buildAnalysisPrompt(marketData, recentCandles, currentPrice);

    console.log(`📊 [OpenAI] Analyzing ${marketData.symbol} at ${currentPrice}`);
    console.log('='.repeat(80));
    console.log('[DEBUG] EXACT PROMPT SENT TO DEEPSEEK:');
    console.log('='.repeat(80));
    console.log(prompt);
    console.log('='.repeat(80));

    const systemInstructions = `You are Fabio, a multi-time Robbins World Cup Champion trading NQ futures with instinct, aggression, and relentless preparation.

Mindset:
- Show up clean every session. Use only today's order flow plus your self-learning database (performance metrics + historical notes). No rigid playbooks.
- Hunt constantly. Even when flat you outline immediate entry zones, precise stops, targets, and \"if price does X, then I do Y\" contingencies.
- Expect to execute multiple high-quality trades per day when the tape permits. Do not sit out waiting for perfection—shape risk intelligently and act.
- Treat performance stats and historical notes as your trophy room: reference them to repeat what works and avoid past mistakes.

Toolbox & Authority:
- Blend market structure, higher-timeframe auction context, volume profile references (POC/VAH/VAL/LVNs), and classical technical analysis whenever useful.
- Level 2 / order-flow signals (CVD, major prints, resting liquidity) are additive context—not the only driver—so balance them with structure and trend.
- You may apply any smart-money concept or pretrained knowledge that fits the current tape; you are not restricted to predefined playbooks.

Execution Consistency:
- The JSON decision you return is the literal trading instruction. If your reasoning or plan calls for BUY/SELL, the JSON must also say BUY/SELL with concrete entry/stop/target numbers.
- Only return HOLD when you truly have no trade and explain the missing ingredient plus the trigger that would flip you to BUY/SELL.
- Never emit text like \"Trade Decision: BUY\" while the JSON decision is HOLD; they must always match.
- Be mindful of stop hunts: consider resting liquidity/swing placement when choosing stops to reduce sweep risk, but stay flexible to the tape.

Philosophy: Read the multi-timeframe auction, interpret footprint clues (CVD, whale flows, resting liquidity), and strike when reward-to-risk skews in your favor. You are here to execute, not theorize.

Expectations:
- Deliver an actionable trade plan (entry price/zone, stop, target, timing) whenever opportunity exists now or within the next few bars.
- BUY/SELL calls must detail stop & target placement logic inside riskManagementReasoning, referencing market structure, liquid pockets, or volume nodes.
- When an openPosition is provided, treat it as an active trade: confirm whether to keep, scale, or exit, proactively tighten/relax stops or targets based on the latest tape, and keep the JSON stopLoss/target fields in sync with the bracket you want live.
- If you must HOLD, explain the missing ingredient and the precise trigger that would unlock a trade soon.
- Risk 0.25%-0.5% per idea, place stops just beyond structure, never widen, and be ready to re-enter if the setup resets.

Respond ONLY in JSON with the required fields for execution.`;

    const messages = [
      {
        role: 'system' as const,
        content: systemInstructions,
      },
      {
        role: 'user' as const,
        content: prompt,
      },
    ];

    const model = resolveOpenAIModel();
  const useReasoner = shouldUseReasoningModel(model);
  const responseFormat = useReasoner ? undefined : { type: 'json_object' as const };
  let content: string | null = null;
  const reasoningEffort = resolveReasoningEffort();

  if (useReasoner) {
      const reasonerAttempts: Array<{ stream: boolean; maxTokens: number; temperature: number }> = [
        { stream: false, maxTokens: 3200, temperature: 0.35 },
        { stream: false, maxTokens: 2400, temperature: 0.35 },
      ];

      for (let i = 0; i < reasonerAttempts.length; i += 1) {
        const attempt = reasonerAttempts[i];
        try {
          if (attempt.stream) {
            const stream = await openai.chat.completions.create({
              model,
              messages,
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
              messages,
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
          }

          if (content && content.trim()) {
            break;
          }
          throw new Error('Reasoner returned empty response');
        } catch (error: any) {
          console.warn(`[OpenAI] Reasoner attempt ${i + 1} failed: ${error?.message || error}`);
          const shouldRetry = shouldRetryReasoner(error) && i < reasonerAttempts.length - 1;
          if (!shouldRetry) {
            throw error;
          }
          console.warn('[OpenAI] Retrying reasoner with alternate delivery (non-stream).');
        }
      }
    } else {
      const response = await openai.chat.completions.create({
        model,
        temperature: 0.7,
        max_tokens: 1000,
        messages,
        ...(responseFormat ? { response_format: responseFormat } : {}),
      });
      content = extractMessageText(response.choices[0]?.message);
    }

    if (!content) {
      throw new Error('No content generated by OpenAI response');
    }
    console.log('='.repeat(80));
    console.log('[DEBUG] RAW DEEPSEEK RESPONSE:');
    console.log('='.repeat(80));
    console.log(content);
    console.log('='.repeat(80));
    const decision = parseOpenAIResponse(content);

    console.log(`✅ [OpenAI] ${decision.decision} @ ${decision.entryPrice} | Confidence: ${decision.confidence}%`);

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

  // Build multi-timeframe candle summaries from 5-minute base data
  const allCandles = marketData.candles;

  // 1-minute candles (last 20 candles = 20 minutes)
  const oneMinCandles = allCandles.slice(-20);
  const oneMinSummary = oneMinCandles
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, C=${c.close.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}`)
    .join('\n  ');

  // 15-minute candles (aggregate minute bars, last 8 candles = 2 hours)
  const fifteenMinCandles = aggregateCandlesToTimeframe(allCandles, 15, 1);
  const fifteenMinSummary = fifteenMinCandles.slice(-8)
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(2)}, C=${c.close.toFixed(2)}, H=${c.high.toFixed(2)}, L=${c.low.toFixed(2)}`)
    .join('\n  ');

  const cvdCandlesSummary = marketData.cvdCandles.length > 0
    ? marketData.cvdCandles
        .map(c => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open.toFixed(1)}, H=${c.high.toFixed(1)}, L=${c.low.toFixed(1)}, C=${c.close.toFixed(1)}`)
        .join('\n  ')
    : 'Not provided';
  const cvdCandlesJson = JSON.stringify(marketData.cvdCandles);

  const bigPrints = marketData.orderFlow.bigTrades
    .slice(-5)
    .map((p) => `${p.side.toUpperCase()} ${p.size} @ ${p.price}`)
    .join(', ') || 'None';

  const priceVsProfile = currentPrice > marketData.volumeProfile.poc ? 'above POC' : 'below POC';

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

  // Macrostructure summary
  const macro = marketData.macrostructure;
  const multiDay = macro?.multiDayProfile;
  const higherTFs = macro?.higherTimeframes || [];

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

=== PRICE & FOOTPRINT SNAPSHOT ===

1-MINUTE CANDLES (Last 20 bars ≈ 20 minutes):
${oneMinSummary}

15-MINUTE CANDLES (Last 8 bars ≈ 2 hours):
${fifteenMinSummary || 'Not available'}

=== HIGHER TIMEFRAME CONTEXT ===
${higherTfSummary}

RECENT SESSION PROFILES (Last 5 trading days):
${volumeProfileHistory}

Footprint Readings:
- CVD Trend: ${cvdTrend} | Value: ${marketData.cvd.value.toFixed(2)} | Structure (OHLC): O=${marketData.cvd.ohlc.open}, H=${marketData.cvd.ohlc.high}, L=${marketData.cvd.ohlc.low}, C=${marketData.cvd.ohlc.close}
- Whale Trades: ${whaleTradeSummary}
- Resting Limit Orders: ${restingLimitSummary}
- Recent Large Prints: ${bigPrints}

CVD CANDLES (Session Sample):
${cvdCandlesSummary}

Full CVD Candles JSON (session start → now):
${cvdCandlesJson}

=== VOLUME PROFILE & AUCTION METRICS ===
POC: ${marketData.volumeProfile.poc} | VAH: ${marketData.volumeProfile.vah} | VAL: ${marketData.volumeProfile.val}
LVNs: ${marketData.volumeProfile.lvns.join(', ')}
Session High / Low: ${marketData.volumeProfile.sessionHigh} / ${marketData.volumeProfile.sessionLow}
Current Price vs POC: ${priceVsProfile}
Distance to POC: ${marketData.marketStats.distance_to_poc_ticks.toFixed(1)} ticks
Session Range: ${marketData.marketStats.session_range_ticks.toFixed(1)} ticks (${(marketData.marketStats.session_range_percentile * 100).toFixed(0)}th percentile)
Time Above / In / Below Value: ${marketData.marketStats.time_above_value_sec.toFixed(0)}s / ${marketData.marketStats.time_in_value_sec.toFixed(0)}s / ${marketData.marketStats.time_below_value_sec.toFixed(0)}s
POC Crosses (last 5 / 15 / 30 min): ${marketData.pocCrossStats.count_last_5min} / ${marketData.pocCrossStats.count_last_15min} / ${marketData.pocCrossStats.count_last_30min}
Time Since Last Cross: ${marketData.pocCrossStats.time_since_last_cross_sec.toFixed(1)}s | Current Side: ${marketData.pocCrossStats.current_side}

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
- Identify the best trade(s) available RIGHT NOW or within the next few bars.
- Provide direction (BUY/SELL/HOLD), confidence (0-100%), entry price/zone, stop, target, and timing/invalidation details.
- When calling BUY/SELL, defend the stop & target placement using structure/order-flow context and surface that explanation in riskManagementReasoning.
- Include add-on or re-entry logic if price overshoots or structure resets.
- Multiple trades per day are expected when the tape cooperates. If you stay on HOLD, state exactly what must happen (price level, order-flow cue, time event) to trigger a trade soon.
`}

Respond **only** with valid JSON conforming to this schema:
{
  "decision": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "entryPrice": null or number,
  "stopLoss": null or number,
  "target": null or number,
  "riskRewardRatio": null or number,
  "riskPercent": number,
  "plan": "High-level summary of entry/stop/target/if-then contingencies",
  "timingPlan": "Describe when/how you intend to execute (immediate, on break, on pullback, etc.)",
  "reEntryPlan": "Describe add-on or re-entry logic if stopped/missed",
  "reasoning": "Tie together price action, footprint cues, volume profile, and lessons/performance",
  "riskManagementReasoning": "Specific explanation for WHY these SL/TP levels make sense (structure, liquidity, high-volume node, etc.). Required when decision is BUY/SELL.",
  "noteForFuture": "Optional reminder or null"
}

Always include an actionable plan even for HOLD decisions. 
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

    return {
      decision,
      confidence: Math.min(100, Math.max(0, parsed.confidence || 50)),
      marketState,
      location,
      setupModel,
      entryPrice: parsed.entryPrice || null,
      stopLoss: parsed.stopLoss || null,
      target: parsed.target || null,
      reasoning: parsed.reasoning || 'No reasoning provided',
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
