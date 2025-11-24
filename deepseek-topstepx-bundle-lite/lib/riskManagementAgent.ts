/**
 * Risk Management Agent
 * Secondary AI agent dedicated to managing open positions
 * Focuses on profit maximization and risk minimization through dynamic bracket adjustments
 */

import OpenAI from 'openai';
import { ActivePosition } from './executionManager';
import { TopstepXFuturesBar } from './topstepx';
import { buildDeepseekCacheOptions } from './deepseekCache';

// Hard floor for protective behavior to prevent giving back winners
const BREAKEVEN_TRIGGER_POINTS = 4; // move stop to entry once this profit is reached
const LOCK_PROFIT_TRIGGER_POINTS = 10; // start locking in profit once this profit is reached
const LOCK_PROFIT_AMOUNT_POINTS = 5; // amount of profit to lock once trigger hit
const TARGET_PROX_TRIGGER = 0.8; // when 80%+ of the way to target, lock most profit
const TARGET_LOCK_SHARE = 0.7; // lock at least 70% of open profit when near target
const STOP_SAFETY_GAP = 0.5; // small gap off current price to avoid immediate stop-out
const GRADUAL_POINTS = [3, 6, 9]; // curve waypoints for aggression scaling

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL?.trim() || 'https://api.deepseek.com',
  timeout: 180000, // Give DeepSeek Reasoner enough time
});

export interface RiskManagementDecision {
  action: 'HOLD_BRACKETS' | 'ADJUST_STOP' | 'ADJUST_TARGET' | 'ADJUST_BOTH' | 'CLOSE_POSITION';
  newStopLoss: number | null;
  newTarget: number | null;
  reasoning: string;
  urgency: 'low' | 'medium' | 'high'; // How urgently this should be executed
  riskLevel: 'conservative' | 'balanced' | 'aggressive';
  positionVersion?: number;
}

interface MarketSnapshot {
  currentPrice: number;
  recentBars: TopstepXFuturesBar[];
  cvd: number;
  cvdTrend: 'up' | 'down' | 'neutral';
  orderFlowPressure: 'bullish' | 'bearish' | 'neutral';
  volumeProfile?: {
    poc: number;
    vah: number;
    val: number;
  };
  whaleActivity?: string;
  marketStructure?: string;
  // Optional microstructure and computed feature fields
  distToPoc?: number;
  distToVah?: number;
  distToVal?: number;
  distToNextHvn?: number;
  distToNextLvn?: number;
  distToSessionHigh?: number;
  distToSessionLow?: number;
  distToRoundNumber?: number;
  distToNearestHvnAbove?: number;
  distToNearestHvnBelow?: number;
  distToNearestLvnAbove?: number;
  distToNearestLvnBelow?: number;
  singlePrintZoneNearby?: { low: number; high: number } | null;
  inSinglePrintZone?: boolean;
  deltaLast1m?: number;
  deltaLast5m?: number;
  cvdSlopeShort?: number;
  cvdSlopeLong?: number;
  cvdDivergence?: 'none' | 'weak' | 'strong';
  absorptionZone?: { side: 'bid' | 'ask'; price: number; strength: number } | null;
  exhaustionFlag?: { side: 'bid' | 'ask'; strength: number } | null;
  largePrints?: Array<{ side: 'bid' | 'ask'; size: number; price: number }>;
  restingLiquidityWalls?: Array<{ side: 'bid' | 'ask'; price: number; size: number }>;
  liquidityPullDetected?: boolean;
  atr1m?: number;
  atr5m?: number;
  volRegime?: 'low' | 'normal' | 'high';
  currentRangeVsAtr?: number;
  structureState?: 'trend_up' | 'trend_down' | 'range' | 'breakout' | 'failed_breakout' | 'chop';
  invalidationPrice?: number;
  lastSwingPrice?: number;
  trendStrength?: number;
  PcEstimate?: number;
  PrEstimate?: number;
}

/**
 * Analyze position and recommend risk management adjustments
 */
export async function analyzePositionRisk(
  position: ActivePosition,
  market: MarketSnapshot
): Promise<RiskManagementDecision> {
  console.log(`[RiskMgmt] 🎯 Analyzing position risk for ${position.side.toUpperCase()} position...`);

  const profitLoss = market.currentPrice - position.entryPrice;
  const profitLossPoints = position.side === 'long' ? profitLoss : -profitLoss;
  const profitLossPercent = (profitLossPoints / position.entryPrice) * 100;

  const distanceToStop = position.side === 'long'
    ? position.entryPrice - position.stopLoss
    : position.stopLoss - position.entryPrice;

  const distanceToTarget = position.side === 'long'
    ? position.target - position.entryPrice
    : position.entryPrice - position.target;

  const percentToTarget = (profitLossPoints / distanceToTarget) * 100;
  const riskRewardRatio = distanceToTarget / distanceToStop;

  const prompt = `You are a RISK MANAGEMENT & TRADE OPTIMIZATION SPECIALIST for futures trading.
You operate at a Robbins World Cup / hedge-fund desk level.
You manage already-open positions from other traders.
Your mandate is capital protection first, EV-maximizing profit management second.
Your Core Mission
Your job is to maximize expected value (EV) of the open trade under changing conditions:

Cut losses fast when new market data materially weakens the trade’s edge.
Lock in profit efficiently as reversal risk rises or remaining reward shrinks.
Allow breathing room when continuation edge remains strong.
Near TP, tighten aggressively if remaining reward is small relative to profit at risk.
If continuation edge increases, you may tighten stop to lock profit AND extend TP to capture more trend — only when confidence and structure support it.
You are NOT a rigid trailing-stop rules bot.
You are a probabilistic trade manager.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CURRENT POSITION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Direction: ${position.side.toUpperCase()}
Entry Price: ${position.entryPrice.toFixed(2)}
Current Price: ${market.currentPrice.toFixed(2)}
Current P&L: ${profitLossPoints >= 0 ? '+' : ''}${profitLossPoints.toFixed(2)} pts (${profitLossPercent >= 0 ? '+' : ''}${profitLossPercent.toFixed(2)}%)
Current Stop Loss: ${position.stopLoss.toFixed(2)}
Current Target: ${position.target.toFixed(2)}
Risk:Reward Ratio: 1:${riskRewardRatio.toFixed(2)}
Progress to Target: ${percentToTarget.toFixed(1)}%
Time in Trade: ${(position.positionAgeSeconds / 60).toFixed(1)} minutes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 CURRENT MARKET CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recent Price Action (last 5 bars):
${market.recentBars && market.recentBars.length > 0 ? market.recentBars.slice(-5).map((bar, i) =>
  `  ${i + 1}. O: ${bar.open.toFixed(2)}, H: ${bar.high.toFixed(2)}, L: ${bar.low.toFixed(2)}, C: ${bar.close.toFixed(2)}, Vol: ${bar.volume}`
).join('\n') : '  No recent bar data available'}

Order Flow:
- CVD: ${market.cvd.toFixed(2)} (Trend: ${market.cvdTrend.toUpperCase()})
- Order Flow Pressure: ${market.orderFlowPressure.toUpperCase()}
${market.whaleActivity ? `- Whale Activity: ${market.whaleActivity}` : ''}

${market.volumeProfile ? `Volume Profile:
- POC: ${market.volumeProfile.poc.toFixed(2)}
- VAH: ${market.volumeProfile.vah.toFixed(2)}
- VAL: ${market.volumeProfile.val.toFixed(2)}` : ''}

${market.marketStructure ? `Market Structure: ${market.marketStructure}` : ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔬 MICROSTRUCTURE & COMPUTED FEATURES (fields may be null/undefined)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Distances are SIGNED (currentPrice - level); negative = price below level, positive = above.
- nextHvn/nextLvn are the next liquidity/value nodes in the trade direction.
- Use volRegime/currentRangeVsAtr for breathing-room decisions.
- PcEstimate = probability of TP hit before stop; PrEstimate = probability of stop/invalidation before TP.

Key Levels & Distances:
- distToPoc/Vah/Val: ${market.distToPoc} / ${market.distToVah} / ${market.distToVal}
- distToNextHvn/Lvn (directional): ${market.distToNextHvn} / ${market.distToNextLvn}
- nearest HVN/LVN above/below: ${market.distToNearestHvnAbove} / ${market.distToNearestHvnBelow} / ${market.distToNearestLvnAbove} / ${market.distToNearestLvnBelow}
- distToSessionHigh/Low: ${market.distToSessionHigh} / ${market.distToSessionLow}
- distToRoundNumber: ${market.distToRoundNumber}
- singlePrintZoneNearby: ${market.singlePrintZoneNearby ? JSON.stringify(market.singlePrintZoneNearby) : 'none'}
- inSinglePrintZone: ${market.inSinglePrintZone}

Order Flow & Liquidity:
- deltaLast1m/5m: ${market.deltaLast1m} / ${market.deltaLast5m}
- cvdSlopeShort/Long: ${market.cvdSlopeShort} / ${market.cvdSlopeLong}
- cvdDivergence: ${market.cvdDivergence}
- absorptionZone: ${market.absorptionZone ? JSON.stringify(market.absorptionZone) : 'none'}
- exhaustionFlag: ${market.exhaustionFlag ? JSON.stringify(market.exhaustionFlag) : 'none'}
- largePrints (top): ${market.largePrints ? JSON.stringify(market.largePrints) : 'none'}
- restingLiquidityWalls (top): ${market.restingLiquidityWalls ? JSON.stringify(market.restingLiquidityWalls) : 'none'}
- liquidityPullDetected: ${market.liquidityPullDetected}

Volatility & Structure:
- atr1m/atr5m: ${market.atr1m} / ${market.atr5m}
- volRegime: ${market.volRegime} (derived from currentRangeVsAtr: ${market.currentRangeVsAtr})
- structureState: ${market.structureState}
- invalidationPrice: ${market.invalidationPrice}
- lastSwingPrice: ${market.lastSwingPrice}
- trendStrength (0-1): ${market.trendStrength}
- PcEstimate / PrEstimate: ${market.PcEstimate} / ${market.PrEstimate}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR RISK MANAGEMENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
At each update, estimate:
- Continuation Probability (Pc): chance we reach TP before stop.
- Reversal Probability (Pr): chance we reverse into stop or invalidate.
- Edge Strength (E): structure alignment + orderflow + CVD + key levels + volatility regime.
- Local Risk:Reward (LRR) from current price: remaining reward vs profit at risk.

Then act to improve EV.

1) STOP LOSS OPTIMIZATION (adaptive, not automatic)
Breathing room rule:
- If Pc is still high and structure/orderflow support continuation, keep stop at a technically logical invalidation level, not arbitrarily tight.

Protection rule:
- Tighten stop when Pr rises, Pc falls, or LRR deteriorates.

Stop placement logic:
When moving stop, prefer locations that make market sense:
- behind HVN / value edge that should hold for your side
- behind single prints / LVN continuation lanes
- behind absorption zones or stacked resting liquidity (L2 walls)
- behind most recent structure swing defining the trade

Near-TP tightening:
- As price approaches TP, re-evaluate LRR.
- If remaining reward is small vs profit at risk, tighten sharply or take profit.
- Do NOT leave a wide stop when 80–95% to target unless Pc is extremely high.

⚠️ CRITICAL: BREAKEVEN CALCULATION RULES (DO NOT GET THIS WRONG!):
- For LONG positions: Breakeven stop = Entry Price EXACTLY (${position.side === 'long' ? position.entryPrice.toFixed(2) : 'N/A'})
- For SHORT positions: Breakeven stop = Entry Price EXACTLY (${position.side === 'short' ? position.entryPrice.toFixed(2) : 'N/A'})
❌ NEVER set stop ABOVE entry for SHORT
❌ NEVER set stop BELOW entry for LONG
✅ Breakeven = Entry Price EXACTLY

2) TARGET OPTIMIZATION (EV-based)
- Extend TP if Pc rises AND structure/orderflow show continuation strength, but only after profit is locked with a rational trailing stop.
- Hold TP if edge remains stable and volatility supports continuation.
- Take profit early / partial or full if:
  - Pc drops meaningfully,
  - momentum fades into resistance/support,
  - CVD diverges or flips against the position,
  - L2 liquidity shifts against you,
  - LRR is no longer favorable.

3) Priority Philosophy (hedge-fund style)
- Protect profit when EV says so, not because a fixed point threshold was hit.
- Let winners run when edge remains high.
- Tighten fast when edge decays.
- Near TP, protect aggressively unless continuation edge is exceptional.
- Extend targets only when continuation edge increases AND profit is already secured.

RESPOND WITH ONLY VALID JSON:
{
  "action": "HOLD_BRACKETS" | "ADJUST_STOP" | "ADJUST_TARGET" | "ADJUST_BOTH" | "CLOSE_POSITION",
  "newStopLoss": <number or null>,
  "newTarget": <number or null>,
  "reasoning": "<explain Pc/Pr/E/LRR and why this improves EV>",
  "urgency": "low" | "medium" | "high",
  "riskLevel": "conservative" | "balanced" | "aggressive"
}`;

  try {
    console.log(`[RiskMgmt] 💭 Calling DeepSeek Chat for risk analysis (fast)...`);

    const stream = await openai.chat.completions.create({
      model: 'deepseek-chat',
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 1,
      max_tokens: 8000,
      stream: true, // Enable streaming to prevent timeouts
    }, buildDeepseekCacheOptions('risk-mgmt'));

    let fullContent = '';
    let reasoningContent = '';
    let chunkCount = 0;

    try {
      for await (const chunk of stream) {
        chunkCount++;
        const delta = chunk.choices[0]?.delta;
        if (delta?.content) {
          fullContent += delta.content;
        }
        if (delta?.reasoning_content) {
          reasoningContent += delta.reasoning_content;
        }
      }
    } catch (streamError: any) {
      console.error(`[RiskMgmt] ❌ Streaming error after ${chunkCount} chunks:`, streamError.message);
      throw new Error(`Streaming failed: ${streamError.message}`);
    }

    if (!fullContent) {
      console.log(`[RiskMgmt] ⚠️ Empty response after ${chunkCount} chunks (reasoning: ${reasoningContent.length} chars)`);
      throw new Error('No response from risk management agent');
    }

    console.log(`[RiskMgmt] 📝 Received risk management response (${fullContent.length} chars, ${reasoningContent.length} reasoning chars, ${chunkCount} chunks)`);

    // Parse JSON from response
    const decision = parseRiskManagementResponse(fullContent);

    console.log(`[RiskMgmt] ✅ Decision: ${decision.action}`);
    console.log(`[RiskMgmt] 📊 New Stop: ${decision.newStopLoss?.toFixed(2) || 'unchanged'} | New Target: ${decision.newTarget?.toFixed(2) || 'unchanged'}`);
    console.log(`[RiskMgmt] 💡 Reasoning: ${decision.reasoning.substring(0, 150)}...`);

    // Enforce minimum protection: breakeven → lock-in
    const breakeven = position.entryPrice;
    const lockIn = position.side === 'long'
      ? position.entryPrice + LOCK_PROFIT_AMOUNT_POINTS
      : position.entryPrice - LOCK_PROFIT_AMOUNT_POINTS;

    const needsBreakeven = profitLossPoints >= BREAKEVEN_TRIGGER_POINTS;
    const needsLock = profitLossPoints >= LOCK_PROFIT_TRIGGER_POINTS;
    const nearTarget = percentToTarget >= TARGET_PROX_TRIGGER * 100;

    // Only tighten stops (never loosen)
    if (needsLock) {
      const proposed = decision.newStopLoss;
      const isTighter = position.side === 'long'
        ? !proposed || proposed < lockIn
        : !proposed || proposed > lockIn;
      if (isTighter) {
        decision.action = decision.action === 'ADJUST_TARGET' ? 'ADJUST_BOTH' : 'ADJUST_STOP';
        decision.newStopLoss = lockIn;
        decision.reasoning = `${decision.reasoning} | Enforced lock-in: stop moved to ${lockIn.toFixed(2)} after ${profitLossPoints.toFixed(2)} pts gain.`;
      }
    } else if (needsBreakeven) {
      const proposed = decision.newStopLoss;
      const isTighter = position.side === 'long'
        ? !proposed || proposed < breakeven
        : !proposed || proposed > breakeven;
      if (isTighter) {
        decision.action = decision.action === 'ADJUST_TARGET' ? 'ADJUST_BOTH' : 'ADJUST_STOP';
        decision.newStopLoss = breakeven;
        decision.reasoning = `${decision.reasoning} | Enforced breakeven: stop moved to entry after ${profitLossPoints.toFixed(2)} pts gain.`;
      }
    }

    // Gradual curve: scale stop tightening by percentage of distance traveled to target
    const totalRun = distanceToTarget;
    const traveled = Math.max(0, profitLossPoints);
    const progressRatio = totalRun > 0 ? Math.min(1, traveled / totalRun) : 0;

    // Aggression curve (0..1): early slow, then accelerating as we approach target
    const aggression =
      progressRatio < 0.3 ? 0.2 :
      progressRatio < 0.6 ? 0.5 :
      progressRatio < 0.9 ? 0.75 : 1.0;

    // Define desired lock based on aggression (percentage of traveled profit to keep)
    const desiredLockShare = Math.max(0.3, aggression * 0.8); // between 30% and up to 80% of traveled profit
    const dynamicLock = position.side === 'long'
      ? position.entryPrice + traveled * desiredLockShare
      : position.entryPrice - traveled * desiredLockShare;

    if (progressRatio > 0.05) { // only once in profit
      const proposed = decision.newStopLoss;
      const isTighter = position.side === 'long'
        ? !proposed || proposed < dynamicLock
        : !proposed || proposed > dynamicLock;

      if (isTighter) {
        decision.action = decision.action === 'ADJUST_TARGET' ? 'ADJUST_BOTH' : 'ADJUST_STOP';
        decision.newStopLoss = dynamicLock;
        decision.reasoning = `${decision.reasoning} | Dynamic lock: securing ~${Math.round(desiredLockShare * 100)}% of traveled profit (${traveled.toFixed(2)} pts), stop -> ${dynamicLock.toFixed(2)}.`;
      }
    }

    // Near-target lock: secure majority of open profit when ~80%+ to target (with safety gap)
    if (nearTarget) {
      const lockAmount = profitLossPoints * TARGET_LOCK_SHARE;
      const targetLock = position.side === 'long'
        ? position.entryPrice + lockAmount
        : position.entryPrice - lockAmount;

      // Keep a small safety gap from current price to avoid immediate stop-out
      const maxTight = position.side === 'long'
        ? Math.min(targetLock, market.currentPrice - STOP_SAFETY_GAP)
        : Math.max(targetLock, market.currentPrice + STOP_SAFETY_GAP);

      const proposed = decision.newStopLoss;
      const isTighter = position.side === 'long'
        ? !proposed || proposed < maxTight
        : !proposed || proposed > maxTight;

      if (isTighter) {
        decision.action = decision.action === 'ADJUST_TARGET' ? 'ADJUST_BOTH' : 'ADJUST_STOP';
        decision.newStopLoss = maxTight;
        decision.reasoning = `${decision.reasoning} | Near-target lock: securing ~${Math.round(TARGET_LOCK_SHARE * 100)}% of open profit at ${maxTight.toFixed(2)} (${profitLossPoints.toFixed(2)} pts unrealized).`;
      }
    }

    return decision;

  } catch (error: any) {
    console.error('[RiskMgmt] ❌ Error analyzing position risk:', error.message);

    // Return conservative default: hold current brackets
    return {
      action: 'HOLD_BRACKETS',
      newStopLoss: null,
      newTarget: null,
      reasoning: `Error occurred during risk analysis: ${error.message}. Maintaining current brackets for safety.`,
      urgency: 'low',
      riskLevel: 'conservative',
    };
  }
}

/**
 * Parse risk management response from AI
 */
function parseRiskManagementResponse(content: string): RiskManagementDecision {
  try {
    // Try to extract JSON from markdown code blocks or direct JSON
    let jsonStr: string | null = null;

    const markdownMatch = content.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
    if (markdownMatch) {
      jsonStr = markdownMatch[1].trim();
    }

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

    if (!jsonStr) {
      throw new Error('No JSON found in response');
    }

    const parsed = JSON.parse(jsonStr);

    // Validate required fields
    if (!parsed.action || !parsed.reasoning) {
      throw new Error('Missing required fields in response');
    }

    return {
      action: parsed.action,
      newStopLoss: parsed.newStopLoss,
      newTarget: parsed.newTarget,
      reasoning: parsed.reasoning,
      urgency: parsed.urgency || 'medium',
      riskLevel: parsed.riskLevel || 'balanced',
      positionVersion: parsed.positionVersion,
    };

  } catch (error: any) {
    console.error('[RiskMgmt] Failed to parse risk management response:', error.message);
    throw error;
  }
}
