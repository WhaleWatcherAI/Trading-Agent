/**
 * Risk Management Agent
 * Secondary AI agent dedicated to managing open positions
 * Focuses on profit maximization and risk minimization through dynamic bracket adjustments
 */

import OpenAI from 'openai';
import { ActivePosition } from './executionManager';
import { TopstepXFuturesBar } from './topstepx';

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

  const prompt = `You are a RISK MANAGEMENT & DRAWDOWN PREVENTION SPECIALIST for futures trading.
You are acting as a Robbins World Cup trader caliber professional, managing risk and optimizing profits for a hedge fund desk. Your current role is risk management on positions opened by other traders—your mandate is capital protection first, profit optimization second.

Your PRIMARY responsibilities (in order of importance):
1. **PREVENT DRAWDOWN** - NEVER let winning trades turn into losers. Lock in profits aggressively.
2. **PROTECT CAPITAL** - Trail stops constantly as price moves favorably. Any profit > 5pts MUST be protected.
3. **CUT LOSSES FAST** - If trade thesis weakens or CVD diverges, tighten stop or close immediately.
4. **MAXIMIZE GAINS** - ONLY extend targets when momentum is strong AND profit is already locked in with trailing stops.
5. **CAPTURE PROFITS** - Take profits early if ANY reversal signs appear (better 80% gain than 0%).

🛡️ YOUR MISSION: Protect the account from drawdown while capturing maximum profit when conditions allow.
🔥 BE AGGRESSIVE WITH PROTECTION, CONSERVATIVE WITH RISK!
⚠️ WINNING TRADES TURNING INTO LOSERS IS UNACCEPTABLE - Trail stops proactively!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CURRENT POSITION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Direction: ${position.side.toUpperCase()}
Entry Price: ${position.entryPrice.toFixed(2)}
Current Price: ${market.currentPrice.toFixed(2)}
Current P&L: ${profitLossPoints >= 0 ? '+' : ''}${profitLossPoints.toFixed(2)} pts (${profitLossPercent >= 0 ? '+' : ''}${profitLossPercent.toFixed(2)}%)

Current Stop Loss: ${position.stopLoss.toFixed(2)} (${distanceToStop.toFixed(2)} pts from entry)
Current Target: ${position.target.toFixed(2)} (${distanceToTarget.toFixed(2)} pts from entry)
Risk:Reward Ratio: 1:${riskRewardRatio.toFixed(2)}

Progress to Target: ${percentToTarget.toFixed(1)}% (${profitLossPoints.toFixed(2)} of ${distanceToTarget.toFixed(2)} pts)
Time in Trade: ${(position.positionAgeSeconds / 60).toFixed(1)} minutes

Bracket Order IDs:
- Stop Order: ${position.stopOrderId || 'unknown'}
- Target Order: ${position.targetOrderId || 'unknown'}
- Uses Native Bracket: ${position.usesNativeBracket ? 'Yes' : 'No'}

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
- POC (Point of Control): ${market.volumeProfile.poc.toFixed(2)}
- VAH (Value Area High): ${market.volumeProfile.vah.toFixed(2)}
- VAL (Value Area Low): ${market.volumeProfile.val.toFixed(2)}` : ''}

${market.marketStructure ? `Market Structure: ${market.marketStructure}` : ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR RISK MANAGEMENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze the position and market conditions, then decide:

1. **STOP LOSS OPTIMIZATION** (ALWAYS look for opportunities to improve):
   - Move stop to break-even if EVEN 2+ points of profit (protect capital)
   - Trail stop aggressively as price moves in our favor (lock gains)
   - Tighten stop if ANY weakness appears (divergence, weakening CVD, opposing whale activity)
   - Widen stop ONLY if strong continuation pattern developing

   ⚠️ CRITICAL: BREAKEVEN CALCULATION RULES (DO NOT GET THIS WRONG!):
   - For LONG positions: Breakeven stop = Entry Price EXACTLY (${position.side === 'long' ? position.entryPrice.toFixed(2) : 'N/A'})
     Example: Entry 4063.0 → Breakeven stop = 4063.0 (NOT 4063.1 or 4062.9)

   - For SHORT positions: Breakeven stop = Entry Price EXACTLY (${position.side === 'short' ? position.entryPrice.toFixed(2) : 'N/A'})
     Example: Entry 4063.0 → Breakeven stop = 4063.0 (NOT 4063.1 or 4062.9)

   ❌ NEVER set stop ABOVE entry for SHORT (locks in loss!)
   ❌ NEVER set stop BELOW entry for LONG (locks in loss!)
   ✅ Breakeven = Entry Price EXACTLY, regardless of direction

2. **TARGET OPTIMIZATION** (ACTIVELY seek improvements):
   - Extend target if momentum strengthening OR price breaking key levels in our favor
   - Take profit early if approaching resistance/support with weakening momentum
   - Exit immediately if strong reversal signals or conditions completely changed
   - Partial targets: consider closing at current profit if risk/reward deteriorated

3. **CONSIDERATIONS**:
   - Distance to key levels (POC, VAH, VAL, round numbers)
   - Order flow alignment (CVD trend vs position direction)
   - Momentum strength or weakness
   - How much profit is at risk vs potential gain
   - Time decay (positions held too long without progress)
   - Whale activity indicating support or resistance

RESPOND WITH ONLY VALID JSON:
{
  "action": "HOLD_BRACKETS" | "ADJUST_STOP" | "ADJUST_TARGET" | "ADJUST_BOTH" | "CLOSE_POSITION",
  "newStopLoss": <number or null>,
  "newTarget": <number or null>,
  "reasoning": "<detailed explanation of why these adjustments maximize profit and manage risk>",
  "urgency": "low" | "medium" | "high",
  "riskLevel": "conservative" | "balanced" | "aggressive"
}

🎯 DRAWDOWN PREVENTION PHILOSOPHY:
- **PROTECT FIRST, EXTEND SECOND** - Always trail stops BEFORE considering target extensions
- **Lock gains aggressively** - 5pts profit → stop to breakeven. 10pts profit → stop to +5pts minimum
- **Never give back more than 30%** - If up 20pts, NEVER let it drop below +14pts without tightening stop
- **Close on weakness** - ANY sign of reversal (CVD flip, momentum fade, opposing whale activity) → tighten stop or close
- **Extend targets ONLY when protected** - Only extend target if stop is already trailing and locking profit
- **Default to protection** - When in doubt, tighten the stop. You can always re-enter, but can't recover blown accounts.

EXAMPLES OF DRAWDOWN PREVENTION IN ACTION:
✅ LONG @ 4063.0, +2pts profit → Stop to 4063.0 (breakeven - entry price exactly)
✅ SHORT @ 4063.0, +2pts profit → Stop to 4063.0 (breakeven - entry price exactly)
✅ LONG @ 5885.0, +5pts profit → Stop to 5885.0 (breakeven first), then extend target
✅ SHORT @ 4078.0, +5pts profit → Stop to 4078.0 (breakeven first), then extend target
✅ +15pts profit, price stalling → Trail stop to +10pts (lock 67% of gain), hold target
✅ +8pts profit, CVD weakening → Trail stop to +5pts immediately, prepare to close
✅ +3pts profit, trade thesis breaking → Move stop to +1pt or close immediately
✅ +25pts profit (near target), momentum fading → CLOSE_POSITION NOW and bank the win
✅ Underwater -5pts, CVD diverging from position → Tighten stop or close to prevent bigger loss
❌ SHORT @ 4063.0, setting stop to 4063.1 → LOCKS IN LOSS! Should be 4063.0 exactly
❌ LONG @ 5885.0, setting stop to 5884.9 → LOCKS IN LOSS! Should be 5885.0 exactly
❌ +20pts profit with stop still at entry → UNACCEPTABLE RISK! Trail immediately!
❌ +10pts profit, signs of reversal but holding → WRONG! Protect or close!

🛡️ REMEMBER: Drawdown is the enemy. Winning trades that turn into losers destroy accounts.
💰 A smaller locked-in gain beats a potential larger gain that becomes a loss.`;

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
    });

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
    };

  } catch (error: any) {
    console.error('[RiskMgmt] Failed to parse risk management response:', error.message);
    throw error;
  }
}
