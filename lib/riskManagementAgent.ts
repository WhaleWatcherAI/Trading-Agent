/**
 * Risk Management Agent
 * Secondary AI agent dedicated to managing open positions
 * Focuses on profit maximization and risk minimization through dynamic bracket adjustments
 */

import OpenAI from 'openai';
import { ActivePosition } from './executionManager';
import { TopstepXFuturesBar } from './topstepx';
import { buildDeepseekCacheOptions } from './deepseekCache';

// BELL CURVE STOP MANAGEMENT - Balance breathing room with profit protection
// Philosophy: Give room to grow early, tighten aggressively near target
const BREAKEVEN_PLUS_ONE_TRIGGER = 2; // in ticks: move stop to breakeven+1 tick once we have ~2 ticks profit (enough cushion to lock +1 tick)
const MIN_PROFIT_LOCK = 1; // minimum profit to lock (points, not ticks)
const STOP_SAFETY_GAP = 0.25; // tight gap - quarter tick buffer

// BELL CURVE WAYPOINTS - Gradual early, aggressive late
// Early: Wide breathing room to let winners develop
// Late: Tight stops - less wiggle room needed near target
const BELL_CURVE_STOPS = [
  { percentToTarget: 10, lockPercent: 0 },     // At 10% to target: breakeven (plenty of room to grow)
  { percentToTarget: 20, lockPercent: 5 },     // At 20%: lock 5% (still lots of breathing room)
  { percentToTarget: 30, lockPercent: 10 },    // At 30%: lock 10%
  { percentToTarget: 40, lockPercent: 20 },    // At 40%: lock 20% (starting to protect)
  { percentToTarget: 50, lockPercent: 30 },    // At 50%: lock 30% (bell curve middle)
  { percentToTarget: 60, lockPercent: 45 },    // At 60%: lock 45% (acceleration begins)
  { percentToTarget: 70, lockPercent: 60 },    // At 70%: lock 60% (getting tight)
  { percentToTarget: 80, lockPercent: 75 },    // At 80%: lock 75% (minimal wiggle room)
  { percentToTarget: 90, lockPercent: 85 },    // At 90%: lock 85% (very tight - almost at target)
  { percentToTarget: 95, lockPercent: 92 },    // At 95%: lock 92% (extremely tight - target imminent)
];

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
 * FAST rule-based stop management (executes in milliseconds)
 * Handles obvious cases without LLM latency
 */
function calculateBellCurveStop(pos: ActivePosition, currentPrice: number): { newStop: number; lockPercent: number; reasoning: string } | null {
  const profitLoss = currentPrice - pos.entryPrice;
  const profitLossPoints = pos.side === 'long' ? profitLoss : -profitLoss;

  const distanceToTarget = pos.side === 'long'
    ? pos.target - pos.entryPrice
    : pos.entryPrice - pos.target;

  const percentToTarget = (profitLossPoints / distanceToTarget) * 100;

  // Not profitable yet - don't move stop
  if (profitLossPoints <= 0) {
    return null;
  }

  // Find appropriate lock percentage based on bell curve waypoints
  let lockPercent = 0;
  for (const waypoint of BELL_CURVE_STOPS) {
    if (percentToTarget >= waypoint.percentToTarget) {
      lockPercent = waypoint.lockPercent;
    } else {
      break;
    }
  }

  // Calculate new stop based on lock percentage
  const profitToLock = profitLossPoints * (lockPercent / 100);
  const newStop = pos.side === 'long'
    ? pos.entryPrice + profitToLock
    : pos.entryPrice - profitToLock;

  // Don't move stop if it would loosen
  const currentStop = pos.stopLoss;
  const isLoosening = pos.side === 'long'
    ? newStop < currentStop
    : newStop > currentStop;

  if (isLoosening) {
    return null;
  }

  // Don't move if change is too small
  const stopChange = Math.abs(newStop - currentStop);
  if (stopChange < 0.5) {
    return null;
  }

  const reasoning = `Bell curve: ${percentToTarget.toFixed(1)}% to target. Locking ${lockPercent}% of ${profitLossPoints.toFixed(2)}pts profit. Moving stop from ${currentStop.toFixed(2)} to ${newStop.toFixed(2)} (+${(newStop - pos.entryPrice).toFixed(2)}pts from entry).`;

  return { newStop, lockPercent, reasoning };
}

/**
 * Analyze position and recommend risk management adjustments
 * HYBRID APPROACH:
 * - Phase 1: LLM manages getting to breakeven + 1 tick (with wiggle room for microstructure)
 * - Phase 2: Once breakeven + 1 tick secured, fast bell curve takes over (overrides lagging LLM)
 */
export async function analyzePositionRisk(
  pos: ActivePosition | null,
  market: MarketSnapshot,
  tickSize: number = 0.25 // NQ default, Gold should pass 0.1
): Promise<RiskManagementDecision> {
  if (!pos) {
    console.warn('[RiskMgmt] 🚫 analyzePositionRisk called without a position. Skipping.');
    return {
      action: 'HOLD_BRACKETS',
      newStopLoss: null,
      newTarget: null,
      reasoning: 'No position available for risk analysis; holding brackets.',
      urgency: 'low',
      riskLevel: 'conservative',
    };
  }

  console.log(`[RiskMgmt] 🎯 Analyzing position risk for ${pos.side.toUpperCase()} position...`);

  const profitLoss = market.currentPrice - pos.entryPrice;
  const profitLossPoints = pos.side === 'long' ? profitLoss : -profitLoss;
  const profitLossTicks = profitLossPoints / tickSize;

  // Calculate breakeven + 1 tick
  const breakevenPlusOneTick = pos.side === 'long'
    ? pos.entryPrice + tickSize
    : pos.entryPrice - tickSize;

  // Check if stop is already at or past breakeven + 1 tick
  const stopIsSecured = pos.side === 'long'
    ? pos.stopLoss >= breakevenPlusOneTick
    : pos.stopLoss <= breakevenPlusOneTick;

  const stopBelowBreakevenPlusOne = pos.side === 'long'
    ? pos.stopLoss < breakevenPlusOneTick
    : pos.stopLoss > breakevenPlusOneTick;

  // PHASE 1A: HARD GUARANTEE - once we have enough cushion, FORCE move to breakeven + 1 tick
  // This ensures the trade always locks out loss and at least a tiny profit, even if the LLM is slow/indecisive.
  if (!stopIsSecured && stopBelowBreakevenPlusOne && profitLossTicks >= BREAKEVEN_PLUS_ONE_TRIGGER) {
    const newStop = breakevenPlusOneTick;

    const isTightening = pos.side === 'long'
      ? newStop > pos.stopLoss
      : newStop < pos.stopLoss;

    if (isTightening) {
      console.log(`[RiskMgmt] 🧱 PHASE 1 AUTO: Profit ${profitLossTicks.toFixed(1)} ticks >= ${BREAKEVEN_PLUS_ONE_TRIGGER} ticks. Forcing stop to breakeven + 1 tick at ${newStop.toFixed(2)} to secure the trade.`);
      return {
        action: 'ADJUST_STOP',
        newStopLoss: newStop,
        newTarget: null,
        reasoning: `[PHASE 1 AUTO] Sufficient profit (${profitLossTicks.toFixed(1)} ticks) - moving stop to breakeven + 1 tick (${newStop.toFixed(2)}) to lock out loss and hand off to bell curve.`,
        urgency: 'high',
        riskLevel: 'conservative',
        positionVersion: pos.positionVersion,
      };
    }
  }

  // PHASE 2: If breakeven + 1 tick is secured, use FAST bell curve logic (overrides LLM)
  if (stopIsSecured) {
    const bellCurveResult = calculateBellCurveStop(pos, market.currentPrice);
    if (bellCurveResult) {
      console.log(`[RiskMgmt] ⚡ BELL CURVE (Phase 2): ${bellCurveResult.reasoning}`);
      return {
        action: 'ADJUST_STOP',
        newStopLoss: bellCurveResult.newStop,
        newTarget: null,
        reasoning: `[BELL CURVE - PHASE 2] ${bellCurveResult.reasoning}`,
        urgency: bellCurveResult.lockPercent > 70 ? 'high' : 'medium',
        riskLevel: 'balanced',
        positionVersion: pos.positionVersion,
      };
    }
  }

  // PHASE 1: Use LLM to manage getting to breakeven + 1 tick
  console.log(`[RiskMgmt] 🤖 LLM (Phase 1): Managing stop to reach breakeven + 1 tick (${breakevenPlusOneTick.toFixed(2)})`);
  console.log(`[RiskMgmt] 📊 Current stop: ${pos.stopLoss.toFixed(2)} | Target: ${breakevenPlusOneTick.toFixed(2)} | Secured: ${stopIsSecured}`);

  const profitLossPercent = (profitLossPoints / pos.entryPrice) * 100;

  const distanceToStop = pos.side === 'long'
    ? pos.entryPrice - pos.stopLoss
    : pos.stopLoss - pos.entryPrice;

  const distanceToTarget = pos.side === 'long'
    ? pos.target - pos.entryPrice
    : pos.entryPrice - pos.target;

  const percentToTarget = (profitLossPoints / distanceToTarget) * 100;
  const riskRewardRatio = distanceToTarget / distanceToStop;

  const prompt = `You are a RISK MANAGEMENT & TRADE OPTIMIZATION SPECIALIST for futures trading.
You operate at a Robbins World Cup / hedge-fund desk level.
You manage already-open positions from other traders.
Your mandate is capital protection first, EV-maximizing profit management second.

🎯 YOUR SPECIFIC MISSION (PHASE 1 - AGGRESSIVE STOP TIGHTENING):
The entry agent leaves WIDE stops (often 1:1 RR). Your job is to AGGRESSIVELY tighten them toward BREAKEVEN + 1 TICK.

TARGET STOP POSITION: ${breakevenPlusOneTick.toFixed(2)}
CURRENT STOP POSITION: ${pos.stopLoss.toFixed(2)}
GAP TO CLOSE: ${Math.abs(pos.stopLoss - breakevenPlusOneTick).toFixed(2)} points

🚨 MANDATE: You MUST tighten the stop on EVERY check. Don't wait. Don't hesitate.

Tightening strategy:
- If current stop is VERY wide (>5pts from entry): Make BIG jumps toward breakeven (tighten by 2-3 pts)
- If stop is moderately wide (2-5pts from entry): Make medium jumps (tighten by 1-2 pts)
- If stop is close (<2pts from entry): Make small jumps, use microstructure for placement
- Give some wiggle room (0.5-1pt buffer) but BIAS TOWARD TIGHTENING

Your goal: Close the gap FAST. Get to breakeven + 1 tick within 2-3 checks maximum.
Once achieved, automated bell curve takes over.

You are a DECISIVE risk manager. Tighten stops AGGRESSIVELY.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CURRENT POSITION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Direction: ${pos.side.toUpperCase()}
Entry Price: ${pos.entryPrice.toFixed(2)}
Current Price: ${market.currentPrice.toFixed(2)}
Current P&L: ${profitLossPoints >= 0 ? '+' : ''}${profitLossPoints.toFixed(2)} pts (${profitLossPercent >= 0 ? '+' : ''}${profitLossPercent.toFixed(2)}%)
Current Stop Loss: ${pos.stopLoss.toFixed(2)}
Current Target: ${pos.target.toFixed(2)}
Risk:Reward Ratio: 1:${riskRewardRatio.toFixed(2)}
Progress to Target: ${percentToTarget.toFixed(1)}%
Time in Trade: ${(pos.positionAgeSeconds / 60).toFixed(1)} minutes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 CURRENT MARKET CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recent Price Action (last 5 bars):
${market.recentBars && market.recentBars.length > 0 ? market.recentBars.slice(-5).map((bar, i) =>
  `  ${i + 1}. O: ${bar.open.toFixed(2)}, H: ${bar.high.toFixed(2)}, L: ${bar.low.toFixed(2)}, C: ${bar.close.toFixed(2)}, Vol: ${bar.volume}`
).join('\n') : '  No recent bar data available'}

🔥🔥🔥 PRIMARY INDICATORS (MOST IMPORTANT - ANALYZE THESE FIRST):

Order Flow:
- CVD: ${market.cvd.toFixed(2)} (Trend: ${market.cvdTrend ? market.cvdTrend.toUpperCase() : 'N/A'})
- Order Flow Pressure: ${market.orderFlowPressure ? market.orderFlowPressure.toUpperCase() : 'N/A'}
${market.whaleActivity ? `- Whale Activity: ${market.whaleActivity}` : ''}

${market.marketStructure ? `Market Structure: ${market.marketStructure}` : ''}

⚠️ SECONDARY INDICATORS (LESS ACCURATE - USE ONLY FOR CONFIRMATION):

${market.volumeProfile ? `Volume Profile (Full Session) - ⚠️ LESS ACCURATE, use last:
- POC: ${market.volumeProfile.poc.toFixed(2)}
- VAH: ${market.volumeProfile.vah.toFixed(2)}
- VAL: ${market.volumeProfile.val.toFixed(2)}` : ''}

${market.tradeLegProfile ? `Trade Leg Volume Profile - ⚠️ LESS ACCURATE, use for confirmation only:
- Trade Range: ${market.tradeLegProfile.minPrice.toFixed(2)} to ${market.tradeLegProfile.maxPrice.toFixed(2)} (${market.tradeLegProfile.rangeSize.toFixed(2)} pts)
- POC within Trade Leg: ${market.tradeLegProfile.poc ? market.tradeLegProfile.poc.toFixed(2) : 'None'}
- High Volume Nodes (Support/Resistance):
${market.tradeLegProfile.highVolumeNodes.map((n: any) => `  • ${n.price.toFixed(2)} (${n.volume} vol)`).join('\n') || '  None'}
- Low Volume Nodes (Potential Quick Move Zones):
${market.tradeLegProfile.lowVolumeNodes.map((n: any) => `  • ${n.price.toFixed(2)} (${n.volume} vol)`).join('\n') || '  None'}
- Total Volume in Range: ${market.tradeLegProfile.totalVolume}` : ''}

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

🔥🔥🔥 Order Flow & L2 Liquidity (ANALYZE FIRST):
- 🔥 restingLiquidityWalls (L2 bid/ask walls - MOST IMPORTANT): ${market.restingLiquidityWalls ? JSON.stringify(market.restingLiquidityWalls) : 'none'}
- 🔥 absorptionZone (order flow absorption - CRITICAL): ${market.absorptionZone ? JSON.stringify(market.absorptionZone) : 'none'}
- deltaLast1m/5m: ${market.deltaLast1m} / ${market.deltaLast5m}
- cvdSlopeShort/Long: ${market.cvdSlopeShort} / ${market.cvdSlopeLong}
- cvdDivergence: ${market.cvdDivergence}
- exhaustionFlag: ${market.exhaustionFlag ? JSON.stringify(market.exhaustionFlag) : 'none'}
- largePrints (top): ${market.largePrints ? JSON.stringify(market.largePrints) : 'none'}
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
- Edge Strength (E): 🔥 L2 WALLS + ORDER FLOW + CVD (MOST IMPORTANT) >> structure + key levels
- Local Risk:Reward (LRR) from current price: remaining reward vs profit at risk.

Then act to improve EV.

1) STOP LOSS OPTIMIZATION (REQUIRED MICROSTRUCTURE ANALYSIS)

🚨 MANDATORY: You MUST analyze and cite specific microstructure levels for stop placement.
DO NOT give generic reasoning. USE THE DATA PROVIDED.

🔥🔥🔥 CRITICAL ANALYSIS ORDER - YOU MUST FOLLOW THIS EXACTLY:
1. START your reasoning by analyzing restingLiquidityWalls (L2 bid/ask walls with $ amounts)
2. THEN analyze absorptionZone (order flow absorption - where aggressive orders were absorbed)
3. THEN analyze CVD trends and divergence
4. THEN check single print zones and invalidation levels
5. ONLY AT THE END mention volume profile HVN/LVN as SECONDARY confirmation

❌ FORBIDDEN - DO NOT DO THIS:
- DO NOT start your reasoning with "Price is at VAL support" or "Price is at VAH resistance"
- DO NOT lead with "We're below/above POC"
- DO NOT say "at value area low/high where rejection often occurs"
- DO NOT make volume profile (VAL/VAH/POC) the PRIMARY focus of your analysis
- Volume profile is LESS ACCURATE and should be mentioned LAST, if at all

✅ CORRECT REASONING STRUCTURE:
"L2 shows [X] lot bid/ask wall at [price]. Absorption zone detected at [price] on [side] with strength [X]. CVD shows [trend]. Single prints at [price]. Therefore placing stop at [price]. (Volume profile HVN at [price] provides secondary confirmation.)"

🔥 PRIORITY: Focus on L2 resting liquidity walls and order flow absorption zones!
Volume profile (HVN/LVN) is SECONDARY and less accurate.

When in profit, trail stop aggressively by placing it behind technical support/resistance:

REQUIRED ANALYSIS STEPS (IN ORDER OF IMPORTANCE):
1. 🔥 Check restingLiquidityWalls FIRST - identify L2 bid/ask walls (REAL $ protecting levels)
2. 🔥 Check absorptionZone - where aggressive orders were absorbed (shows strength/weakness)
3. Check CVD slopes and divergence - cumulative delta shows true buying/selling pressure
4. Check singlePrintZoneNearby - single prints show imbalance, likely to fill
5. Check invalidationPrice from structure analysis
6. Check distToNearestHvnAbove/Below LAST - HVN levels (SECONDARY, less accurate)

🚨 CRITICAL: LEVEL 2 RESTING WALLS AND ORDER FLOW ARE THE MOST IMPORTANT INDICATORS!
Volume Profile data is LESS accurate and should be used ONLY as secondary confirmation.

STOP PLACEMENT PRIORITY (in order of importance):
For LONG positions trailing up:
1. 🔥 ABOVE resting bid liquidity walls (L2 data) - MOST IMPORTANT, real $ protecting position
2. 🔥 ABOVE absorption zones on bid side - where selling was absorbed, shows strength
3. Above CVD support zones - where cumulative delta turned positive
4. Above single print zones (unfilled auction areas) - likely to fill and provide support
5. Above recent swing low / invalidation price
6. Above nearest HVN below (SECONDARY - volume profile less accurate)

For SHORT positions trailing down:
1. 🔥 BELOW resting ask liquidity walls (L2 data) - MOST IMPORTANT, real $ protecting position
2. 🔥 BELOW absorption zones on ask side - where buying was absorbed, shows weakness
3. Below CVD resistance zones - where cumulative delta turned negative
4. Below single print zones - unfilled areas that may fill on downward continuation
5. Below recent swing high / invalidation price
6. Below nearest HVN above (SECONDARY - volume profile less accurate)

⚠️ CRITICAL FOR SHORT: Stop goes ABOVE resistance to protect from UPWARD reversals.
Place stop "behind" the structural barrier that should prevent upward moves.

YOUR REASONING MUST INCLUDE:
✅ "Within trade leg: HVN at [PRICE] provides support/resistance" - USE TRADE LEG PROFILE
✅ "Trailing stop to [PRICE] - positioned behind trade leg HVN at [SPECIFIC PRICE]"
✅ "Trade leg POC at [PRICE] acts as magnet/pivot within range"
✅ "LVN at [PRICE] within trade leg suggests quick move zone"
✅ "Current profit: [X] pts. Locking [%] by moving stop from [OLD] to [NEW]"
❌ DO NOT use overall market VAH/VAL for stop placement - use TRADE LEG nodes
❌ DO NOT say generic things like "adequate breathing room" or "provides protection"
❌ DO NOT ignore volume nodes within your trade range

🔔 BELL CURVE STOP MANAGEMENT (MANDATORY):
Follow this progressive tightening schedule based on % to target:
- 10-20% to target: Move to breakeven (lock out losses, give breathing room)
- 20-40% to target: Lock 5-20% of profit (still wide breathing room for growth)
- 40-60% to target: Lock 20-45% of profit (middle of bell curve, balance protection & room)
- 60-80% to target: Lock 45-75% of profit (acceleration begins, less wiggle room needed)
- 80-95% to target: Lock 75-92% of profit (extremely tight, target is close)

Philosophy: Give room to GROW early (don't choke winners), tighten AGGRESSIVELY near target (lock gains).
The closer to target, the LESS wiggle room you need - tighten stops progressively.

⚠️ CRITICAL: BREAKEVEN CALCULATION RULES (DO NOT GET THIS WRONG!):
- For LONG positions: Breakeven stop = Entry Price EXACTLY (${pos.side === 'long' ? pos.entryPrice.toFixed(2) : 'N/A'})
- For SHORT positions: Breakeven stop = Entry Price EXACTLY (${pos.side === 'short' ? pos.entryPrice.toFixed(2) : 'N/A'})
❌ NEVER set stop ABOVE entry for SHORT
❌ NEVER set stop BELOW entry for LONG
✅ Breakeven = Entry Price EXACTLY
✅ After breakeven, lock at least 1 tick of profit ASAP (breakeven + 1 tick)

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
  "reasoning": "<MUST cite specific levels - see examples below>",
  "urgency": "low" | "medium" | "high",
  "riskLevel": "conservative" | "balanced" | "aggressive"
}

REASONING EXAMPLES:

✅ GOOD (L2 and order flow FIRST, volume profile last):
"L2 shows 1200 lot ask wall at 24492.00 protecting this level. Absorption zone at 24498.50 (ask side, strength 0.75) where selling was absorbed. CVD negative trend supporting continuation. Single prints above 24500. Trailing stop from 24514.00 to 24495.00 locks 60% of +14.5pts profit. (HVN at 24500.00 provides secondary confirmation.)"

✅ GOOD (cites L2 walls and order flow):
"Resting bid liquidity wall at 25195.00 with 850 contracts provides strong support. Order flow shows absorption on bid side at 25193.50 (strength 0.82). CVD positive slope supporting long. No single prints below. Tightening stop to 25190.00 to lock profit while staying above L2 protection."

❌ BAD (leads with volume profile - FORBIDDEN):
"Price is at VAL support (25061.25) with recent bounce from 25061.25 low. We're below POC but at value area low where rejection often occurs. Structure shows higher lows. Neutral flow but location provides edge."

❌ BAD (mentions POC/VAH/VAL prominently):
"Stop at 24514.00 provides adequate breathing room above VAH. POC acts as magnet. Edge remains intact at value area high."

❌ BAD (generic, no L2 or order flow data):
"Current brackets look good, market structure balanced, holding position."`;

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
    const decision = parseRiskManagementResponse(fullContent, pos, market);

    console.log(`[RiskMgmt] ✅ Decision: ${decision.action}`);
    console.log(`[RiskMgmt] 📊 New Stop: ${decision.newStopLoss?.toFixed(2) || 'unchanged'} | New Target: ${decision.newTarget?.toFixed(2) || 'unchanged'}`);
    console.log(`[RiskMgmt] 💡 Reasoning: ${decision.reasoning.substring(0, 150)}...`);

    // 🚨 CRITICAL SAFETY: NEVER ALLOW STOP LOSS TO LOOSEN (move away from current price)
    // Stops can ONLY tighten (move closer to current price), NEVER loosen
    if (decision.newStopLoss !== null && decision.newStopLoss !== undefined) {
      const currentStop = pos.stopLoss;
      const isLoosening = pos.side === 'long'
        ? decision.newStopLoss < currentStop  // LONG: new stop below current = loosening (BAD)
        : decision.newStopLoss > currentStop; // SHORT: new stop above current = loosening (BAD)

      if (isLoosening) {
        console.error(`[RiskMgmt] 🚫 REJECTED: AI tried to LOOSEN stop from ${currentStop.toFixed(2)} to ${decision.newStopLoss.toFixed(2)}!`);
        console.error(`[RiskMgmt] 🚫 Stops can ONLY tighten, NEVER loosen. Keeping current stop.`);
        decision.newStopLoss = null; // Reject the loosening
        decision.action = decision.action === 'ADJUST_STOP' ? 'HOLD_BRACKETS' :
                         decision.action === 'ADJUST_BOTH' ? 'ADJUST_TARGET' : decision.action;
        decision.reasoning = `${decision.reasoning} | SAFETY OVERRIDE: Rejected stop loosening from ${currentStop.toFixed(2)} to proposed ${decision.newStopLoss?.toFixed(2)}. Stops can only tighten.`;
      }
    }

    return decision;

  } catch (error: any) {
    console.error('[RiskMgmt] ❌ Error analyzing position risk:', error?.message || error, pos ? `| Position: ${pos.side} @ ${pos.entryPrice}` : '| Position: none');
    if (error?.stack) {
      console.error('[RiskMgmt] Stack trace:', error.stack);
    }

    // Return conservative default: hold current brackets
    return {
      action: 'HOLD_BRACKETS',
      newStopLoss: null,
      newTarget: null,
      reasoning: `RISK_ERR_V2 | ${error?.message || error}. Holding brackets.`,
      urgency: 'low',
      riskLevel: 'conservative',
    };
  }
}

/**
 * Parse risk management response from AI
 */
function parseRiskManagementResponse(content: string, pos: ActivePosition, market: MarketSnapshot): RiskManagementDecision {
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

    // Calculate progress to target for bell curve stops
    const profitLoss = market.currentPrice - pos.entryPrice;
    const profitLossPoints = pos.side === 'long' ? profitLoss : -profitLoss;
    const distanceToTarget = pos.side === 'long'
      ? pos.target - pos.entryPrice
      : pos.entryPrice - pos.target;
    const percentToTarget = distanceToTarget > 0 ? (profitLossPoints / distanceToTarget) * 100 : 0;

    // BELL CURVE STOP MANAGEMENT: Progressive profit locking as we approach target
    let minimumStop = parsed.newStopLoss;

    if (profitLossPoints > 0 && percentToTarget > 0) {
      // Define the bell curve stop levels
      let lockPercentage = 0;

      if (percentToTarget >= 90) {
        // Near target: Lock 85% of profit (very tight)
        lockPercentage = 0.85;
        console.log(`[RiskMgmt] 🎯 90%+ to target - Locking 85% of profit`);
      } else if (percentToTarget >= 75) {
        // 75% to target: Lock 70% of profit
        lockPercentage = 0.70;
        console.log(`[RiskMgmt] 🎯 75%+ to target - Locking 70% of profit`);
      } else if (percentToTarget >= 50) {
        // Halfway: Lock 40% of profit
        lockPercentage = 0.40;
        console.log(`[RiskMgmt] 🎯 50%+ to target - Locking 40% of profit`);
      } else if (percentToTarget >= 25) {
        // Quarter way: Move to breakeven
        lockPercentage = 0;
        console.log(`[RiskMgmt] 🎯 25%+ to target - Moving to breakeven`);
      }

      // Calculate minimum stop based on profit locking
      if (lockPercentage > 0) {
        const profitToLock = profitLossPoints * lockPercentage;
        if (pos.side === 'long') {
          minimumStop = pos.entryPrice + profitToLock;
        } else {
          minimumStop = pos.entryPrice - profitToLock;
        }
      } else if (percentToTarget >= 25) {
        // At least breakeven
        minimumStop = pos.entryPrice;
      }

      // Override LLM stop if it's not tight enough
      if (typeof minimumStop === 'number' && typeof parsed.newStopLoss === 'number') {
        const llmStopInsufficient = pos.side === 'long'
          ? parsed.newStopLoss < minimumStop
          : parsed.newStopLoss > minimumStop;

        if (llmStopInsufficient) {
          console.log(`[RiskMgmt] ⚡ BELL CURVE OVERRIDE: LLM stop ${parsed.newStopLoss.toFixed(2)} not tight enough. Using ${minimumStop.toFixed(2)}`);
          parsed.reasoning += ` [Bell Curve Override: ${percentToTarget.toFixed(0)}% to target requires ${(lockPercentage * 100).toFixed(0)}% profit lock]`;
        }
      }
    }

    // Use the tighter of LLM suggestion or bell curve minimum
    let snappedStop = parsed.newStopLoss;
    if (typeof minimumStop === 'number' && typeof parsed.newStopLoss === 'number') {
      if (pos.side === 'long') {
        snappedStop = Math.max(parsed.newStopLoss, minimumStop);
      } else {
        snappedStop = Math.min(parsed.newStopLoss, minimumStop);
      }
    }

    // Snap to exact breakeven if very close
    const breakevenPad = 0.5;
    if (typeof snappedStop === 'number') {
      if (pos.side === 'long' && snappedStop >= pos.entryPrice - breakevenPad && snappedStop <= pos.entryPrice + breakevenPad) {
        snappedStop = pos.entryPrice;
      }
      if (pos.side === 'short' && snappedStop <= pos.entryPrice + breakevenPad && snappedStop >= pos.entryPrice - breakevenPad) {
        snappedStop = pos.entryPrice;
      }
    }

    // Guard: never move stop beyond take-profit (prevents stop hiding behind TP)
    if (typeof snappedStop === 'number' && typeof pos.target === 'number') {
      const stopBeyondTarget = pos.side === 'long'
        ? snappedStop >= pos.target
        : snappedStop <= pos.target;
      if (stopBeyondTarget) {
        console.warn(`[RiskMgmt] 🚫 Stop proposal crosses target (stop ${snappedStop}, target ${pos.target}) - dropping stop adjustment.`);
        snappedStop = null;
        parsed.action = parsed.action === 'ADJUST_STOP' ? 'HOLD_BRACKETS'
          : parsed.action === 'ADJUST_BOTH' ? 'ADJUST_TARGET'
          : parsed.action;
        parsed.reasoning = `${parsed.reasoning} | SAFETY: Stop proposal crossed target, stop adjustment dropped.`;
      }
    }

    // Guard: only block stop moves in drawdown if they do NOT improve risk
    const unrealizedPoints = pos.side === 'long'
      ? market.currentPrice - pos.entryPrice
      : pos.entryPrice - market.currentPrice;
    const stopProposalImprovesRisk = typeof snappedStop === 'number' && typeof pos.stopLoss === 'number'
      ? (pos.side === 'long' ? snappedStop > pos.stopLoss : snappedStop < pos.stopLoss)
      : true; // if we don't have a current stop, allow the proposal
    if (unrealizedPoints <= 0
      && (parsed.action === 'ADJUST_STOP' || parsed.action === 'ADJUST_BOTH')
      && !stopProposalImprovesRisk) {
      console.warn('[RiskMgmt] 🚫 Skip stop tightening: position not in profit and proposal does not improve risk.');
      snappedStop = null;
      parsed.action = parsed.action === 'ADJUST_STOP' ? 'HOLD_BRACKETS'
        : parsed.action === 'ADJUST_BOTH' ? 'ADJUST_TARGET'
        : parsed.action;
      parsed.reasoning = `${parsed.reasoning} | SAFETY: Stop not tightened; trade is not in profit and proposal was not an improvement.`;
    }

    return {
      action: parsed.action,
      newStopLoss: snappedStop,
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
