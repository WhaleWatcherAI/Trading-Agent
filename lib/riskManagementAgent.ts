/**
 * Risk Management Agent - DETERMINISTIC VERSION (NO LLM)
 * Pure math-based smooth bell curve trailing stop
 * Starts immediately when position fills, moves forward with price
 */

import { ActivePosition } from './executionManager';
import { TopstepXFuturesBar } from './topstepx';

export interface RiskManagementDecision {
  action: 'HOLD_BRACKETS' | 'ADJUST_STOP' | 'ADJUST_TARGET' | 'ADJUST_BOTH' | 'CLOSE_POSITION';
  newStopLoss: number | null;
  newTarget: number | null;
  reasoning: string;
  urgency: 'low' | 'medium' | 'high';
  riskLevel: 'conservative' | 'balanced' | 'aggressive';
  positionVersion?: number;
}

export interface MarketSnapshot {
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
  // All other optional fields (not needed for deterministic bell curve)
  [key: string]: any;
}

/**
 * SMOOTH FORWARD-MOVING BELL CURVE TRAILING STOP
 *
 * Philosophy:
 * - Starts IMMEDIATELY on FIRST TICK in profit (no waiting for breakeven)
 * - Moves FORWARD from entry price toward take profit as price moves in our favor
 * - Stop drags behind current price at an exponentially shrinking distance
 * - Uses smooth continuous interpolation (no big jumps)
 * - Only needs: entry price, current price, initial stop, take profit
 * - Works on every price tick update
 */
function calculateSmoothBellCurveStop(pos: ActivePosition, currentPrice: number, tickSize: number = 0.25): { newStop: number; lockPercent: number; reasoning: string } | null {
  const profitLoss = currentPrice - pos.entryPrice;
  const profitLossPoints = pos.side === 'long' ? profitLoss : -profitLoss;

  const distanceToTarget = pos.side === 'long'
    ? pos.target - pos.entryPrice
    : pos.entryPrice - pos.target;

  // DEFENSIVE: Detect inverted brackets and warn (indicates upstream bug)
  if (distanceToTarget <= 0) {
    console.error(`[RiskMgmt] ❌ INVERTED BRACKETS DETECTED! ${pos.side.toUpperCase()} position has invalid brackets:`);
    console.error(`[RiskMgmt]    Entry=${pos.entryPrice.toFixed(2)}, Stop=${pos.stopLoss.toFixed(2)}, Target=${pos.target.toFixed(2)}`);
    console.error(`[RiskMgmt]    distanceToTarget=${distanceToTarget.toFixed(2)} (should be positive)`);
    console.error(`[RiskMgmt]    This prevents bell curve trailing stop from activating!`);
    return null;
  }

  // Not profitable yet - don't move stop
  if (profitLossPoints <= 0) {
    console.log(`[RiskMgmt] ⏸️  Position not profitable yet: profitLossPoints=${profitLossPoints.toFixed(2)}, currentPrice=${currentPrice.toFixed(2)}, entryPrice=${pos.entryPrice.toFixed(2)}`);
    return null;
  }

  const profitTicks = profitLossPoints / tickSize;
  console.log(`[RiskMgmt] 💰 Position IN PROFIT: +${profitTicks.toFixed(1)} ticks (+${profitLossPoints.toFixed(2)} pts) | Entry: ${pos.entryPrice.toFixed(2)}, Current: ${currentPrice.toFixed(2)}`);

  // IMMEDIATE BELL CURVE: Start dragging from FIRST TICK in profit
  // Calculate progress from ENTRY to TAKE PROFIT (0% to 100%)
  const percentToTarget = (profitLossPoints / distanceToTarget) * 100;

  // DRAGGING STOP: Stop drags behind current price at a distance
  // The distance SHRINKS as we approach target (bell curve shape)
  //
  // Distance Multiplier (how far stop drags behind current price):
  // - 0-30% progress: 0.90-0.70 (loose, 70-90% of remaining distance)
  // - 30-60% progress: 0.70-0.40 (tightening, 40-70%)
  // - 60-100% progress: 0.40-0.05 (very tight, 5-40%)

  let distanceMultiplier: number;

  if (percentToTarget <= 0) {
    distanceMultiplier = 0.95; // Very loose at start
  } else if (percentToTarget >= 100) {
    distanceMultiplier = 0.05; // Very tight at target
  } else if (percentToTarget <= 30) {
    // 0-30%: Linear from 0.95 to 0.70
    // Lots of breathing room
    const progress = percentToTarget / 30;
    distanceMultiplier = 0.95 - (progress * 0.25);
  } else if (percentToTarget <= 60) {
    // 30-60%: Quadratic from 0.70 to 0.40
    // Moderate acceleration
    const progress = (percentToTarget - 30) / 30;
    const quadratic = Math.pow(progress, 2);
    distanceMultiplier = 0.70 - (quadratic * 0.30);
  } else {
    // 60-100%: Cubic from 0.40 to 0.05
    // Aggressive tightening near target
    const progress = (percentToTarget - 60) / 40;
    const cubic = Math.pow(progress, 3);
    distanceMultiplier = 0.40 - (cubic * 0.35);
  }

  // Calculate distance from current price to target
  const distanceToTargetFromCurrent = pos.side === 'long'
    ? pos.target - currentPrice
    : currentPrice - pos.target;

  // Stop drags behind at the calculated distance
  const dragDistance = distanceToTargetFromCurrent * distanceMultiplier;

  const newStop = pos.side === 'long'
    ? currentPrice - dragDistance
    : currentPrice + dragDistance;

  // Don't move stop if it would loosen
  const isLoosening = pos.side === 'long'
    ? newStop < pos.stopLoss
    : newStop > pos.stopLoss;

  if (isLoosening) {
    return null;
  }

  // Calculate stop change
  const stopChange = Math.abs(newStop - pos.stopLoss);

  // Skip if change is negligible (less than 0.01 points)
  if (stopChange < 0.01) {
    return null;
  }

  const profitFromEntry = pos.side === 'long'
    ? newStop - pos.entryPrice
    : pos.entryPrice - newStop;

  const dragDistanceFormatted = dragDistance.toFixed(2);
  const profitTicksFormatted = profitTicks.toFixed(1);
  const reasoning = `📊 BELL CURVE: +${profitTicksFormatted} ticks (${percentToTarget.toFixed(1)}% to TP) → stop ${dragDistanceFormatted}pts behind price | ${pos.stopLoss.toFixed(2)} → ${newStop.toFixed(2)} (${profitFromEntry >= 0 ? '+' : ''}${profitFromEntry.toFixed(2)}pts from entry)`;

  return { newStop, lockPercent: (1 - distanceMultiplier) * 100, reasoning };
}

/**
 * Analyze position and recommend risk management adjustments
 * DETERMINISTIC APPROACH - NO LLM:
 * - Uses smooth bell curve trailing stop immediately from entry
 * - No phases, no LLM delays
 * - Pure math-based forward-moving stop management
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

  console.log(`[RiskMgmt] 🎯 DETERMINISTIC RISK MGMT: Analyzing ${pos.side.toUpperCase()} position with smooth bell curve...`);

  // ONLY USE SMOOTH BELL CURVE - NO LLM
  const bellCurveResult = calculateSmoothBellCurveStop(pos, market.currentPrice, tickSize);

  if (bellCurveResult) {
    console.log(`[RiskMgmt] ⚡ SMOOTH BELL CURVE: ${bellCurveResult.reasoning}`);
    return {
      action: 'ADJUST_STOP',
      newStopLoss: bellCurveResult.newStop,
      newTarget: null,
      reasoning: `[DETERMINISTIC] ${bellCurveResult.reasoning}`,
      urgency: bellCurveResult.lockPercent > 80 ? 'high' : 'medium',
      riskLevel: 'balanced',
      positionVersion: pos.positionVersion,
    };
  }

  // No adjustment needed yet
  console.log(`[RiskMgmt] ⏸️  No stop adjustment needed yet (position not profitable enough or change too small).`);
  return {
    action: 'HOLD_BRACKETS',
    newStopLoss: null,
    newTarget: null,
    reasoning: 'Position not profitable enough for bell curve adjustment, or change would be < 1 tick.',
    urgency: 'low',
    riskLevel: 'conservative',
    positionVersion: pos.positionVersion,
  };
}
