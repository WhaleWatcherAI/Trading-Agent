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
 * - Starts immediately when position is filled
 * - Moves FORWARD from entry price toward take profit as price moves in our favor
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

  // Calculate progress: 0% at entry, 100% at take profit
  const percentToTarget = (profitLossPoints / distanceToTarget) * 100;

  // Not profitable yet - don't move stop
  if (profitLossPoints <= 0) {
    return null;
  }

  // SMOOTH BELL CURVE INTERPOLATION
  // Uses a cubic easing function for smooth, gradual tightening
  // Formula: y = x^3 for smooth acceleration
  // Maps percent to target (0-100%) to lock percent (0-95%)
  //
  // This creates a smooth curve that:
  // - Starts slow (gives room to breathe early)
  // - Accelerates in the middle
  // - Tightens aggressively near target

  let lockPercent: number;

  if (percentToTarget <= 0) {
    lockPercent = 0;
  } else if (percentToTarget >= 100) {
    lockPercent = 95; // Max 95% lock at target
  } else {
    // Cubic easing: smooth acceleration
    // Normalized progress (0 to 1)
    const progress = percentToTarget / 100;
    // Apply cubic easing: progress^3
    const eased = Math.pow(progress, 3);
    // Scale to max 95%
    lockPercent = eased * 95;
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

  // Only move stop if change is significant enough (at least 1 tick)
  // This prevents constant micro-adjustments on every price tick
  const stopChange = Math.abs(newStop - currentStop);
  const minChange = tickSize * 1; // Minimum 1 tick movement

  if (stopChange < minChange) {
    return null;
  }

  const profitFromEntry = pos.side === 'long'
    ? newStop - pos.entryPrice
    : pos.entryPrice - newStop;

  const reasoning = `Smooth bell curve: ${percentToTarget.toFixed(1)}% to target → ${lockPercent.toFixed(1)}% lock. Moving stop ${stopChange.toFixed(2)}pts from ${currentStop.toFixed(2)} to ${newStop.toFixed(2)} (+${profitFromEntry.toFixed(2)}pts profit locked).`;

  return { newStop, lockPercent, reasoning };
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
