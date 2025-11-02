import { Strategy } from './scoringEngine';

// Re-export Strategy for convenience
export type { Strategy };

export interface StrategyParams {
  strategy: Strategy;
  targetDelta: number; // Desired delta for options (0-1)
  dteRange: { min: number; max: number }; // Days to expiration
  strikeBias: 'ATM' | 'ITM' | 'OTM'; // Strike preference
  minOpenInterest: number;
  minVolume: number;
  holdTimeEstimate: { minimum: string; target: string; maximum: string };
}

export const STRATEGY_PARAMS: Record<Strategy, StrategyParams> = {
  scalp: {
    strategy: 'scalp',
    targetDelta: 0.60, // Slightly ITM for quick moves
    dteRange: { min: 0, max: 2 }, // 0-2 DTE
    strikeBias: 'ATM',
    minOpenInterest: 100,
    minVolume: 50,
    holdTimeEstimate: {
      minimum: '15 minutes',
      target: '1-2 hours',
      maximum: '4 hours',
    },
  },
  intraday: {
    strategy: 'intraday',
    targetDelta: 0.50, // ATM for maximum gamma
    dteRange: { min: 0, max: 7 }, // 0-7 DTE (same week)
    strikeBias: 'ATM',
    minOpenInterest: 200,
    minVolume: 100,
    holdTimeEstimate: {
      minimum: '2 hours',
      target: '4-6 hours',
      maximum: 'end of day',
    },
  },
  swing: {
    strategy: 'swing',
    targetDelta: 0.40, // Slightly OTM for better risk/reward
    dteRange: { min: 7, max: 30 }, // 1-4 weeks
    strikeBias: 'OTM',
    minOpenInterest: 500,
    minVolume: 200,
    holdTimeEstimate: {
      minimum: '2 days',
      target: '5-7 days',
      maximum: '10 days',
    },
  },
  leap: {
    strategy: 'leap',
    targetDelta: 0.70, // ITM for less theta decay
    dteRange: { min: 60, max: 730 }, // 2 months to 2 years
    strikeBias: 'ITM',
    minOpenInterest: 500,
    minVolume: 50, // Lower volume OK for LEAPs
    holdTimeEstimate: {
      minimum: '30 days',
      target: '90-180 days',
      maximum: '365 days',
    },
  },
};

/**
 * Get strategy parameters for a given strategy
 */
export function getStrategyParams(strategy: Strategy): StrategyParams {
  return STRATEGY_PARAMS[strategy];
}

/**
 * Determine if a trade meets the minimum threshold for recommendation
 */
export function meetsThreshold(
  bullishScore: number,
  bearishScore: number,
  threshold: number = 7
): { passes: boolean; direction: 'bullish' | 'bearish' | null; score: number } {
  if (bullishScore >= threshold && bullishScore > bearishScore) {
    return { passes: true, direction: 'bullish', score: bullishScore };
  }

  if (bearishScore >= threshold && bearishScore > bullishScore) {
    return { passes: true, direction: 'bearish', score: bearishScore };
  }

  return { passes: false, direction: null, score: Math.max(bullishScore, bearishScore) };
}

/**
 * Calculate position sizing based on confidence and strategy
 */
export function calculatePositionSize(
  confidence: number,
  accountSize: number,
  strategy: Strategy
): {
  contractsRecommended: number;
  capitalAllocated: number;
  riskPercentage: number;
} {
  // Risk percentages per strategy
  const riskMap: Record<Strategy, number> = {
    scalp: 0.02,    // 2% of account
    intraday: 0.03, // 3% of account
    swing: 0.05,    // 5% of account
    leap: 0.10,     // 10% of account
  };

  const baseRisk = riskMap[strategy];

  // Adjust risk by confidence (50-100% confidence scales 0.5x-1.5x)
  const confidenceMultiplier = 0.5 + (confidence / 100);
  const adjustedRisk = baseRisk * confidenceMultiplier;

  const capitalAllocated = accountSize * adjustedRisk;

  // Assume average contract costs $200-500 depending on strategy
  const estimatedContractCost: Record<Strategy, number> = {
    scalp: 200,
    intraday: 300,
    swing: 400,
    leap: 2000,
  };

  const contractsRecommended = Math.floor(capitalAllocated / estimatedContractCost[strategy]);

  return {
    contractsRecommended: Math.max(1, contractsRecommended),
    capitalAllocated,
    riskPercentage: adjustedRisk * 100,
  };
}

/**
 * Generate time-sensitive urgency level
 */
export function calculateUrgency(
  strategy: Strategy,
  mostRecentDataAge: number, // minutes
  catalyst?: string
): {
  level: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  reason: string;
} {
  // Urgency thresholds by strategy (minutes)
  const urgencyMap: Record<Strategy, { high: number; medium: number }> = {
    scalp: { high: 5, medium: 15 },      // Very time-sensitive
    intraday: { high: 15, medium: 60 },  // Moderately time-sensitive
    swing: { high: 120, medium: 360 },   // Less time-sensitive
    leap: { high: 1440, medium: 4320 },  // Not time-sensitive (days)
  };

  const thresholds = urgencyMap[strategy];

  let level: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT' = 'LOW';
  let reason = 'Standard entry timing';

  // Catalyst override
  if (catalyst && (catalyst.includes('earnings') || catalyst.includes('FDA') || catalyst.includes('breakout'))) {
    level = 'URGENT';
    reason = `Time-sensitive catalyst: ${catalyst}`;
    return { level, reason };
  }

  // Age-based urgency
  if (mostRecentDataAge <= thresholds.high) {
    level = 'HIGH';
    reason = `Fresh flow data (${mostRecentDataAge}min old) - act quickly`;
  } else if (mostRecentDataAge <= thresholds.medium) {
    level = 'MEDIUM';
    reason = `Recent signal (${mostRecentDataAge}min old) - good entry window`;
  } else {
    level = 'LOW';
    reason = `Older signal (${Math.round(mostRecentDataAge / 60)}hr old) - less urgent`;
  }

  return { level, reason };
}

/**
 * Generate risk warnings based on strategy and market conditions
 */
export function generateRiskWarnings(
  strategy: Strategy,
  ivRank: number | undefined,
  vix: number | undefined,
  nearResistance: boolean,
  technicalDivergence: boolean
): string[] {
  const warnings: string[] = [];

  // IV warnings
  if (ivRank !== undefined) {
    if (ivRank > 75 && (strategy === 'scalp' || strategy === 'intraday')) {
      warnings.push(`High IV rank (${ivRank.toFixed(0)}%) increases option premium - consider debit spreads`);
    }
    if (ivRank < 25 && strategy === 'leap') {
      warnings.push(`Low IV rank (${ivRank.toFixed(0)}%) - good for long-term option purchases`);
    }
  }

  // VIX warnings
  if (vix !== undefined) {
    if (vix > 30) {
      warnings.push(`Elevated VIX (${vix.toFixed(1)}) indicates high market volatility - size positions accordingly`);
    }
    if (vix < 12 && strategy === 'scalp') {
      warnings.push(`Low VIX (${vix.toFixed(1)}) may mean limited intraday movement`);
    }
  }

  // Technical warnings
  if (nearResistance) {
    warnings.push(`Price near resistance - watch for rejection or breakout confirmation`);
  }

  if (technicalDivergence) {
    warnings.push(`Technical indicators show divergence from flow data - proceed with caution`);
  }

  // Strategy-specific warnings
  if (strategy === 'scalp') {
    warnings.push(`Scalp trades require tight stops and active management - not suitable for hands-off trading`);
  }

  if (strategy === 'leap' && vix !== undefined && vix > 25) {
    warnings.push(`Consider waiting for volatility to settle before entering long-term positions`);
  }

  return warnings;
}

/**
 * Generate trading rules/guidelines for the strategy
 */
export function getStrategyGuidelines(strategy: Strategy): {
  entryRules: string[];
  exitRules: string[];
  managementRules: string[];
} {
  const guidelines = {
    scalp: {
      entryRules: [
        'Enter on confirmation of flow direction with technical alignment',
        'Wait for pullback to support/VWAP before entry',
        'Ensure tight bid-ask spread (<$0.10) for quick execution',
      ],
      exitRules: [
        'Take profit at 20-30% gain or predetermined price target',
        'Cut losses immediately at 15-20% loss',
        'Exit before major resistance levels',
        'Close all positions before market close',
      ],
      managementRules: [
        'Monitor position every 15-30 minutes',
        'Trail stop-loss as position moves in your favor',
        'Scale out at multiple profit targets (50% at 20%, 50% at 30%)',
      ],
    },
    intraday: {
      entryRules: [
        'Confirm entry with at least 2 bullish/bearish factors aligning',
        'Enter on retest of breakout/breakdown level',
        'Verify volume confirmation on the move',
      ],
      exitRules: [
        'Target 30-50% profit on premium',
        'Stop loss at 25% premium loss',
        'Exit by 3:30 PM ET to avoid after-hours risk',
      ],
      managementRules: [
        'Check position every 1-2 hours',
        'Adjust stops to breakeven once up 25%',
        'Consider adding to winning positions on pullbacks',
      ],
    },
    swing: {
      entryRules: [
        'Enter when multiple timeframes align (daily + 4hr)',
        'Wait for confirmation day after signal',
        'Consider scaling in over 2-3 days',
      ],
      exitRules: [
        'Target 50-100% profit on premium',
        'Use 30% stop loss from entry',
        'Exit if thesis breaks or bearish reversal confirmed',
      ],
      managementRules: [
        'Review position daily at close',
        'Trail stops using 3-day ATR',
        'Take partial profits at 50% gain, let runners go',
      ],
    },
    leap: {
      entryRules: [
        'Enter during volatility contraction (IV rank <50)',
        'Scale in over weeks, not days',
        'Ensure strong fundamental thesis supports direction',
      ],
      exitRules: [
        'Target 100%+ profit over months',
        'Use 40-50% stop loss (wider for theta decay buffer)',
        'Exit if fundamental thesis changes',
      ],
      managementRules: [
        'Review weekly, not daily',
        'Roll options forward if needed (60 days before expiry)',
        'Consider taking profits at 100% and letting remainder run',
      ],
    },
  };

  return guidelines[strategy];
}
