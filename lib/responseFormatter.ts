import { Strategy } from './scoringEngine';
import { OptionContract, ContractRecommendation } from './contractSelector';

export interface TradeRecommendation {
  strategy: Strategy;
  symbol: string;
  sentiment: 'BULLISH' | 'BEARISH';
  bullishScore: number;
  bearishScore: number;
  confidence: number;

  contract: {
    ticker: string;
    type: 'CALL' | 'PUT';
    strike: number;
    expiration: string;
    daysToExpiration: number;
    currentPrice: number;
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    impliedVolatility: number;
    bidAskSpread: number;
    volume: number;
    openInterest: number;
  };

  alternativeContracts?: {
    ticker: string;
    strike: number;
    expiration: string;
    currentPrice: number;
    delta: number;
  }[];

  estimatedHoldTime: {
    minimum: string;
    target: string;
    maximum: string;
  };

  reasoning: {
    primaryCatalyst: string;
    supportingFactors: string[];
    riskConsiderations: string[];
    timeSensitivity: string;
  };

  factorBreakdown: {
    flow: { bullishScore: number; bearishScore: number; weight: number; contribution: string };
    news: { bullishScore: number; bearishScore: number; weight: number; contribution: string };
    institutional: { bullishScore: number; bearishScore: number; weight: number; contribution: string };
    structure: { bullishScore: number; bearishScore: number; weight: number; contribution: string };
    technical: { bullishScore: number; bearishScore: number; weight: number; contribution: string };
  };

  positionSizing: {
    contractsRecommended: number;
    capitalAllocated: number;
    riskPercentage: number;
  };

  tradingGuidelines: {
    entryRules: string[];
    exitRules: string[];
    managementRules: string[];
  };

  dataFreshness: {
    mostRecentFlow: string;
    mostRecentNews: string;
    lastPriceUpdate: string;
  };

  urgency: {
    level: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
    reason: string;
  };

  riskWarnings: string[];
}

export interface MultiStrategyResponse {
  symbol: string;
  analyzedAt: string;
  currentPrice: number;
  marketContext: {
    vix: number | undefined;
    putCallRatio: number | undefined;
    marketTide: 'bullish' | 'bearish' | 'neutral';
  };
  recommendations: TradeRecommendation[];
  summary: string;
}

/**
 * Format a single trade recommendation
 */
export function formatTradeRecommendation(
  symbol: string,
  strategy: Strategy,
  direction: 'bullish' | 'bearish',
  bullishScore: number,
  bearishScore: number,
  confidence: number,
  contractRec: ContractRecommendation,
  estimatedHoldTime: { minimum: string; target: string; maximum: string },
  reasoning: {
    primaryCatalyst: string;
    supportingFactors: string[];
    riskConsiderations: string[];
    timeSensitivity: string;
  },
  factorBreakdown: {
    flow: { bullishScore: number; bearishScore: number; weight: number };
    news: { bullishScore: number; bearishScore: number; weight: number };
    institutional: { bullishScore: number; bearishScore: number; weight: number };
    structure: { bullishScore: number; bearishScore: number; weight: number };
    technical: { bullishScore: number; bearishScore: number; weight: number };
  },
  positionSizing: {
    contractsRecommended: number;
    capitalAllocated: number;
    riskPercentage: number;
  },
  tradingGuidelines: {
    entryRules: string[];
    exitRules: string[];
    managementRules: string[];
  },
  dataFreshness: {
    mostRecentFlow: string;
    mostRecentNews: string;
    lastPriceUpdate: string;
  },
  urgency: {
    level: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
    reason: string;
  },
  riskWarnings: string[]
): TradeRecommendation {
  const contract = contractRec.contract;

  return {
    strategy,
    symbol,
    sentiment: direction === 'bullish' ? 'BULLISH' : 'BEARISH',
    bullishScore,
    bearishScore,
    confidence,
    contract: {
      ticker: contract.symbol,
      type: contract.type.toUpperCase() as 'CALL' | 'PUT',
      strike: contract.strike,
      expiration: contract.expiration,
      daysToExpiration: contract.daysToExpiration,
      currentPrice: contract.mid,
      delta: contract.delta,
      gamma: contract.gamma,
      theta: contract.theta,
      vega: contract.vega,
      impliedVolatility: contract.impliedVolatility,
      bidAskSpread: contract.ask - contract.bid,
      volume: contract.volume,
      openInterest: contract.openInterest,
    },
    alternativeContracts: contractRec.alternativeContracts.map(alt => ({
      ticker: alt.symbol,
      strike: alt.strike,
      expiration: alt.expiration,
      currentPrice: alt.mid,
      delta: alt.delta,
    })),
    estimatedHoldTime,
    reasoning,
    factorBreakdown: {
      flow: {
        ...factorBreakdown.flow,
        contribution: `${(factorBreakdown.flow.weight * 100).toFixed(0)}%`,
      },
      news: {
        ...factorBreakdown.news,
        contribution: `${(factorBreakdown.news.weight * 100).toFixed(0)}%`,
      },
      institutional: {
        ...factorBreakdown.institutional,
        contribution: `${(factorBreakdown.institutional.weight * 100).toFixed(0)}%`,
      },
      structure: {
        ...factorBreakdown.structure,
        contribution: `${(factorBreakdown.structure.weight * 100).toFixed(0)}%`,
      },
      technical: {
        ...factorBreakdown.technical,
        contribution: `${(factorBreakdown.technical.weight * 100).toFixed(0)}%`,
      },
    },
    positionSizing,
    tradingGuidelines,
    dataFreshness,
    urgency,
    riskWarnings,
  };
}

/**
 * Format multiple recommendations into a comprehensive response
 */
export function formatMultiStrategyResponse(
  symbol: string,
  currentPrice: number,
  marketContext: {
    vix: number | undefined;
    putCallRatio: number | undefined;
    marketTide: 'bullish' | 'bearish' | 'neutral';
  },
  recommendations: TradeRecommendation[],
  summary?: string
): MultiStrategyResponse {
  return {
    symbol,
    analyzedAt: new Date().toISOString(),
    currentPrice,
    marketContext,
    recommendations,
    summary: summary || generateAutoSummary(recommendations),
  };
}

/**
 * Auto-generate a summary if not provided
 */
function generateAutoSummary(recommendations: TradeRecommendation[]): string {
  if (recommendations.length === 0) {
    return 'No trade recommendations meet the threshold criteria (score >= 7).';
  }

  const strategyNames = recommendations.map(r => r.strategy.toUpperCase()).join(', ');
  const sentiment = recommendations[0].sentiment;
  const avgConfidence = Math.round(
    recommendations.reduce((sum, r) => sum + r.confidence, 0) / recommendations.length
  );

  return `Found ${recommendations.length} ${sentiment} trade setups across ${strategyNames} strategies with average confidence of ${avgConfidence}%. Review detailed analysis below.`;
}

/**
 * Format for console/text output (human-readable)
 */
export function formatForConsole(recommendation: TradeRecommendation): string {
  const output = `
╔════════════════════════════════════════════════════════════════════
║ ${recommendation.sentiment} ${recommendation.strategy.toUpperCase()} TRADE - ${recommendation.symbol}
╠════════════════════════════════════════════════════════════════════

📊 SCORES
   Bullish: ${recommendation.bullishScore}/10
   Bearish: ${recommendation.bearishScore}/10
   Confidence: ${recommendation.confidence}%

📋 CONTRACT RECOMMENDATION
   ${recommendation.contract.type}: ${recommendation.symbol} $${recommendation.contract.strike} ${recommendation.contract.expiration}
   Premium: $${recommendation.contract.currentPrice.toFixed(2)}
   Delta: ${recommendation.contract.delta.toFixed(2)} | IV: ${(recommendation.contract.impliedVolatility * 100).toFixed(1)}%
   Volume: ${recommendation.contract.volume.toLocaleString()} | OI: ${recommendation.contract.openInterest.toLocaleString()}
   DTE: ${recommendation.contract.daysToExpiration} days

⏱️  HOLD TIME
   Target: ${recommendation.estimatedHoldTime.target}
   Range: ${recommendation.estimatedHoldTime.minimum} - ${recommendation.estimatedHoldTime.maximum}

🎯 PRIMARY CATALYST
   ${recommendation.reasoning.primaryCatalyst}

✅ SUPPORTING FACTORS
${recommendation.reasoning.supportingFactors.map(f => `   • ${f}`).join('\n')}

⚠️  RISK CONSIDERATIONS
${recommendation.reasoning.riskConsiderations.map(f => `   • ${f}`).join('\n')}

🚨 URGENCY: ${recommendation.urgency.level}
   ${recommendation.urgency.reason}

📈 FACTOR BREAKDOWN
   Flow:          ${recommendation.factorBreakdown.flow.bullishScore}/${recommendation.factorBreakdown.flow.bearishScore} (${recommendation.factorBreakdown.flow.contribution})
   News:          ${recommendation.factorBreakdown.news.bullishScore}/${recommendation.factorBreakdown.news.bearishScore} (${recommendation.factorBreakdown.news.contribution})
   Institutional: ${recommendation.factorBreakdown.institutional.bullishScore}/${recommendation.factorBreakdown.institutional.bearishScore} (${recommendation.factorBreakdown.institutional.contribution})
   Structure:     ${recommendation.factorBreakdown.structure.bullishScore}/${recommendation.factorBreakdown.structure.bearishScore} (${recommendation.factorBreakdown.structure.contribution})
   Technical:     ${recommendation.factorBreakdown.technical.bullishScore}/${recommendation.factorBreakdown.technical.bearishScore} (${recommendation.factorBreakdown.technical.contribution})

💰 POSITION SIZING (example for $100k account)
   Contracts: ${recommendation.positionSizing.contractsRecommended}
   Capital: $${recommendation.positionSizing.capitalAllocated.toFixed(0)}
   Risk: ${recommendation.positionSizing.riskPercentage.toFixed(2)}%

${recommendation.riskWarnings.length > 0 ? `
⚠️  WARNINGS
${recommendation.riskWarnings.map(w => `   • ${w}`).join('\n')}
` : ''}

╚════════════════════════════════════════════════════════════════════
`.trim();

  return output;
}

/**
 * Format compact version for quick scanning
 */
export function formatCompact(recommendation: TradeRecommendation): string {
  return `
${recommendation.sentiment} ${recommendation.strategy.toUpperCase()} | ${recommendation.symbol} | Score: ${Math.max(recommendation.bullishScore, recommendation.bearishScore)}/10 (${recommendation.confidence}% conf)
Contract: ${recommendation.contract.type} $${recommendation.contract.strike} ${recommendation.contract.expiration} @ $${recommendation.contract.currentPrice.toFixed(2)}
Hold: ${recommendation.estimatedHoldTime.target} | Urgency: ${recommendation.urgency.level}
  `.trim();
}
