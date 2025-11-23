import { Strategy } from './scoringEngine';
import { OptionContract } from './contractSelector';

export interface PromptContext {
  symbol: string;
  strategy: Strategy;
  direction: 'bullish' | 'bearish';
  bullishScore: number;
  bearishScore: number;
  confidence: number;

  // Factor contributions
  flowFactors: string[];
  newsFactors: string[];
  institutionalFactors: string[];
  structureFactors: string[];
  technicalFactors: string[];

  // Contract details
  contract: OptionContract;
  currentPrice: number;

  // Market context
  vix?: number;
  putCallRatio?: number;
  ivRank?: number;

  // Time context
  mostRecentDataAge: number; // minutes
}

/**
 * Generate comprehensive OpenAI prompt for trade analysis
 */
export function generateTradeAnalysisPrompt(context: PromptContext): string {
  const {
    symbol,
    strategy,
    direction,
    bullishScore,
    bearishScore,
    confidence,
    flowFactors,
    newsFactors,
    institutionalFactors,
    structureFactors,
    technicalFactors,
    contract,
    currentPrice,
    vix,
    putCallRatio,
    ivRank,
    mostRecentDataAge,
  } = context;

  const strategyDescription = getStrategyDescription(strategy);

  const prompt = `
You are an expert options trader analyzing a ${direction.toUpperCase()} ${strategy.toUpperCase()} trade opportunity for ${symbol}.

## CURRENT MARKET DATA
- Stock Price: $${currentPrice.toFixed(2)}
- VIX: ${vix?.toFixed(1) || 'N/A'}
- Put/Call Ratio: ${putCallRatio?.toFixed(2) || 'N/A'}
- IV Rank: ${ivRank?.toFixed(0) || 'N/A'}%
- Most Recent Signal: ${mostRecentDataAge} minutes ago

## SCORING ANALYSIS
- Bullish Score: ${bullishScore}/10
- Bearish Score: ${bearishScore}/10
- Confidence: ${confidence}%
- Dominant Direction: ${direction.toUpperCase()}

## FACTOR BREAKDOWN

### Options Flow (Weight: Primary)
${flowFactors.length > 0 ? flowFactors.map(f => `- ${f}`).join('\n') : '- No significant flow detected'}

### News & Catalysts
${newsFactors.length > 0 ? newsFactors.map(f => `- ${f}`).join('\n') : '- No major news'}

### Institutional Activity
${institutionalFactors.length > 0 ? institutionalFactors.map(f => `- ${f}`).join('\n') : '- No significant institutional trades'}

### Market Structure (GEX, Greeks, Volatility)
${structureFactors.length > 0 ? structureFactors.map(f => `- ${f}`).join('\n') : '- Neutral structure'}

### Technical Indicators
${technicalFactors.length > 0 ? technicalFactors.map(f => `- ${f}`).join('\n') : '- No strong technical signals'}

## RECOMMENDED CONTRACT
- Type: ${contract.type.toUpperCase()}
- Strike: $${contract.strike}
- Expiration: ${contract.expiration} (${contract.daysToExpiration} DTE)
- Premium: $${contract.mid.toFixed(2)}
- Delta: ${contract.delta.toFixed(2)}
- IV: ${(contract.impliedVolatility * 100).toFixed(1)}%

## STRATEGY CONTEXT
${strategyDescription}

## YOUR TASK
Provide a detailed trading analysis with the following structure:

1. **Primary Catalyst** (2-3 sentences)
   - Identify the single most important factor driving this trade
   - Include specific data points (dollar amounts, timeframes, percentages)
   - Explain why this is time-sensitive

2. **Supporting Factors** (3-5 bullet points)
   - List additional factors that strengthen the trade thesis
   - Quantify each factor with specific metrics
   - Note any confluence of signals

3. **Risk Considerations** (2-4 bullet points)
   - Identify key risks that could invalidate the trade
   - Mention any contradictory signals
   - Note market conditions that could cause issues

4. **Time Sensitivity** (1 sentence)
   - Rate urgency as LOW/MEDIUM/HIGH/URGENT
   - Explain why based on data age and catalyst timing

Keep your response concise but specific. Use exact numbers and timeframes. Focus on actionable insights.
`.trim();

  return prompt;
}

/**
 * Get strategy-specific context description
 */
function getStrategyDescription(strategy: Strategy): string {
  const descriptions = {
    scalp: `
SCALP Strategy: Ultra-short-term trades (15min - 4hrs)
- Focused on immediate price action and fresh flow data
- Requires tight risk management and active monitoring
- Capitalizes on intraday volatility and rapid moves
- Data older than 30 minutes is significantly less relevant
    `.trim(),

    intraday: `
INTRADAY Strategy: Same-day trades (2hrs - EOD)
- Balanced approach between scalping and swing trading
- Leverages intraday momentum and catalyst reactions
- Holds through minor pullbacks for larger moves
- Data from the past 1-2 hours is most critical
    `.trim(),

    swing: `
SWING Strategy: Multi-day positions (2-10 days)
- Captures medium-term trends and catalyst-driven moves
- Less sensitive to intraday noise
- Focuses on daily chart patterns and sustained momentum
- Data from today and yesterday both relevant
    `.trim(),

    leap: `
LEAP Strategy: Long-term positions (30-365+ days)
- Based on fundamental thesis and structural trends
- Low sensitivity to short-term noise
- Emphasizes favorable volatility for entry
- Considers sector rotation and macro trends
    `.trim(),
  };

  return descriptions[strategy];
}

/**
 * Generate a simpler prompt for quick reasoning (legacy support)
 */
export function generateQuickPrompt(
  symbol: string,
  direction: 'bullish' | 'bearish',
  keyFactors: string[]
): string {
  return `
Analyze this ${direction} trade for ${symbol} based on:
${keyFactors.map(f => `- ${f}`).join('\n')}

Provide 2-3 sentences explaining the trade thesis.
  `.trim();
}

/**
 * Generate prompt for multi-strategy comparison
 */
export function generateMultiStrategyPrompt(
  symbol: string,
  direction: 'bullish' | 'bearish',
  strategyScores: Record<Strategy, { bullishScore: number; bearishScore: number; confidence: number }>
): string {
  return `
You are analyzing ${symbol} for a ${direction.toUpperCase()} trade across multiple timeframes.

## STRATEGY SCORES
${Object.entries(strategyScores).map(([strat, scores]) => `
${strat.toUpperCase()}:
- Bullish: ${scores.bullishScore}/10
- Bearish: ${scores.bearishScore}/10
- Confidence: ${scores.confidence}%
`).join('\n')}

Provide a brief (2-3 sentences) summary of which strategies have the strongest setup and why.
  `.trim();
}

/**
 * Format the AI response into structured sections
 */
export function parseAIResponse(aiResponse: string): {
  primaryCatalyst: string;
  supportingFactors: string[];
  riskConsiderations: string[];
  timeSensitivity: string;
} {
  // Simple parsing - split by sections
  const sections = {
    primaryCatalyst: '',
    supportingFactors: [] as string[],
    riskConsiderations: [] as string[],
    timeSensitivity: '',
  };

  // This is a basic parser - in production, you'd want more robust parsing
  const lines = aiResponse.split('\n');
  let currentSection = '';

  for (const line of lines) {
    const lower = line.toLowerCase();

    if (lower.includes('primary catalyst') || lower.includes('1.')) {
      currentSection = 'primaryCatalyst';
      continue;
    } else if (lower.includes('supporting factors') || lower.includes('2.')) {
      currentSection = 'supportingFactors';
      continue;
    } else if (lower.includes('risk considerations') || lower.includes('3.')) {
      currentSection = 'riskConsiderations';
      continue;
    } else if (lower.includes('time sensitivity') || lower.includes('4.')) {
      currentSection = 'timeSensitivity';
      continue;
    }

    // Add content to current section
    const trimmed = line.trim();
    if (trimmed.length > 0) {
      if (currentSection === 'primaryCatalyst') {
        sections.primaryCatalyst += trimmed + ' ';
      } else if (currentSection === 'supportingFactors' && trimmed.startsWith('-')) {
        sections.supportingFactors.push(trimmed.substring(1).trim());
      } else if (currentSection === 'riskConsiderations' && trimmed.startsWith('-')) {
        sections.riskConsiderations.push(trimmed.substring(1).trim());
      } else if (currentSection === 'timeSensitivity') {
        sections.timeSensitivity += trimmed + ' ';
      }
    }
  }

  // Clean up
  sections.primaryCatalyst = sections.primaryCatalyst.trim();
  sections.timeSensitivity = sections.timeSensitivity.trim();

  return sections;
}
