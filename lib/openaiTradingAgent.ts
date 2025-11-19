import OpenAI from 'openai';
import { fabioPlaybook, MarketState, SetupModel } from './fabioPlaybook';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Real-time futures market data from TopStepX
 * This is the actual data being streamed from the Fabio agent
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

  // Order flow absorption/exhaustion
  orderFlow: {
    buyAbsorption: number;  // 0-1 scale
    sellAbsorption: number; // 0-1 scale
    buyExhaustion: number;  // 0-1 scale
    sellExhaustion: number; // 0-1 scale
    bigPrints: Array<{
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
}

/**
 * OpenAI Decision based on Fabio's 3 decision layers:
 * 1. Market State (balanced vs imbalanced)
 * 2. Location (price relative to volume profile)
 * 3. Order Flow Aggression (CVD, absorption, exhaustion)
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
  riskRewardRatio: number | null;
  riskPercent: number; // 0.25-0.5 per Fabio rules
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

    const response = await openai.chat.completions.create({
      model: 'gpt-4-turbo',
      temperature: 0.7,
      max_tokens: 1000,
      messages: [
        {
          role: 'system',
          content: `You are an expert intraday futures trader specializing in Fabio's 3-layer decision framework.

Your trading philosophy: The market is an auction oscillating between BALANCE (rotation around fair value) and IMBALANCE (directional discovery away from prior value).

CRITICAL RULE: Only trade when ALL 3 decision layers align. If any layer is weak or conflicting, recommend HOLD.

Decision Layers:
1. MARKET STATE - Assess if balanced or out-of-balance, detect failed breakouts
2. LOCATION - Identify WHERE price is relative to volume profile (POC, VAH, VAL, LVNs)
3. ORDER FLOW AGGRESSION - Confirm CVD direction, absorption/exhaustion, big prints alignment

Setup Models:
- TREND CONTINUATION: Out-of-balance + Entry at LVN on impulse + Order flow confirms direction
- MEAN REVERSION: Failed breakout + Reclaim LVN + Breakout-side aggression failing

Risk Management (STRICT):
- Risk per trade: 0.25%-0.5% only
- Stops placed just beyond aggressive prints + 1-2 ticks (NEVER widen)
- Targets are balance POC or next balance (no stretching)

You MUST analyze this data systematically through all 3 layers before deciding.
Respond ONLY in JSON format with all required fields for execution.`,
        },
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    const content = response.choices[0]?.message?.content || '';
    const decision = parseOpenAIResponse(content);

    console.log(`✅ [OpenAI] ${decision.decision} @ ${decision.entryPrice} | Confidence: ${decision.confidence}%`);

    return decision;
  } catch (error) {
    console.error('❌ [OpenAI] Analysis failed:', error);
    throw error;
  }
}

/**
 * Build a detailed market analysis prompt for OpenAI
 * Teaches GPT-4 Fabio's 3-layer decision framework
 */
function buildAnalysisPrompt(
  marketData: FuturesMarketData,
  recentCandles: any[],
  currentPrice: number
): string {
  const cvdTrend = marketData.cvd.trend === 'up' ? '🟢 BULLISH' :
                   marketData.cvd.trend === 'down' ? '🔴 BEARISH' : '⚪ NEUTRAL';

  const candlesSummary = recentCandles
    .map((c) => `${new Date(c.timestamp).toLocaleTimeString()}: O=${c.open}, C=${c.close}, H=${c.high}, L=${c.low}`)
    .join('\n  ');

  const bigPrints = marketData.orderFlow.bigPrints
    .slice(-5)
    .map((p) => `${p.side.toUpperCase()} ${p.size} @ ${p.price}`)
    .join(', ') || 'None';

  const priceVsProfile = currentPrice > marketData.volumeProfile.poc ? 'above POC' : 'below POC';

  return `
FABIO'S 3-LAYER TRADING DECISION FRAMEWORK
============================================

The market is an auction oscillating between BALANCE (rotation around fair value) and IMBALANCE (directional discovery away from prior value).
Only trade when ALL 3 decision layers align:

LAYER 1: MARKET STATE
- Balanced: Price rotating around POC, no clear directional bias
- Out-of-Balance Uptrend: Clear directional displacement above prior value area, impulse leg up
- Out-of-Balance Downtrend: Clear directional displacement below prior value area, impulse leg down
- Failed Breakout Above: Multiple attempts to trade above VAH that fail, price reclaims back into value
- Failed Breakout Below: Multiple attempts to trade below VAL that fail, price reclaims back into value

LAYER 2: LOCATION (Price relative to Volume Profile structure)
- Location defines WHERE entries should happen
- POC: Point of control, highest volume level (fair value)
- VAH/VAL: Value area boundaries
- LVNs: Low Volume Nodes inside impulse or reclaim legs (reaction zones)
- Entries at LVNs inside the dominant trend direction = best probability

LAYER 3: ORDER FLOW AGGRESSION (CVD, absorption, exhaustion)
- CVD Trend: Direction of cumulative volume delta (buyers vs sellers in control)
- Absorption: Buyers/sellers holding price levels (absorption = strength)
- Exhaustion: Momentum losing steam (aggression drying up)
- Aggression at entry = market orders hitting the book with clear bias
- Look for: large prints, footprint imbalances, sustained CVD slope in trade direction

SETUP MODELS:
1. TREND CONTINUATION: Out-of-balance → Find LVN on impulse leg → Wait for retest → Confirm aggression → Enter direction of trend
   - Best in: NewYork session, trending days
   - Target: POC of next balance area

2. MEAN REVERSION: Failed breakout → Reclaim leg → Find LVN on reclaim → Confirm breakout side failing → Enter back to POC
   - Best in: London/Other sessions, rotational days
   - Target: Balance POC without stretching

RISK MANAGEMENT:
- Risk per trade: 0.25% - 0.5% of account
- Stop placement: Just beyond aggressive prints + 1-2 tick buffer
- No stop widening ever - if structure fails, exit
- Break-even rule: Move stop to BE early when CVD confirms strong direction

---

CURRENT MARKET DATA for ${marketData.symbol}:
Time: ${marketData.timestamp}
Current Price: ${currentPrice}

=== PRICE ACTION (Last 5 candles - 25 minutes) ===
${candlesSummary}

=== ORDER FLOW (CUMULATIVE VOLUME DELTA) ===
CVD Trend: ${cvdTrend}
CVD Value: ${marketData.cvd.value.toFixed(2)}
CVD Structure (OHLC): O=${marketData.cvd.ohlc.open}, H=${marketData.cvd.ohlc.high}, L=${marketData.cvd.ohlc.low}, C=${marketData.cvd.ohlc.close}
(CVD rising = buyers in control, CVD falling = sellers in control)

=== ORDER FLOW AGGRESSION (Absorption/Exhaustion) ===
Buy Absorption: ${(marketData.orderFlow.buyAbsorption * 100).toFixed(1)}% (buyers holding price)
Sell Absorption: ${(marketData.orderFlow.sellAbsorption * 100).toFixed(1)}% (sellers holding price)
Buy Exhaustion: ${(marketData.orderFlow.buyExhaustion * 100).toFixed(1)}% (buy momentum weakening)
Sell Exhaustion: ${(marketData.orderFlow.sellExhaustion * 100).toFixed(1)}% (sell momentum weakening)

Recent Large Prints (Market Orders): ${bigPrints}

=== VOLUME PROFILE STRUCTURE ===
POC (Fair Value): ${marketData.volumeProfile.poc}
VAH (Value Area High): ${marketData.volumeProfile.vah}
VAL (Value Area Low): ${marketData.volumeProfile.val}
LVNs (Reaction Zones): ${marketData.volumeProfile.lvns.join(', ')}
Session High: ${marketData.volumeProfile.sessionHigh}
Session Low: ${marketData.volumeProfile.sessionLow}
Current Price vs POC: ${priceVsProfile}

=== DETECTED MARKET STATE ===
State: ${marketData.marketState.state}
Buyers Control: ${(marketData.marketState.buyersControl * 100).toFixed(1)}%
Sellers Control: ${(marketData.marketState.sellersControl * 100).toFixed(1)}%
Order Flow Confirmed: ${marketData.orderFlowConfirmed}

=== ACCOUNT ===
Balance: $${marketData.account.balance.toFixed(2)}
Current Position: ${marketData.account.position === 0 ? 'FLAT' : `${marketData.account.position > 0 ? 'LONG' : 'SHORT'} ${Math.abs(marketData.account.position)} contracts`}
Unrealized P&L: $${marketData.account.unrealizedPnL.toFixed(2)}

ANALYSIS TASK:
Analyze all 3 layers of Fabio's framework:
1. MARKET STATE - Is the market balanced or out-of-balance? Is there a failed breakout?
2. LOCATION - Where is price relative to volume profile? Is it at an LVN?
3. ORDER FLOW - Is there clear aggression (CVD + absorption) confirming the directional bias?

Determine which setup model applies (trend continuation vs mean reversion).
Only provide a BUY or SELL decision if ALL 3 layers align with strong order flow confirmation.
Otherwise recommend HOLD.

Format your response as valid JSON with EXACTLY these keys:
{
  "decision": "BUY|SELL|HOLD",
  "confidence": 0-100,
  "marketState": "balanced|out_of_balance_uptrend|out_of_balance_downtrend|balanced_with_failed_breakout_above|balanced_with_failed_breakout_below",
  "location": "at_poc|at_vah|at_val|at_lvn|above_poc|below_poc",
  "setupModel": "trend_continuation|mean_reversion|null",
  "entryPrice": null or number,
  "stopLoss": null or number,
  "target": null or number,
  "riskRewardRatio": null or number,
  "riskPercent": 0.25-0.5,
  "reasoning": "Explain which layers aligned, why this setup triggered, what order flow confirms it"
}
`;
}

/**
 * Parse OpenAI's JSON response
 * Extracts all Fabio decision layers from the response
 */
function parseOpenAIResponse(content: string): OpenAITradingDecision {
  try {
    // Extract JSON from response (sometimes wrapped in markdown)
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No JSON found in response');
    }

    const parsed = JSON.parse(jsonMatch[0]);

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

    return {
      decision: parsed.decision?.toUpperCase() || 'HOLD',
      confidence: Math.min(100, Math.max(0, parsed.confidence || 50)),
      marketState,
      location,
      setupModel,
      entryPrice: parsed.entryPrice || null,
      stopLoss: parsed.stopLoss || null,
      target: parsed.target || null,
      reasoning: parsed.reasoning || 'No reasoning provided',
      riskRewardRatio: parsed.riskRewardRatio || null,
      riskPercent,
    };
  } catch (error) {
    console.error('Failed to parse OpenAI response:', error);
    return {
      decision: 'HOLD',
      confidence: 0,
      marketState: 'balanced',
      location: 'at_poc',
      setupModel: null,
      entryPrice: null,
      stopLoss: null,
      target: null,
      reasoning: 'Failed to parse AI response',
      riskRewardRatio: null,
      riskPercent: 0.35,
    };
  }
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
