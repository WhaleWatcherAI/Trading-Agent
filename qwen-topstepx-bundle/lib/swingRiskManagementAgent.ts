/**
 * Swing Trading Risk Management Agent
 *
 * Designed for multi-day swing trades with wider stops and targets
 * More patient approach - allows for larger intraday fluctuations
 * Focus on daily/4H structure rather than minute-by-minute action
 */

import { ActivePosition } from './executionManager';
import { TopstepXFuturesBar } from './topstepx';
import { ollamaOpenAICompat } from './ollamaClient';

const SWING_MODEL = process.env.OLLAMA_MODEL || 'qwen2.5:7b';

export interface SwingRiskManagementDecision {
  action: 'HOLD_POSITION' | 'ADJUST_STOP' | 'ADJUST_TARGET' | 'ADJUST_BOTH' | 'CLOSE_POSITION';
  newStopLoss: number | null;
  newTarget: number | null;
  reasoning: string;
  urgency: 'low' | 'medium' | 'high';
  riskLevel: 'conservative' | 'balanced' | 'aggressive';
  daysInTrade: number;
  positionVersion?: number;
}

interface SwingMarketSnapshot {
  currentPrice: number;
  dailyBars: TopstepXFuturesBar[];
  fourHourBars: TopstepXFuturesBar[];
  dailyTrend: 'bullish' | 'bearish' | 'neutral';
  multiDayVolumeProfile?: {
    compositePoc: number;
    compositeVah: number;
    compositeVal: number;
    hvns: number[];
    lvns: number[];
  };
  marketProfile?: {
    poc: number;
    vah: number;
    val: number;
  };
  swingLevels: number[];
}

/**
 * Analyze swing position risk - more patient than day trading version
 */
export async function analyzeSwingPositionRisk(
  position: ActivePosition,
  market: SwingMarketSnapshot
): Promise<SwingRiskManagementDecision> {
  console.log(`[SwingRiskMgmt] 🎯 Analyzing SWING position risk for ${position.side.toUpperCase()} position...`);

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

  // Calculate days in trade
  const entryTime = new Date(position.entryTime);
  const now = new Date();
  const daysInTrade = (now.getTime() - entryTime.getTime()) / (1000 * 60 * 60 * 24);

  const prompt = `You are a SWING TRADING RISK MANAGER for multi-day position holds.

🎯 SWING TRADING PHILOSOPHY:
Your role is different from day trading - you manage positions that are meant to capture LARGER moves over DAYS, not minutes.

Key Principles:
1. **PATIENCE IS KEY** - Don't react to every intraday fluctuation. Focus on daily/4H structure.
2. **WIDER STOPS** - Swing trades need room to breathe. Stops typically 50-100 points for S&P futures.
3. **MULTI-DAY PERSPECTIVE** - A position underwater by 10 points intraday is normal. Look at daily structure.
4. **PROTECT BIG GAINS** - Once up 50+ points, START trailing stops more actively.
5. **RESPECT DAILY STRUCTURE** - Only close if daily trend breaks or key swing level violated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CURRENT SWING POSITION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Direction: ${position.side.toUpperCase()}
Entry Price: ${position.entryPrice.toFixed(2)}
Current Price: ${market.currentPrice.toFixed(2)}
Current P&L: ${profitLossPoints >= 0 ? '+' : ''}${profitLossPoints.toFixed(2)} pts (${profitLossPercent >= 0 ? '+' : ''}${profitLossPercent.toFixed(2)}%)

Current Stop Loss: ${position.stopLoss.toFixed(2)} (${distanceToStop.toFixed(2)} pts from entry)
Current Target: ${position.target.toFixed(2)} (${distanceToTarget.toFixed(2)} pts from entry)
Risk:Reward Ratio: 1:${riskRewardRatio.toFixed(2)}

Progress to Target: ${percentToTarget.toFixed(1)}% (${profitLossPoints.toFixed(2)} of ${distanceToTarget.toFixed(2)} pts)
Days in Trade: ${daysInTrade.toFixed(2)} days
Entry Time: ${position.entryTime}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 HIGHER TIMEFRAME MARKET CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Daily Trend: ${market.dailyTrend.toUpperCase()}

Recent Daily Bars (last 5 days):
${market.dailyBars && market.dailyBars.length > 0 ? market.dailyBars.slice(-5).map((bar, i) =>
  `  Day ${i + 1}: O: ${bar.open.toFixed(2)}, H: ${bar.high.toFixed(2)}, L: ${bar.low.toFixed(2)}, C: ${bar.close.toFixed(2)}`
).join('\n') : '  No daily data available'}

Recent 4-Hour Bars (last 6 bars):
${market.fourHourBars && market.fourHourBars.length > 0 ? market.fourHourBars.slice(-6).map((bar, i) =>
  `  Bar ${i + 1}: O: ${bar.open.toFixed(2)}, H: ${bar.high.toFixed(2)}, L: ${bar.low.toFixed(2)}, C: ${bar.close.toFixed(2)}`
).join('\n') : '  No 4H data available'}

${market.multiDayVolumeProfile ? `Multi-Day Volume Profile:
- Composite POC: ${market.multiDayVolumeProfile.compositePoc.toFixed(2)}
- Composite VAH: ${market.multiDayVolumeProfile.compositeVah.toFixed(2)}
- Composite VAL: ${market.multiDayVolumeProfile.compositeVal.toFixed(2)}
- High Volume Nodes: ${market.multiDayVolumeProfile.hvns.slice(0, 3).map(h => h.toFixed(2)).join(', ')}
- Low Volume Nodes: ${market.multiDayVolumeProfile.lvns.slice(0, 3).map(l => l.toFixed(2)).join(', ')}` : ''}

${market.marketProfile ? `Market Profile (Time at Price):
- POC (Most Time): ${market.marketProfile.poc.toFixed(2)}
- VAH: ${market.marketProfile.vah.toFixed(2)}
- VAL: ${market.marketProfile.val.toFixed(2)}` : ''}

Key Swing Levels: ${market.swingLevels.slice(0, 5).map(l => l.toFixed(2)).join(', ')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR SWING RISK MANAGEMENT TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze this SWING trade and decide on adjustments:

1. **STOP LOSS MANAGEMENT** (Patient approach for swing trades):
   - DON'T react to intraday noise (<20 point moves on MES are normal)
   - Move stop to breakeven ONLY after 30+ points profit secured
   - Trail stops based on DAILY swing lows/highs, not 5-minute action
   - Consider multi-day volume profile support/resistance
   - Widen stop if daily trend still intact and just normal pullback

2. **TARGET MANAGEMENT** (Capture big moves):
   - Initial targets usually 50-100 points for swing trades
   - Extend targets if daily trend accelerating and breaking key levels
   - Take partial profits at major resistance/support zones (HVNs, POC)
   - Hold for multi-day moves - don't exit just because of 1-2 hour consolidation
   - Close entire position if daily trend reverses (e.g., daily close below key EMA)

3. **SWING TRADE CONSIDERATIONS**:
   - **Daily Structure** - Most important. Is daily trend intact?
   - **Multi-Day Profile** - Where is price relative to composite POC/VAH/VAL?
   - **Time in Trade** - Swings can take 3-7 days. Don't rush.
   - **Swing Levels** - Has a major swing high/low been violated?
   - **4H Structure** - Confirms daily or showing reversal?
   - **Intraday Noise** - IGNORE minor fluctuations. Think in daily candles.

4. **WHEN TO CLOSE** (These are the ONLY reasons to exit early):
   - Daily candle closes against you beyond key swing level
   - Multi-day trend reversal confirmed (e.g., daily EMA crossover)
   - Price breaks composite VAL/VAH in wrong direction with volume
   - Target reached or very close (95%+)
   - Position underwater >70 points (stop should have hit by then)

RESPOND WITH ONLY VALID JSON:
{
  "action": "HOLD_POSITION" | "ADJUST_STOP" | "ADJUST_TARGET" | "ADJUST_BOTH" | "CLOSE_POSITION",
  "newStopLoss": <number or null>,
  "newTarget": <number or null>,
  "reasoning": "<detailed explanation focusing on daily/4H structure>",
  "urgency": "low" | "medium" | "high",
  "riskLevel": "conservative" | "balanced" | "aggressive",
  "daysInTrade": ${daysInTrade.toFixed(2)}
}

🎯 SWING TRADING MINDSET:
- **DEFAULT TO HOLDING** - Swing trades are meant to run. Don't over-manage.
- **Daily structure trumps hourly** - A few hours of weakness means nothing if daily trend intact
- **Room to breathe** - Stops should be 50-100 points, not 10-20 points
- **Patience wins** - Best swing trades take 5-7 days to play out fully
- **Trail slowly** - Only trail after significant profit (30+ points), use daily swing points

EXAMPLES OF PROPER SWING RISK MANAGEMENT:
✅ +15pts profit, daily trend intact → HOLD_POSITION (too early to adjust, let it run)
✅ +60pts profit, daily uptrend continuing → ADJUST_STOP to +40pts (lock 2/3 of gain), hold target
✅ +50pts profit, price at composite VAH resistance → ADJUST_BOTH: stop to +35pts, target to next resistance
✅ -10pts underwater, daily trend intact, only 1 day in trade → HOLD_POSITION (normal intraday noise)
✅ Daily close below key swing low → CLOSE_POSITION (daily structure broken)
✅ +80pts profit, approaching target, 5 days in trade → Take profits, close or move stop very tight
❌ +10pts profit, 2 hours of consolidation → DON'T adjust! Way too soon for swing trade
❌ -15pts underwater after 4 hours → DON'T panic! Check daily structure first
❌ Trailing stop after every 5-point move → TOO AGGRESSIVE for swing trading

🛡️ REMEMBER: Swing trades need TIME and ROOM. Don't day-trade a swing position!`;

  try {
    console.log(`[SwingRiskMgmt] 💭 Calling Qwen via Ollama for swing risk analysis...`);

    const completion = await ollamaOpenAICompat.chat.completions.create({
      model: SWING_MODEL,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 1,
      max_tokens: 16000,
    });

    const fullContent = completion.choices?.[0]?.message?.content || '';

    if (!fullContent) {
      console.log(`[SwingRiskMgmt] ⚠️ Empty response from Qwen`);
      throw new Error('No response from swing risk management agent');
    }

    console.log(`[SwingRiskMgmt] 📝 Received response (${fullContent.length} chars)`);

    // Parse JSON from response
    const decision = parseSwingRiskManagementResponse(fullContent, daysInTrade);

    console.log(`[SwingRiskMgmt] ✅ Decision: ${decision.action}`);
    console.log(`[SwingRiskMgmt] 📊 New Stop: ${decision.newStopLoss?.toFixed(2) || 'unchanged'} | New Target: ${decision.newTarget?.toFixed(2) || 'unchanged'}`);
    console.log(`[SwingRiskMgmt] 💡 Reasoning: ${decision.reasoning.substring(0, 150)}...`);

    return decision;

  } catch (error: any) {
    console.error('[SwingRiskMgmt] ❌ Error analyzing swing position risk:', error.message);

    // Return conservative default: hold position (swing trades need patience)
    return {
      action: 'HOLD_POSITION',
      newStopLoss: null,
      newTarget: null,
      reasoning: `Error occurred during swing risk analysis: ${error.message}. Maintaining current position for swing trade.`,
      urgency: 'low',
      riskLevel: 'conservative',
      daysInTrade,
    };
  }
}

/**
 * Parse swing risk management response from AI
 */
function parseSwingRiskManagementResponse(content: string, daysInTrade: number): SwingRiskManagementDecision {
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
      urgency: parsed.urgency || 'low',
      riskLevel: parsed.riskLevel || 'balanced',
      daysInTrade: parsed.daysInTrade || daysInTrade,
      positionVersion: parsed.positionVersion,
    };

  } catch (error: any) {
    console.error('[SwingRiskMgmt] Failed to parse response:', error.message);
    throw error;
  }
}
