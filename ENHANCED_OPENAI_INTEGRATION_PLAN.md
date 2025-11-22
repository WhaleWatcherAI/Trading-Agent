# Enhanced OpenAI Integration - Implementation Plan

## What's Being Added

This brings the TypeScript OpenAI integration to 100% parity with the Python enhanced system from `SELF_LEARNING_SYSTEM_DOCS.md`.

## New Features

### 1. POC Cross Tracking ✅ CREATED
**File:** `lib/enhancedFeatures.ts`
- Tracks how many times price crosses POC in 5min, 15min, 30min windows
- Records time since last cross
- Determines current side (above/below POC)

```typescript
{
  count_last_5min: 2,
  count_last_15min: 5,
  count_last_30min: 12,
  time_since_last_cross_sec: 45.3,
  current_side: "above_poc"
}
```

### 2. Raw Market Statistics ✅ CREATED
**File:** `lib/enhancedFeatures.ts`
- Session range in ticks with historical percentiles
- Distance to POC in ticks
- Time spent in/above/below value area
- CVD slopes (5min, 15min) for regime detection

```typescript
{
  session_range_ticks: 120,
  session_range_percentile: 0.75, // 75th percentile
  distance_to_poc_ticks: 15,
  time_above_value_sec: 1800,
  time_below_value_sec: 600,
  time_in_value_sec: 2400,
  cvd_slope_5min: 0.8,
  cvd_slope_15min: 0.6
}
```

### 3. Performance Metrics ✅ CREATED
**File:** `lib/enhancedFeatures.ts`
- Win rate tracking
- Average P&L per trade
- Profit factor calculation
- Self-learning feedback loop

### 4. Historical Notes ✅ CREATED
**File:** `lib/enhancedFeatures.ts`
- Stores notes for future reference
- Context-specific learning
- "Notes to future self" feature

## Next Steps (TO DO)

### Step 1: Update FuturesMarketData Interface
**File:** `lib/openaiTradingAgent.ts`
Add these fields to the interface:
```typescript
export interface FuturesMarketData {
  // ... existing fields ...

  // NEW: POC Cross Statistics
  pocCrossStats: POCCrossStats;

  // NEW: Raw Market Statistics (for regime inference)
  marketStats: MarketStatistics;

  // NEW: Performance Feedback (self-learning)
  performance: PerformanceMetrics | null;

  // NEW: Historical Notes
  historicalNotes: HistoricalNote[];
}
```

### Step 2: Update System Prompt for Regime Inference
**File:** `lib/openaiTradingAgent.ts`

Add regime inference rules to the system prompt:
- **Trend Regime**: Range > 60th %ile, POC crosses < 10/30min, CVD slope > 0.5
- **Range Regime**: Range 30-70th %ile, POC crosses 10-25/30min
- **Chop Regime**: Range < 30th %ile, POC crosses > 25/30min, CVD slope < 0.1

### Step 3: Update buildFuturesMarketData()
**File:** `lib/fabioOpenAIIntegration.ts`

Initialize and use the enhanced feature trackers:
```typescript
// Global trackers (outside function)
const pocCrossTracker = new POCCrossTracker();
const marketStatsCalc = new MarketStatsCalculator();
const performanceTracker = new PerformanceTracker();
const notesManager = new HistoricalNotesManager();

// Inside buildFuturesMarketData()
const pocCrossStats = pocCrossTracker.update(currentPrice, volumeProfile.poc);
const marketStats = marketStatsCalc.calculate(currentPrice, volumeProfile.poc, ...);
const performance = performanceTracker.getMetrics();
const historicalNotes = notesManager.getRecentNotes(10);
```

### Step 4: Update Prompt Building
**File:** `lib/openaiTradingAgent.ts` - `buildAnalysisPrompt()`

Add new sections:
```
=== POC CROSS STATISTICS (Critical for Regime Detection) ===
POC Crosses (last 5min): ${pocCrossStats.count_last_5min}
POC Crosses (last 15min): ${pocCrossStats.count_last_15min}
POC Crosses (last 30min): ${pocCrossStats.count_last_30min}
Time Since Last Cross: ${pocCrossStats.time_since_last_cross_sec}s
Current Side: ${pocCrossStats.current_side}

=== RAW MARKET STATISTICS (for Regime Inference) ===
Session Range: ${marketStats.session_range_ticks} ticks (${marketStats.session_range_percentile * 100}th percentile)
Distance to POC: ${marketStats.distance_to_poc_ticks} ticks
Time Above Value: ${marketStats.time_above_value_sec}s
Time In Value: ${marketStats.time_in_value_sec}s
Time Below Value: ${marketStats.time_below_value_sec}s
CVD Slope (5min): ${marketStats.cvd_slope_5min}
CVD Slope (15min): ${marketStats.cvd_slope_15min}

REGIME INFERENCE GUIDELINES:
- TREND: Range > 60th %ile, Distance to POC > 20 ticks, POC crosses < 10/30min, CVD slope > 0.5
- RANGE: Range 30-70th %ile, POC crosses 10-25/30min, Price rotating VAH-VAL
- CHOP: Range < 30th %ile, Distance to POC < 5 ticks, POC crosses > 25/30min, CVD slope < 0.1

=== PERFORMANCE FEEDBACK (Self-Learning) ===
Recent Win Rate: ${performance.win_rate * 100}%
Average P&L: $${performance.avg_pnl}
Trade Count: ${performance.trade_count}
Profit Factor: ${performance.profit_factor}

=== HISTORICAL NOTES (Lessons Learned) ===
${historicalNotes.map(n => `- ${n.context}: ${n.note}`).join('\n')}
```

### Step 5: Add Self-Learning Response Fields
**File:** `lib/openaiTradingAgent.ts` - `OpenAITradingDecision` interface

```typescript
export interface OpenAITradingDecision {
  // ... existing fields ...

  // NEW: Self-learning fields
  inferredRegime?: 'trend' | 'range' | 'chop';
  regimeConfidence?: number;
  noteForFuture?: string; // Note to add to historical notes
}
```

### Step 6: Update Response Parsing
**File:** `lib/openaiTradingAgent.ts` - `parseOpenAIResponse()`

Parse the new fields from OpenAI's response.

### Step 7: Record Performance After Trades
**File:** `lib/fabioOpenAIIntegration.ts` - `updatePositionAndCheckExits()`

```typescript
if (closedDecisionId) {
  const outcome = tradingDB.getOutcome(closedDecisionId);
  performanceTracker.recordTrade(outcome.profitLoss);

  // Add note if needed
  if (outcome.profitLoss < -200) {
    notesManager.addNote(
      `Large loss on ${setupModel} setup - review entry conditions`,
      marketState
    );
  }
}
```

## Benefits of Enhanced Integration

### 1. Regime-Aware Trading
LLM infers market regime from raw stats instead of being told:
- Adapts to changing conditions automatically
- More nuanced regime detection
- Better decision quality

### 2. Self-Learning
System learns from outcomes:
- Tracks what works vs what doesn't
- Builds historical context
- Improves over time

### 3. POC Cross Intelligence
Critical indicator for chop vs trend:
- High crosses = choppy, avoid trading
- Low crosses = trending, trade the trend
- Time-windowed analysis (5min, 15min, 30min)

### 4. Session Context
Historical percentiles provide context:
- Is this a big range day or small?
- Should we expect more movement?
- Adjust targets accordingly

## Testing Plan

1. **Unit Test Enhanced Features**
   ```bash
   npx tsx test-enhanced-features.ts
   ```

2. **Test with Real Data**
   ```bash
   npx tsx test-real-nq-data-enhanced.ts
   ```

3. **Validate Regime Inference**
   - Compare LLM-inferred regime vs actual market behavior
   - Verify POC cross counts are accurate

4. **Performance Tracking**
   - Run 20-30 trades
   - Verify win rate and profit factor calculations
   - Check historical notes are being stored

## Implementation Order

1. ✅ Create `lib/enhancedFeatures.ts` (DONE)
2. ⏳ Update `FuturesMarketData` interface
3. ⏳ Update system prompt with regime rules
4. ⏳ Initialize trackers in `fabioOpenAIIntegration.ts`
5. ⏳ Update `buildFuturesMarketData()` to use trackers
6. ⏳ Update `buildAnalysisPrompt()` with new sections
7. ⏳ Add self-learning response fields
8. ⏳ Update parsing logic
9. ⏳ Add performance tracking after trades
10. ⏳ Test with real data

## Questions for User

1. Should I proceed with full implementation now?
2. Do you want to test each piece incrementally or implement all at once?
3. Any specific regime inference rules you want to adjust?

---

**Status:** Step 1 Complete - Enhanced features module created
**Next:** Update FuturesMarketData interface and system prompt
