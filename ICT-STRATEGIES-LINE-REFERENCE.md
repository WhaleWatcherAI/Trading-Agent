# ICT Strategies - Specific Line Number References

## STRATEGY #1: LIQUIDITY-SWEEP + FVG RETURN
**File:** `backtest-ict-liquidity-sweep-fvg.ts`

| Topic | Lines | Details |
|-------|-------|---------|
| **Entry Rules** | 1-35 | Liquidity sweep, FVG detection (3-bar), 50% midpoint entry |
| **Config Interface** | 47-63 | BacktestConfig with all parameters |
| **Defaults** | 130-150 | Environment variables and defaults (reclaimBars=5, SL=2, TP1=1, TP2=2) |
| **Slippage Model** | 100-120 | fillEntry, fillTP, fillStop functions with σ (sigma) calculations |
| **Session Filter** | 160+ | NY session detection (09:30-11:30, 13:30-15:30 ET) |
| **ATR Calculation** | 178+ | 14-period ATR for filter validation |
| **FVG Detection** | 206-257 | detectFVG function (bullish: low>high[-2], bearish: high<low[-2]) |
| **Position Management** | 375-416 | exitPosition function with fees and PnL calculation |

---

## STRATEGY #2: BOS/CHOCH + FVG
**File:** `backtest-ict-bos-choch-fvg.ts`

| Topic | Lines | Details |
|-------|-------|---------|
| **Core Concept** | 1-32 | Break of Structure using pivot detection, 50% FVG midpoint entry |
| **Config Interface** | 43-64 | BacktestConfig with pivotLength, minBOSDisplacement, minBOSBodyPercent |
| **Defaults** | 130-150 | pivotLength=6 (was 3), SL=2, TP1=1, TP2=2 |
| **Pivot Detection** | ~200+ | detectPivots function - swing high/low detection with lookback |
| **BOS Logic** | ~300+ | Detect break of swing high (bullish) or swing low (bearish) |
| **Filters** | ~300-350 | minBOSDisplacement=1.0×ATR, minBOSBodyPercent=0.6 |
| **Cooldown** | ~150 | cooldownBars=20, maxLongsPerSession=1, maxShortsPerSession=1 |

---

## STRATEGY #3: POWER OF THREE (PO3) LITE
**File:** `backtest-ict-po3-lite.ts`

| Topic | Lines | Details |
|-------|-------|---------|
| **Core Concept** | 1-31 | 3-phase model: Accumulation (Asia) → Manipulation (London) → Distribution (NY) |
| **Config Interface** | 42-58 | BacktestConfig with all 7 filter parameters |
| **Defaults** | 130-147 | minAsiaRangeATR=0.8, minFVGSizeTicks=6, minFVGSizeATR=0.35, minSweepTicks=5, minBarsAfterSweep=30 |
| **Session Times** | 150-175 | getSessionType function: Asia (20:00-00:00), London (02:00-05:00), NY (09:30-11:30, 13:30-15:30 ET) |
| **ATR Calculation** | 178-196 | 14-period ATR for filter validation |
| **FVG Detection** | 206-257 | detectFVG with size filters (minFVGSizeTicks + minFVGSizeATR) |
| **Phase 1: Asia Range** | 522-530 | Track high/low during Asia session |
| **Phase 2: London Sweep** | 532-573 | Detect bullish (sweeps low) or bearish (sweeps high) manipulation |
| **Phase 3: NY Entry** | 575-687 | Entry on FVG + validation (entry inside range) + one trade per day |
| **TP1 Logic** | 443-470 | Scale at Asia midpoint, move stop to breakeven |
| **TP2 Logic** | 472-491 | Exit at 75% of range from midpoint (tp2RangePercent=0.75) |
| **Stop Loss** | 605/651 | LONG: `fvgLower - buffer`, SHORT: `fvgUpper + buffer` |

---

## LIVE IMPLEMENTATION
**File:** `live-topstepx-mgc-po3.ts`

| Topic | Lines | Details |
|-------|-------|---------|
| **Strategy Config** | 108-123 | StrategyConfig interface with all 8 parameters |
| **Defaults** | 268-276 | Same as backtest: minAsiaRangeATR=0.8, minFVGSizeTicks=6, etc. |
| **Session Times** | Similar to backtest | getSessionType function with ET timezone conversion |

---

## DOCUMENTATION FILES

### ICT-STRATEGIES-README.md (321 lines)
- **Strategy #1 Overview:** Lines 17-78 (entry, risk/targets, config, examples)
- **Strategy #2 Overview:** Lines 82-143 (BOS/CHOCH logic, config, examples)
- **Strategy #3 Overview:** Lines 147-207 (PO3 phases, config, examples)
- **FVG Detection:** Lines 212-217 (bullish/bearish patterns, 50% midpoint)
- **Common Features:** Lines 218-234 (slippage, commission model)

### STRATEGY-3-IMPROVEMENTS.md (246 lines)
- **Before/After MNQZ5:** Lines 7-31 (43.8% → 75.0% WR, 1.56 → 19.09 PF)
- **Before/After MESZ5:** Lines 34-58 (37.5% → 63.6% WR, 2.29 → 17.40 PF)
- **Before/After MGCZ5:** Lines 61-87 (61.1% → 87.5% WR, 2.88 → 40.99 PF) - STAR PERFORMER
- **Before/After M6EZ5:** Lines 90-117 (27.3% → 100% WR, but only 1 trade - filters too tight)
- **7 Key Improvements:** Lines 122-153 (Asia filter, sweep size, FVG size, wait time, entry validation, 1 trade/day, less aggressive TP2)
- **Profitable Stops Phenomenon:** Lines 155-172 (explains why stops are profitable)
- **Configuration Reference:** Lines 221-241 (optimal settings for equity indices vs FX)

### ict-backtest-results.txt (156 lines)
- **Strategy #1 Results:** Lines 8-53 (MNQZ5: 77.8%, ES: 66.7%, GC: 100%, 6E: 62.5%)
- **Strategy #2 Results:** Lines 57-102 (All symbols losing or marginal)
- **Strategy #3 Results:** Lines 105-151 (MNQZ5: 43.8%, MES: 37.5%, MGC: 61.1%, M6E: 27.3%)

---

## KEY CODE PATTERNS

### Fair Value Gap Detection
```typescript
// Lines 206-257 in backtest-ict-po3-lite.ts
// Bullish FVG: low[t] > high[t-2]
if (current.low > twoAgo.high) {
  const fvgSize = current.low - twoAgo.high;
  const minSize = Math.max(
    config.minFVGSizeTicks * tickSize,
    config.minFVGSizeATR * atr
  );
  if (fvgSize < minSize) return null;
  return {
    type: 'bullish',
    midpoint: (current.low + twoAgo.high) / 2,
    upper: current.low,
    lower: twoAgo.high,
    barIndex: currentIndex,
  };
}
```

### Stop Loss Positioning (PO3 LONG)
```typescript
// Lines 605 in backtest-ict-po3-lite.ts
const stopLoss = roundToTick(fvg.lower - CONFIG.stopLossBuffer * tickSize);
```

### Take Profit Levels (PO3)
```typescript
// Lines 606-611 in backtest-ict-po3-lite.ts
const asiaMid = (currentDay.asiaHigh + currentDay.asiaLow) / 2;
const tp1 = roundToTick(asiaMid); // Scale out here

// TP2: 75% of Asia range from midpoint
const rangeFromMid = (currentDay.asiaHigh - asiaMid) * CONFIG.tp2RangePercent;
const tp2 = roundToTick(asiaMid + rangeFromMid);
```

### Session Detection
```typescript
// Lines 150-175 in backtest-ict-po3-lite.ts
function getSessionType(timestamp: string): 'asia' | 'london' | 'ny' | 'other'
// Asia:   20:00-00:00 ET = 1200-1440 minutes
// London: 02:00-05:00 ET = 120-300 minutes
// NY:     09:30-11:30 ET = 570-690 minutes
//         13:30-15:30 ET = 810-930 minutes
```

---

## OPTIMAL PARAMETER SETTINGS

### Equity Indices (MNQZ5, MESZ5, MGCZ5) - PRODUCTION READY
```bash
# PO3 Strategy (Best Performance)
ICT_PO3_MIN_ASIA_ATR=0.8        # Skip choppy days
ICT_PO3_MIN_FVG_TICKS=6         # Min 6 ticks
ICT_PO3_MIN_FVG_ATR=0.35        # Min 35% of ATR
ICT_PO3_MIN_SWEEP_TICKS=5       # Min 5 ticks
ICT_PO3_MIN_BARS_AFTER_SWEEP=30 # Wait 30 minutes
ICT_PO3_TP2_RANGE_PCT=0.75      # 75% of range
ICT_PO3_SL_BUFFER=2             # 2 ticks
```

### FX Markets (M6EZ5) - NEEDS RELAXED FILTERS
```bash
# Suggested for currency pairs (lower volatility)
ICT_PO3_MIN_ASIA_ATR=0.5        # Lower from 0.8
ICT_PO3_MIN_FVG_TICKS=4         # Lower from 6
ICT_PO3_MIN_FVG_ATR=0.25        # Lower from 0.35
ICT_PO3_MIN_SWEEP_TICKS=3       # Lower from 5
ICT_PO3_MIN_BARS_AFTER_SWEEP=20 # Shorter from 30
```

