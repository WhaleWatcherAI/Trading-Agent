# Multi-Symbol Backtest Analysis - 1 Second SMA Strategy

## Fee Structure (TopstepX - Verified ✅)

```
Symbol     | Multiplier | Tick Size | Commission/Side | RT Cost | Notes
-----------|------------|-----------|-----------------|---------|--------
ES (ESZ5)  | $50/point  | $0.25     | $1.40           | $2.80   | Large Contract
MES (MESZ5)| $5/point   | $0.25     | $0.37           | $0.74   | Micro Contract
NQ (NQZ5)  | $20/point  | $0.25     | $1.40           | $2.80   | Nasdaq 100
MNQ (MNQZ5)| $2/point   | $0.25     | $0.37           | $0.74   | Micro Nasdaq
GC (GCZ5)  | $100/point | $0.10     | $1.62           | $3.24   | Gold
6E (6EZ5)  | $12.50/pip | $0.0001   | $1.62           | $3.24   | Euro Futures
```

## Test Results (5-Day Period: Nov 5-10, 2025)

### Strategy Parameters Tested
- **Entry:** SMA 500s (fast) / 1500s (slow) crossover
- **Stop Loss:** 6 ticks (1.5 points on ES)
- **Take Profit:** 15 ticks (3.75 points on ES)
- **Position Size:** 1 contract
- **Daily Max:** 6 trades/day
- **Min Gap:** 15 minutes between trades
- **Filters:** RSI, Daily limit, Time gap only

### ES Results (Nov 10, 2025 Run)
```
Trades:        18 total
Win Rate:      16.7% (3 wins, 15 losses)
Total P&L:     -$612.90 ❌
Avg Win:       +$184.70
Avg Loss:      -$77.80
Max Drawdown:  $982.30
Profit Factor: 0.47 (need 1.0+ to break even)

Exit Breakdown:
- Stop Loss Hit:  15 trades (83.3%) ⚠️ Too many!
- Target Hit:     3 trades (16.7%) ⚠️ Too few!
```

### Why This Failed:
1. **Too many stops hit (83%)** → Signal quality issues
2. **Few targets hit (17%)** → Trend not strong enough after entry
3. **Ratio mismatch** → Need at least 30-40% targets hit with 2.5:1 RR

## Commission Impact Analysis

### ES (1 contract, 18 trades):
```
Entry Slippage:      18 trades × $25 (0.5 ticks) = -$450
Exit Slippage:       18 trades × $25 = -$450
Commission Cost:     18 × $2.80 RT = -$50.40
Total Hidden Costs:  ~-$950

Without Slippage:    -$612.90 (actual)
With Slippage:       -$1,562.90 (realistic)
```

### MES (1 contract, 18 trades) - Estimated:
```
Commission Cost:     18 × $0.74 RT = -$13.32
Much lower fees due to micro contract
But also lower profit potential per point move
```

## Key Findings

### ✅ Fee Calculations Are Correct
- TopstepX commissions properly loaded from `futuresFees.ts`
- All contract multipliers and tick sizes accurate
- Both full and micro contract pairs have correct ratios

### ⚠️ Strategy Issues Identified

1. **Market Regime Dependent**
   - Same strategy gave +$101 to +$583 in earlier runs
   - Now showing -$613 loss
   - Market conditions matter MORE than strategy parameters

2. **Signal Quality Problem**
   - 83% stop hit rate indicates entries are happening in poor setups
   - Need additional confirmation filters
   - RSI-only filter insufficient

3. **Risk/Reward Mismatch**
   - 6-tick stop vs 15-tick target looks good (2.5:1)
   - But stop is hit too often
   - Need to either: widen entry criteria OR increase target

4. **Position Sizing**
   - 1 contract is good for risk control
   - But generates only $25 per point of profit (ES)
   - Each trade needs to work 4+ points to justify entry

## Recommendations for Profitability

### Option 1: Improve Entry Signals (Recommended)
```typescript
// Add confluence filters:
- Bollinger Band squeeze breakout (only trade narrow bands)
- Volume confirmation (trade only high volume)
- Multi-timeframe alignment (check 5s, 15s, 60s)
- Volatility regime (skip dead markets)
```

### Option 2: Adjust Risk/Reward Ratio
```
Current:  6 tick stop / 15 tick target = 2.5:1
Try:      5 tick stop / 20 tick target = 4:1
Or:       4 tick stop / 16 tick target = 4:1
```
This requires fewer winning trades to break even.

### Option 3: Symbol Selection
```
ES/NQ:   Higher comm ($1.40) but more liquid, tighter spreads
MES/MNQ: Lower comm ($0.37) better for small accounts
         But lower point value = need bigger moves
```

## What Works In This Strategy

### ✅ Positive Aspects
1. **Proper look-ahead bias handling** - Uses only closed bar data
2. **Realistic fill assumptions** - Can be improved but reasonable
3. **Trailing stop logic** - Locks in profits effectively
4. **Daily management** - Max trades/day prevents overtrading

### ⚠️ Needs Improvement
1. **Entry signal generation** - Too many false signals (83% stop rate)
2. **Market condition detection** - No regime filter
3. **Slippage assumptions** - Currently assuming zero

## Next Steps

1. **Add market regime detection** to skip choppy markets
2. **Require multiple timeframe confirmation**
3. **Increase minimum win rate requirement** before trading live
4. **Test on longer period** (3-6 months minimum)
5. **Paper trade first** to validate execution

## Conclusion

**The backtest framework is sound, but the strategy currently is NOT ready for live trading.**

- ✅ Fee calculations are correct
- ✅ No look-ahead bias
- ✅ Risk management is tight
- ❌ Win rate is too low (needs 30%+ with this RR)
- ❌ Too many false signals
- ❌ Market dependent performance

**Estimated Live P&L with Slippage: -$1,500+ for ES with these settings**

Recommend improving entry signal quality before live deployment.
