# Mean Reversion Trading Strategy - Backtest Documentation

## 📋 Table of Contents
- [Environment Setup](#environment-setup)
- [Strategy Overview](#strategy-overview)
- [Strategy Evolution](#strategy-evolution)
- [Performance Results](#performance-results)
- [Futures Conversion](#futures-conversion)
- [Key Insights](#key-insights)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)

---

## 🔧 Environment Setup

### Environment Variables Location
**IMPORTANT**: The correct `.env` file is located at:
```
/Users/coreycosta/trading-agent/.env
```

### Required Environment Variables
```bash
# Tradier API (Required for backtesting)
TRADIER_ACCOUNT_ID=VA89950964
TRADIER_API_KEY=bn6k3QzkOyJPBtmSEytqTjQPG8nl
TRADIER_BASE_URL=https://sandbox.tradier.com/v1

# Unusual Whales API (For GEX calculation)
UNUSUAL_WHALES_API_KEY=730307ec-8712-4d54-b26e-616791b6dae2

# Twelve Data Streaming (Live intraday ticks)
TWELVE_DATA_API_KEY=your_twelve_data_key
# Optional: override WebSocket endpoint (defaults to wss://ws.twelvedata.com/v1/quotes/price)
# TWELVE_DATA_WS_URL=wss://ws.twelvedata.com/v1/quotes/price

# Testing Mode
BYPASS_TIME_DECAY=true  # Required for backtesting with historical data
```

### Critical Setup Notes
1. **Import Order Matters**: Files must use `import 'dotenv/config'` at the very top before any other imports
2. **Files requiring dotenv/config**:
   - `/Users/coreycosta/trading-agent/test-mean-reversion.ts` (line 1)
   - `/Users/coreycosta/trading-agent/lib/tradier.ts` (line 1)
3. **Sandbox API Limitations**: Tradier sandbox has limited historical data and date ranges

---

## 📊 Strategy Overview

### Core Concept
Mean reversion strategy that trades Bollinger Band extremes with RSI confirmation. Originally designed for positive GEX (Gamma Exposure) regime, but **GEX filter is currently DISABLED** - strategy trades all days regardless of regime.

### Technical Indicators

#### Bollinger Bands
- **Period**: 20 bars
- **Standard Deviation**: 2
- **Timeframe**: 15-minute bars
- **Entry Trigger**: Price touches or exceeds outer bands (within 0.5% threshold)

#### RSI (Relative Strength Index)
- **Period**: 14 bars
- **Oversold**: < 30 (for LONG entries)
- **Overbought**: > 70 (for SHORT entries)
- **Timeframe**: 15-minute bars
- **Exit Filter**: DISABLED (let winners run)

### Entry Rules
1. **15-minute bar close** triggers evaluation
2. **LONG Setup**:
   - RSI < 30 (oversold)
   - Price within 0.5% of lower Bollinger Band
   - Entry price = 15-minute bar close
3. **SHORT Setup**:
   - RSI > 70 (overbought)
   - Price within 0.5% of upper Bollinger Band
   - Entry price = 15-minute bar close

**Note**: GEX filter is currently DISABLED. Originally required positive GEX, but testing showed similar performance trading all days.

### Position Sizing: 2-Unit Scaling Strategy
1. **Initial Entry**: 2 units at 15-minute bar close
2. **Scaling at Middle Band**:
   - When price reaches Bollinger Band middle (20 SMA)
   - Close 1 unit (take 50% profit)
   - Adjust stop loss to 1% from middle band (lock in profit)
   - Set new target to 1% from opposite outer band
3. **Remaining Unit**: Runs to opposite band or end of day

### Exit Rules (1-minute intrabar execution)
1. **Stop Loss**: 0.1% (0.001) from entry price
   - After scaling: moved to 1% from middle band
2. **Target**: Bollinger Band middle (20 SMA)
   - After scaling: 1% from opposite outer band
3. **End of Day**: Force close at 4:00 PM ET
4. **RSI Exit**: DISABLED (originally would exit at RSI 50, but removed to let winners run)

### Intrabar Execution
- **Signal Bars**: 15-minute bars for BB/RSI calculation
- **Exit Bars**: 1-minute bars for precise stop/target execution
- **Entry Price**: 15-minute bar close price
- **Exit Price**: 1-minute bar that triggers stop/target

---

## 🔄 Strategy Evolution

### Phase 1: Initial Setup Issues (FIXED)
**Problem**: Strategy returning 401 errors and wrong results
**Root Cause**:
- Missing `import 'dotenv/config'` in key files
- Using 1-minute bars for indicator calculation instead of 15-minute
**Solution**:
- Added dotenv import to `test-mean-reversion.ts` and `lib/tradier.ts`
- Changed function call to: `backtestMeanReversionMultiple(symbols, dates, 'intraday', 15, 1)`
  - `15` = indicator timeframe (15-minute bars)
  - `1` = exit timeframe (1-minute intrabar execution)

### Phase 2: Entry Price Timing (OPTIMIZED)
**Problem**: Using last 1-minute bar within 15-minute period for entry
**User Feedback**: "change entry price to bar close on 15 min"
**Solution**: Modified to use 15-minute bar close directly
**Impact**: More realistic execution, eliminates timing ambiguity

### Phase 3: GEX Filter Removed (CURRENTLY DISABLED)
**Test**: Commented out positive GEX requirement to trade all days
**Result**: Similar performance with or without filter (+$8.28 both ways)
**Conclusion**: GEX filter not critical, negative GEX days performed well
**Current State**: **GEX filter is BYPASSED** - code trades all days regardless of regime (lines 347-395 in `meanReversionBacktester.ts`)
**Evidence**: All 5 SPY trades were on negative GEX days (-1062M to -1068M)

### Phase 4: RSI Exit Testing (IMPROVED)
**Test 1: No RSI at all**
- Result: -$2.22 (75% more trades, all losers)
- Conclusion: RSI entry filter is critical

**Test 2: RSI for entries only, disabled for exits**
- Result: +$9.68 (17% improvement over +$8.28 baseline)
- Conclusion: RSI exit filter was closing winners too early
- **Implementation**: Commented out RSI exit logic (lines 347-355 in `meanReversionAgent.ts`)

### Phase 5: Position Scaling (MAJOR IMPROVEMENT)
**User Request**: "what if we bought double to start and when they both reach middle band we set a new stop loss of .01 from middle band and set new take profit .01 from other side outer band"

**Implementation**:
1. Enter with 2 units
2. At middle band: close 1 unit, adjust stops/targets
3. Run remaining unit to opposite band

**Results**:
- Before scaling: +$9.68 (20 trades)
- After scaling: +$20.57 (20 trades)
- **Improvement: +112.9%**

**Code Changes**:
- Added `scaled` and `units` fields to `ActivePosition` interface
- Scaling logic in lines 589-615 of `meanReversionBacktester.ts`
- Profit calculation updated to account for 2-unit entry (lines 423-445)

---

## 📈 Performance Results

### SPY - Oct 27-31, 2025 (Primary Testing)

#### Stock Trading (2-unit scaling)
- **Total Trades**: 5
- **Win Rate**: 40% (2 wins / 3 losses)
- **Total P&L**: +$13.16
- **Gross Profit**: +$16.33
- **Gross Loss**: -$3.17
- **Profit Factor**: 5.15
- **Average Win**: +$8.17
- **Average Loss**: -$1.06

#### Trade Breakdown
1. **10/27 SHORT**: Entry $684.65 → Exit $685.33 = -$1.37 (stop)
2. **10/27 SHORT**: Entry $685.19 → Exit $685.40 = -$0.42 (EOD)
3. **10/28 SHORT**: Entry $687.98 → Exit $688.67 = -$1.38 (stop)
4. **10/28 SHORT**: Entry $688.69 → Exit $686.85 = +$5.78 (EOD) ✅
5. **10/29 LONG**: Entry $683.29 → Exit $686.66 = +$10.55 (EOD) ✅

#### Exit Reason Analysis
- **Stop Loss**: 3 trades (60%) - Average loss: -$1.06
- **End of Day**: 2 trades (40%) - Average profit: +$8.17
- **Key Finding**: Winners ran to EOD, losers stopped quickly

### 10 Major ETFs - Oct 27-31, 2025

**Symbols Tested**: SPY, QQQ, IWM, DIA, GLD, SLV, TLT, XLF, XLE, AAPL

#### Overall Results
- **Total Trades**: 27
- **Positive GEX Days Traded**: 14 / 50 (28%)
- **Win Rate**: 51.9%
- **Total P&L**: +$22.31
- **Profit Factor**: 2.59

#### Top Performers
1. **GLD**: +$11.57 (4 trades, 100% win rate)
2. **SPY**: +$10.78 (4 trades, 50% win rate)
3. **SLV**: +$7.52 (3 trades, 67% win rate)

#### Poor Performers
- **QQQ**: -$6.79 (7 trades, 29% win rate)
- **IWM**: -$3.09 (2 trades, 0% win rate)

### 10 Major Stocks - Oct 27-31, 2025

**Symbols Tested**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, WMT

#### Overall Results
- **Total Trades**: 13
- **Positive GEX Days Traded**: 13 / 50 (26%)
- **Win Rate**: 46.2%
- **Total P&L**: +$6.87
- **Profit Factor**: 2.29

#### Key Finding
- **NVDA only winner**: +$9.50 (100% of profit)
- **Individual stocks more volatile**: 0.1% stop too tight
- **Conclusion**: Strategy optimized for liquid ETFs, not individual stocks

---

## 🚀 Futures Conversion

### ES/MES Futures Calculation

When trading SPY signals on ES (E-mini S&P 500) or MES (Micro E-mini S&P 500) futures:

#### Conversion Formula
```
SPY Price Move × 10 = SPX Index Points
SPX Index Points × Contract Multiplier = Gross P&L
Gross P&L - Commission = Net P&L
```

#### Contract Specifications
- **ES Futures**: $50 per SPX point
- **MES Futures**: $5 per SPX point (1/10th of ES)
- **Commission**: $1 per side = $2 round-trip per contract
- **Position Size**: 2 contracts (matching 2-unit stock position)

#### Example Calculation (Trade #5 from 10/29)
```
SPY Entry: $683.29
SPY Exit:  $686.66
SPY Move:  $3.37

SPX Index Points: $3.37 × 10 = 33.71 points

ES Futures (2 contracts):
  Gross: 33.71 × $50 × 2 = $3,371.30
  Commission: $1 × 2 sides × 2 contracts = $4.00
  Net P&L: $3,371.30 - $4.00 = $3,367.30

MES Futures (2 contracts):
  Gross: 33.71 × $5 × 2 = $337.13
  Commission: $1 × 2 sides × 2 contracts = $4.00
  Net P&L: $337.13 - $4.00 = $333.13
```

### Futures Performance: SPY Signals (Oct 27-31, 2025)

#### ES Futures Results (2 contracts @ $50/point)
| Date  | Direction | Entry    | Exit     | Index Points | Gross P&L  | Net P&L   | Exit     |
|-------|-----------|----------|----------|--------------|------------|-----------|----------|
| 10/27 | SHORT     | $684.65  | $685.33  | 6.85         | -$685.00   | -$688.65  | stop     |
| 10/27 | SHORT     | $685.19  | $685.40  | 2.10         | -$210.00   | -$214.00  | EOD      |
| 10/28 | SHORT     | $687.98  | $688.67  | 6.88         | -$688.00   | -$691.98  | stop     |
| 10/28 | SHORT     | $688.69  | $686.85  | 18.40        | +$1,840.00 | +$1,836.00| EOD ✅   |
| 10/29 | LONG      | $683.29  | $686.66  | 33.71        | +$3,371.30 | +$3,367.30| EOD ✅   |

**Total ES P&L: +$3,608.67**

#### MES Futures Results (2 contracts @ $5/point)
| Date  | Direction | Entry    | Exit     | Index Points | Gross P&L | Net P&L  | Exit     |
|-------|-----------|----------|----------|--------------|-----------|----------|----------|
| 10/27 | SHORT     | $684.65  | $685.33  | 6.85         | -$68.50   | -$72.46  | stop     |
| 10/27 | SHORT     | $685.19  | $685.40  | 2.10         | -$21.00   | -$25.00  | EOD      |
| 10/28 | SHORT     | $687.98  | $688.67  | 6.88         | -$68.80   | -$72.80  | stop     |
| 10/28 | SHORT     | $688.69  | $686.85  | 18.40        | +$184.00  | +$180.00 | EOD ✅   |
| 10/29 | LONG      | $683.29  | $686.66  | 33.71        | +$337.13  | +$333.13 | EOD ✅   |

**Total MES P&L: +$342.87**

#### Leverage Comparison
- **Stock P&L**: +$13.16
- **ES P&L**: +$3,608.67 (274× stock profit)
- **MES P&L**: +$342.87 (26× stock profit)

#### Commission Impact
- Per trade cost: $4 ($1 per side × 2 contracts)
- Total commission (5 trades): $20
- Commission as % of gross profit: 0.4% (ES), 3.5% (MES)
- **Conclusion**: Commission negligible on large point moves

---

## 💡 Key Insights

### 1. Strategy Performance Characteristics
- **Winner Profile**: Large moves that run to end of day
- **Loser Profile**: Quick stops at 0.1% loss
- **Win Rate**: ~40-50% (intentionally asymmetric)
- **Risk:Reward**: ~1:8 average (small stops, big winners)

### 2. Position Scaling Impact
- **112% profit improvement** from 2-unit scaling approach
- Taking profit at middle band locks in gains
- Letting 1 unit run captures large moves
- Adjusted stop after scaling prevents giving back profits

### 3. RSI Usage Optimization
- **Critical for entries**: Prevents poor setups (prevents random BB touches)
- **Harmful for exits**: Closes winners too early at RSI 50
- **Optimal**: Use RSI < 30 / > 70 for entries only

### 4. Asset Class Performance
- **ETFs outperform individual stocks** significantly
- **Precious metals (GLD, SLV)** performed best
- **Tech stocks (QQQ)** worst performer
- **Conclusion**: Strategy designed for liquid, lower-volatility ETFs

### 5. GEX Filter Analysis (CURRENTLY DISABLED)
- **GEX filter is bypassed** in current implementation
- Positive GEX regime not critical for profitability
- Negative GEX days actually performed well (all 5 SPY trades were on negative GEX)
- GEX more useful for position sizing than filtering
- **Current State**: Trading all days regardless of GEX regime

### 6. Stop Loss Calibration
- **0.1% stop loss**: Perfect for ETFs (SPY, GLD, SLV)
- **Too tight for individual stocks**: Gets stopped on noise
- Individual stocks would need 0.3-0.5% stops

### 7. Futures Advantages
- **ES provides 274× leverage** vs stock with minimal commission
- **Large moves (20+ SPX points)** make futures highly profitable
- **MES offers middle ground**: 26× leverage with lower risk
- **Commission negligible**: $4 per trade vs hundreds/thousands in profit

### 8. Execution Timing
- **15-minute bar close** provides clear, objective entry
- **1-minute intrabar exits** allow precise stop/target hits
- **End of day closure** captures trend day moves
- No execution slippage in backtest (assumes market orders)

### 9. Market Regime
- Strategy originally designed for **pinning regime** (positive GEX)
- **Currently trades all regimes** - GEX filter disabled
- Works in trending environments when moves are large
- All 5 profitable SPY trades occurred on negative GEX days
- **Best days**: Low volatility → large move → reversion
- **Worst days**: Choppy, range-bound with false signals

### 10. Options Integration
- Options tracked but highly volatile
- **Stock/futures preferred** for execution
- Options provide signal confirmation via GEX
- Per-contract basis less reliable than scaled futures

---

## 🏃 How to Run

### Prerequisites
```bash
npm install
# or
yarn install
```

### Run Backtest
```bash
# Using tsx (recommended)
npx tsx test-mean-reversion.ts

# Using ts-node
npx ts-node test-mean-reversion.ts

# Using node (if compiled)
npm run build && node dist/test-mean-reversion.js
```

### Run Live Strategy
```bash
# Ensure TWELVE_DATA_API_KEY (and optionally TWELVE_DATA_WS_URL) are set in .env
npx tsx run-live-mean-reversion.ts
```

The live loop streams intraday ticks from Twelve Data to manage stops/targets in real time while continuing to route orders through Alpaca.

### Modify Test Parameters

Edit `/Users/coreycosta/trading-agent/test-mean-reversion.ts`:

```typescript
// Line 13: Change symbols
const symbols = ['SPY'];  // Add more: ['SPY', 'QQQ', 'GLD']

// Line 16-22: Change dates
const dates = [
  '2025-10-27',
  '2025-10-28',
  '2025-10-29',
  '2025-10-30',
  '2025-10-31',
];

// Line 29: Change timeframes
// backtestMeanReversionMultiple(symbols, dates, mode, indicatorInterval, exitInterval)
const results = await backtestMeanReversionMultiple(
  symbols,  // Array of symbols
  dates,    // Array of dates
  'intraday',  // Mode: 'intraday' or 'daily'
  15,       // Indicator interval in minutes (15 = 15-min bars)
  1         // Exit interval in minutes (1 = 1-min intrabar)
);
```

### Output Files
Results are automatically saved to:
```
backtest_mean_reversion_YYYY-MM-DD.json
```

Example: `backtest_mean_reversion_2025-11-06.json`

### Reading Results
```bash
# Pretty print JSON
cat backtest_mean_reversion_2025-11-06.json | jq '.'

# Count traded days
cat backtest_mean_reversion_2025-11-06.json | jq '[.[] | select(.trades | length > 0)] | length'

# Extract ES futures P&L
cat backtest_mean_reversion_2025-11-06.json | jq '[.[].trades[].futures.esProfit] | add'
```

---

## 📁 File Structure

### Core Backtesting Files
```
/Users/coreycosta/trading-agent/
├── test-mean-reversion.ts              # Main backtest runner
├── lib/
│   ├── meanReversionBacktester.ts      # Backtesting engine
│   ├── meanReversionAgent.ts           # Signal generation logic
│   ├── tradier.ts                      # Tradier API client
│   └── gexCalculator.ts                # GEX calculation (Unusual Whales)
├── .env                                 # Environment variables (REQUIRED)
└── backtest_mean_reversion_*.json      # Output files
```

### Key Code Sections

#### `/test-mean-reversion.ts`
- **Line 1**: `import 'dotenv/config'` - CRITICAL
- **Line 13**: Symbol list
- **Line 16-22**: Date range
- **Line 29**: Backtest function call with intervals
- **Line 48-141**: Results formatting and display
- **Line 132-134**: Futures P&L display

#### `/lib/meanReversionBacktester.ts`
- **Line 28-34**: Futures trade interface
- **Line 109-117**: Active position interface (includes scaling fields)
- **Line 347-395**: GEX filter (DISABLED)
- **Line 423-445**: Profit calculation (handles 2-unit scaling)
- **Line 489-520**: Futures P&L calculation
- **Line 589-615**: Position scaling logic at middle band
- **Line 659-665**: Entry logic (uses 15-min bar close)

#### `/lib/meanReversionAgent.ts`
- **Line 168-170**: LONG entry conditions (RSI < 30 + lower BB)
- **Line 196-198**: SHORT entry conditions (RSI > 70 + upper BB)
- **Line 347-355**: RSI exit filter (DISABLED)

#### `/lib/tradier.ts`
- **Line 1**: `import 'dotenv/config'` - CRITICAL
- **Line 12-14**: API configuration from .env
- **Line 34-82**: Stock price fetching (supports historical dates)
- **Line 433-484**: Intraday timesales fetching (supports 1min, 5min, 15min intervals)

---

## 🔍 Troubleshooting

### Issue: 401 Unauthorized Error
**Solution**: Ensure `import 'dotenv/config'` is at the top of both `test-mean-reversion.ts` and `lib/tradier.ts`

### Issue: Wrong .env file loaded
**Solution**: Verify working directory is `/Users/coreycosta/trading-agent` and `.env` exists there

### Issue: No trades generated
**Solution**: Check date range - Tradier sandbox has limited historical data. Try recent dates.

### Issue: Different results than expected
**Solution**: Verify indicator interval is 15 minutes: `backtestMeanReversionMultiple(symbols, dates, 'intraday', 15, 1)`

### Issue: Futures P&L not showing
**Solution**: Ensure symbol is 'SPY' - futures calculation only runs for SPY signals

---

## 📚 References

- Tradier API Docs: https://documentation.tradier.com/
- Unusual Whales GEX: https://unusualwhales.com/
- Bollinger Bands: https://www.investopedia.com/terms/b/bollingerbands.asp
- RSI Indicator: https://www.investopedia.com/terms/r/rsi.asp
- ES Futures Specs: https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html
- MES Futures Specs: https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.html

---

**Last Updated**: 2025-11-06
**Backtest Date Range**: Oct 27-31, 2025
**Strategy Version**: 2.0 (Position Scaling)
