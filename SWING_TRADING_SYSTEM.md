# FABIO Swing Trading System - MES

## Overview

A **multi-day swing trading system** for MES (Micro E-mini S&P 500) futures that focuses on capturing larger moves over days/weeks rather than intraday scalping.

### Key Differences from Day Trading Version

| Aspect | Day Trading (NQ/MGC) | Swing Trading (MES) |
|--------|---------------------|---------------------|
| **Timeframes** | 1-min primary, 5-min secondary | Daily primary, 4H/1H secondary, 5-min for entry only |
| **Analysis Frequency** | Real-time (every bar) | Every 5 minutes |
| **Hold Time** | Minutes to hours (close EOD) | Days to weeks (3-7 day typical) |
| **Stop Loss** | 10-30 points | 50-100 points |
| **Target** | 20-50 points | 100-200 points |
| **Level 2 Importance** | Critical for entries | Supplementary only |
| **Risk Management** | Runs every 30 seconds | Runs every 5 minutes |
| **Volume Profile** | Intraday session | Multi-day composite (7 days) |
| **Key Indicator** | CVD, Order Flow | Market Profile (time at price) |

## Core Components

### 1. Market Profile Analysis

**What is Market Profile?**
- Shows how much **TIME** price spent at each level (not just volume)
- POC (Point of Control) = price where most time was spent
- Value Area = range where 70% of time was spent
- More reliable than volume profile for swing trading

**Why It Matters:**
- Institutions accumulate/distribute over time, not just volume
- Time at price = acceptance/rejection by market
- POC acts as magnet - price tends to revisit these levels

### 2. Multi-Day Volume Profile

**Composite Profile (7 days):**
- Combines volume from past week into single profile
- Identifies major support/resistance for swing trades
- POC from composite = strongest level in recent market structure

**High/Low Volume Nodes:**
- **HVNs** (High Volume Nodes) = Strong support/resistance where lots traded
- **LVNs** (Low Volume Nodes) = Weak areas, price moves fast through these

### 3. Higher Timeframe Priority

**Daily Timeframe (Most Important):**
- EMA alignment (20/50/200) determines trend
- Daily swing highs/lows are stop placement zones
- Daily close above/below key levels triggers entries/exits

**4-Hour Timeframe (Secondary):**
- Confirms daily trend
- Identifies swing structure within daily trend
- Entry zones based on 4H pullbacks

**1-Hour Timeframe (Entry Refinement):**
- Pullback to key moving averages
- Entry zones for limit orders

**5-Minute Timeframe (Trigger Only):**
- NOT used for analysis
- Only used to confirm entry trigger when in entry zone
- Micro-structure check before placing order

## Trading Philosophy

### Patience Over Speed

```
Day Trading Mindset: "Get in, get out, capture quick moves"
Swing Trading Mindset: "Wait for setup, enter patiently, let it run for days"
```

**Key Principles:**

1. **Let Winners Run**
   - Don't exit after a few hours of consolidation
   - Swing moves take 3-7 days to fully play out
   - Trail stops based on DAILY swing lows, not 5-minute action

2. **Give It Room to Breathe**
   - Intraday drawdown of 10-20 points is NORMAL
   - Don't panic on hourly weakness if daily structure intact
   - Stops at daily swing levels, not arbitrary points

3. **Higher Quality Setups**
   - Fewer trades (maybe 2-4 per month)
   - Each trade aims for 100+ points
   - Only trade when daily AND 4H aligned

4. **Structure Over Noise**
   - Ignore intraday chop
   - Focus on daily candle closes
   - React to daily EMA crosses, not 5-min patterns

## Risk Management - Swing Version

### Rules Are LOOSER Than Day Trading

**Why?**
- Swing trades need room for normal market oscillation
- Can't judge a swing trade by its first few hours
- Daily volatility of 20-30 points is expected

### Stop Loss Guidelines

```typescript
Minimum Stop: 50 points
Typical Stop: 60-80 points
Maximum Stop: 100 points
```

**Stop Placement:**
- Below daily swing low (for longs)
- Above daily swing high (for shorts)
- Below composite VAL (for longs above composite POC)
- NOT based on ATR or arbitrary percentage

**Trailing Stops (Patient Approach):**
- After 30+ points profit → Move to breakeven
- After 60+ points profit → Trail to +40 points (lock 67%)
- After 100+ points profit → Trail to +70 points (lock 70%)
- Use DAILY swing lows/highs as trail levels

### Target Management

**Initial Targets:**
- Minimum: 50 points (1:1 risk/reward with 50pt stop)
- Typical: 100 points (2:1 risk/reward)
- Aggressive: 150-200 points (3:1+ risk/reward)

**Target Placement:**
- Next major daily swing level
- Composite POC (if approaching from below/above)
- Major HVN (high volume node from multi-day profile)
- Daily resistance/support from previous week

**When to Extend Targets:**
- Daily trend accelerating (breaking key levels)
- Already in 50+ point profit with stop protecting gains
- Cleared composite VAH/VAL and heading to next level

### Position Sizing

**More Conservative Than Day Trading:**

```typescript
Day Trading: Risk 0.5-1% per trade
Swing Trading: Risk 0.25-0.5% per trade
```

**Why Smaller Size?**
- Holding overnight = gap risk
- Larger stop losses = need smaller position to maintain same dollar risk
- Fewer trades = can't diversify risk across many setups

## Entry Process

### Step 1: Daily Analysis (Most Important)

✅ **Daily Trend Checklist:**
- [ ] Daily close above/below all major EMAs?
- [ ] Daily swing structure bullish/bearish?
- [ ] Daily candles showing higher highs/higher lows (or opposite)?
- [ ] Volume expanding on trend moves?

### Step 2: Multi-Day Profile Check

✅ **Volume Profile Alignment:**
- [ ] Is price at composite POC (balance) or extremes (directional)?
- [ ] Are we at major HVN (resistance) or LVN (breakout zone)?
- [ ] Where does composite VAH/VAL sit relative to price?

### Step 3: 4-Hour Confirmation

✅ **4H Structure:**
- [ ] Does 4H trend match daily trend?
- [ ] Is there a clear swing structure on 4H?
- [ ] Are we in a pullback zone or breakout zone?

### Step 4: Entry Zone (1-Hour)

✅ **Entry Refinement:**
- [ ] Has price pulled back to 1H EMA 20?
- [ ] Is there support/resistance at this level?
- [ ] Is entry zone aligned with daily swing low/high?

### Step 5: Trigger (5-Minute)

✅ **Final Confirmation:**
- [ ] 5-minute structure showing entry pattern?
- [ ] NOT about perfect candle patterns - just confirmation
- [ ] Place limit order in entry zone

## Example Swing Trade

### Long Setup Example

**Daily Analysis:**
- MES daily trend: Bullish (price above EMA 20/50/200)
- Recent daily swing low: 5850
- Daily structure: Higher highs, higher lows

**Multi-Day Profile:**
- Composite POC: 5900
- Composite VAH: 5920
- Composite VAL: 5880
- Current price: 5885 (pulled back to VAL)

**4-Hour:**
- Pulled back from 5935 to 5885
- 4H EMA 20 at 5880
- Swing low on 4H: 5875

**Entry:**
- Wait for price in 5880-5890 zone (near VAL and 4H EMA)
- Confirm with 5-min showing consolidation/reversal
- Enter LONG at 5885

**Risk Management:**
- Stop: 5850 (below daily swing low) = 35 points
- Target: 5970 (next daily resistance / previous high) = 85 points
- Risk/Reward: 1:2.4
- Position size: Risk $100 on $400 account = 0.25% = 2 micro contracts

**Management:**
- Day 1: Price at 5895 (+10 pts) → HOLD (too early, daily structure intact)
- Day 2: Price at 5910 (+25 pts) → HOLD (approaching composite VAH, let it run)
- Day 3: Price at 5945 (+60 pts) → ADJUST STOP to 5920 (+35 pts locked)
- Day 4: Price at 5965 (+80 pts) → Near target, trail stop to 5940 (+55 pts)
- Day 5: Price hits 5970 target → CLOSE, +85 points

## Dashboard Features

### Real-Time Display (Port 3350)

**Multi-Timeframe Trends:**
- Daily: Bullish/Bearish/Neutral
- 4-Hour: Bullish/Bearish/Neutral
- 1-Hour: Bullish/Bearish/Neutral
- 5-Min: Current micro-structure

**Market Profile Visualization:**
- Time-at-price bars showing POC
- Value Area High/Low
- Visual representation of where price spent time

**Multi-Day Volume Profile:**
- 7-day composite POC/VAH/VAL
- High Volume Nodes (support/resistance)
- Low Volume Nodes (breakout zones)

**Position Monitoring:**
- Current swing position details
- Days in trade
- Unrealized P&L
- Stop/Target levels

**Risk Management Panel:**
- Latest risk management decision
- Reasoning (focused on daily structure)
- Recommended adjustments

## Running the Swing Trading Agent

### Prerequisites

```bash
# Environment variables
TOPSTEPX_API_KEY=your_key
TOPSTEPX_API_SECRET=your_secret
TOPSTEPX_ACCOUNT_ID=your_account_id
TOPSTEPX_CONTRACT_ID_MES=contract_id
OPENAI_API_KEY=your_deepseek_key
OPENAI_BASE_URL=https://api.deepseek.com
```

### Start the Agent

```bash
# Run directly
npx tsx live-fabio-swing-mes.ts

# Or with PM2
pm2 start ecosystem.config.js --only swing-mes-trading

# View logs
pm2 logs swing-mes-trading

# Monitor dashboard
open http://localhost:3350
```

### Configuration

**Key Parameters (in live-fabio-swing-mes.ts):**

```typescript
const SYMBOL = 'MES';
const ANALYSIS_INTERVAL_MS = 300_000; // 5 minutes
const SWING_LOOKBACK_DAYS = 7; // Analyze past week
const MIN_SWING_TARGET_POINTS = 50;
const MAX_SWING_STOP_POINTS = 100;
const TYPICAL_SWING_HOLD_DAYS = 3-7;
```

## Best Practices

### DO ✅

1. **Wait for Daily Setup** - Don't force trades
2. **Use Multi-Day Profile** - Composite POC is your anchor
3. **Trail Stops Slowly** - Use daily swing lows, not hourly
4. **Hold Through Consolidation** - A few hours of chop is normal
5. **Check Market Profile** - Where did price spend time?
6. **Ignore Intraday Noise** - Focus on daily candle closes
7. **Let Winners Run** - Don't exit just because it's been 2 days

### DON'T ❌

1. **Don't Day-Trade Swing Positions** - Manage based on daily structure
2. **Don't Use Tight Stops** - Minimum 50 points for swing trades
3. **Don't Panic on Hourly Weakness** - Check if daily trend broken
4. **Don't Overtrade** - Quality over quantity (2-4 trades/month is good)
5. **Don't Ignore Overnight Risk** - Size positions smaller than day trades
6. **Don't React to 5-Min Patterns** - Use 5-min only for entry trigger
7. **Don't Chase** - Wait for pullbacks to entry zones

## Monitoring & Alerts

### What to Watch

**Daily (End of Day):**
- Did daily candle close above/below key EMAs?
- Did we make new swing high/low?
- Is position still aligned with daily trend?

**Every Few Hours:**
- Check dashboard for risk management decisions
- Are we approaching target or stop?
- Any major news events coming?

**5-Minute Updates:**
- Agent automatically analyzes every 5 minutes
- Dashboard updates in real-time
- Risk management runs every 5 minutes for active positions

### Alert Conditions

Set alerts for:
- Position profit/loss exceeds +/- 50 points
- Daily close breaks key swing level
- Risk management agent recommends CLOSE_POSITION
- Position held > 7 days (consider taking profits)

## Troubleshooting

### "Agent too aggressive with exits"
→ Check if daily trend is truly broken or just intraday noise
→ Swing risk management should default to HOLD_POSITION more often

### "Stops too tight, getting stopped out"
→ Ensure stops at daily swing lows/highs, not arbitrary levels
→ Minimum 50 points for MES swing trades

### "Missing good entries"
→ Be patient - swing setups are rare (2-4 per month)
→ Check if waiting for all timeframes to align

### "Position running without adjustment"
→ This is intentional! Swing trades need room
→ Only adjust after 30+ points profit

## Performance Metrics

### Expected Statistics

**Win Rate:** 50-60% (lower than day trading due to bigger targets)
**Average Win:** 80-120 points
**Average Loss:** 40-60 points
**Profit Factor:** 2.0-3.0 (need higher to justify fewer trades)
**Trades Per Month:** 2-4 (quality over quantity)
**Max Holding Period:** 7-10 days

### Good Swing Trading Performance

```
Month: March 2025
Trades: 3
Wins: 2
Losses: 1
Avg Win: +95 points
Avg Loss: -55 points
Net P&L: +135 points on MES = $675 (at $5/point)
```

## Comparison: Day Trading vs Swing Trading

### When to Day Trade (NQ/MGC)

- You can monitor markets frequently during the day
- You prefer smaller, quicker wins
- You want to close all positions EOD (no overnight risk)
- You like high activity (many trades per day)
- You're comfortable with tight stop management

### When to Swing Trade (MES)

- You can't watch markets all day
- You prefer larger, less frequent wins
- You're OK with overnight/weekend risk
- You want fewer decisions (less screen time)
- You have patience to hold through volatility

## Conclusion

The swing trading system is designed for **patient traders** who want to capture **larger moves** with **less frequent monitoring**. By focusing on daily structure and multi-day profiles, it filters out intraday noise and aims for high-quality setups that can run for days.

**Key Takeaway:** Think in daily candles, not 5-minute bars. Be patient, give trades room to develop, and let the higher timeframe structure guide your decisions.

---

**Dashboard:** http://localhost:3350
**Symbol:** MES (Micro E-mini S&P 500)
**Timeframe:** Daily/4H/1H (5-min for entry only)
**Hold Time:** 3-7 days typical
**Analysis:** Every 5 minutes
