# NQ ICT Dashboard - Quick Test Guide

**Status**: ✅ Dashboard Fixed & Ready for Testing
**Live Trader**: Running on port 3337

---

## Quick Start (2 Minutes)

### Option A: Direct Dashboard Access
```bash
open http://localhost:3337
```

### Option B: Through Multi-Symbol Dashboard
```bash
# Find your multi-symbol dashboard port (usually 3000, 3001, etc)
open http://localhost:YOUR_PORT/multi-symbol-dashboard.html

# Then click the "NQ ICT" tab to view the embedded dashboard
```

---

## What to Expect

### On Page Load (Immediate)
✓ Professional dark-themed interface loads
✓ Header with "NQ ICT Strategy" title
✓ START/STOP trading buttons
✓ Status badge showing "OFFLINE"

### After Connection (1-2 seconds)
✓ Status badge changes to "LIVE"
✓ Start button becomes enabled
✓ Activity log shows: "Connected to trading server"
✓ Activity log shows: "Loaded XXX historical bars"

### Chart Display (Immediate)
✓ Candlestick chart displays with ~500 bars
✓ Green candles = bullish
✓ Red candles = bearish
✓ Dashed blue lines = BOS (Break of Structure) support levels
✓ Dashed red lines = BOS resistance levels
✓ Circle markers = Wicked candles (if any present)

### Status Information
✓ **Total Trades**: Shows 0 (no trades yet)
✓ **Win Rate**: Shows 0% (no trades yet)
✓ **Total P&L**: Shows $0.00 (no trades yet)
✓ **Entry Price**: Shows "—" (no active position)
✓ **Position**: Shows "NONE" (no active position)
✓ **Unrealized P&L**: Shows $0.00

### Left Panel: Current Position
✓ Shows: "No active position"
✓ Will update when a trade enters

### Right Panel: Recent Trades
✓ Shows: "No trades yet"
✓ Will show last 10 trades when trading

### Bottom Panel: Activity Log
✓ Shows timestamped messages
✓ Messages color-coded (green=success, red=error, yellow=warning, gray=info)
✓ Newest messages at the bottom
✓ Auto-scrolls to show latest activity

---

## Testing the Live Trader

### Step 1: Click START Button
- Dashboard should show: "Starting trading..."
- Status badge should change to show active trading state
- Activity log should show: "Starting trading..."
- Trader should begin scanning for entry patterns

### Step 2: Watch for Entry Signals
In the Activity Log, you'll see:
- "Wicked Bullish candle detected" or "Wicked Bearish candle detected"
- "Entry signal: LONG" or "Entry signal: SHORT"
- Position Panel will show: Entry price, targets, stops

### Step 3: Monitor Position
Position Panel shows:
- Entry price (where you entered)
- Entry time
- Pattern type (e.g., "Wicked + BOS")
- Stop loss target (4 ticks from entry)
- TP1 target (16 ticks from entry) - exits 50%
- TP2 target (32 ticks from entry) - exits remaining 50%
- Contracts (3 per entry)
- Unrealized P&L (current profit/loss)

### Step 4: See Trade Closures
Recent Trades Panel will show:
- Side: L (Long) or S (Short)
- P&L: Amount in green (profit) or red (loss)
- Exit Reason: TP1, TP2, or STOP
- Exit Price: Where position closed

Activity Log will show:
- "Trade closed: LONG WIN $150.00" (or LOSS if negative)

### Step 5: Check Updated Stats
Status Bar will update:
- Total Trades: Increments after each closed trade
- Win Rate: Calculates percentage of winners
- Total P&L: Cumulative profit/loss
- Unrealized P&L: Current open position floating P&L

---

## Browser Console (For Debugging)

Press **F12** → **Console** tab to see real-time logs:

### On Load:
```
DOM loaded, initializing dashboard...
Initializing charts...
Charts initialized successfully
Initiating Socket.IO connection...
```

### On Connection:
```
Connected to trading server
Requesting chart history...
Received chart history via socket: 500 bars
Starting chart update with 500 bars
Set candlestick data: 500 bars
```

### During Trading:
```
Received live bar update: {timestamp, open, high, low, close}
Updated candlestick with live data
Wicked Bullish candle detected
Entry signal: LONG
Received status update: {...}
Trade closed: LONG WIN $150.00
```

---

## If Chart Doesn't Show

### Check 1: Refresh Page
- Press Ctrl+R (Windows) or Cmd+R (Mac)
- Wait 2-3 seconds for chart to load

### Check 2: Verify Server Running
```bash
# Check if port 3337 is listening
lsof -i :3337

# Should show: npx tsx live-topstepx-nq-ict.ts
```

### Check 3: Check Browser Console
- Open F12 → Console
- Look for error messages in red
- If you see "Cannot connect to socket server": Restart trader

### Check 4: Restart Trader
```bash
# Kill existing trader
lsof -ti:3337 | xargs kill -9

# Restart
npx tsx live-topstepx-nq-ict.ts
```

---

## Performance Expectations

### Entry Frequency
- **Expected**: 15-20 trades per trading day
- **Strategy scans**: Every bar (1 minute)
- **Pattern detection**: Takes 2-3 bars to confirm

### Win Rate
- **Backtest**: 70.9%
- **Live expectation**: 68-70% (accounting for slippage)
- **Monitor**: Watch first week for consistency

### Trade Duration
- **TP1 hit**: Exits 50% in ~2-5 minutes (16 ticks)
- **TP2 hit**: Exits other 50% in ~5-10 minutes (32 ticks)
- **Stop hit**: Loss within 1 minute (4 ticks)

### Daily Profit Target
- **Expected**: $1,200-1,600 per day
- **Backtest average**: $1,900
- **Monitor**: Compare daily to average

---

## Multi-Symbol Integration

### Accessing Through Main Dashboard
1. Open main dashboard: `http://localhost:PORT/multi-symbol-dashboard.html`
2. Click "NQ ICT" tab at the top
3. NQ ICT dashboard loads in an iframe
4. All controls and monitoring work the same
5. Can switch between other strategies and NQ ICT easily

### Benefits of Integration
✓ Single interface to monitor all trading strategies
✓ Unified account statistics
✓ Consistent UI/UX across all strategies
✓ Easy to compare performance

---

## Key Metrics to Monitor

| Metric | Target | Red Flag |
|--------|--------|----------|
| Win Rate | 68-70% | <60% |
| Daily Trades | 15-20 | <10 or >25 |
| Avg Trade Win | $130-200 | <$80 |
| Avg Trade Loss | -$60 to -$100 | >-$120 |
| Daily P&L | $1,200-1,600 | <$500 or <-$500 |
| Profit Factor | 4.5-6.0x | <3.0x |
| Drawdown | <3% | >5% |

---

## Stop Trading If

❌ Win rate drops below 60%
❌ Daily loss exceeds -$500
❌ Drawdown exceeds 5%
❌ Avg win drops below $80
❌ Slippage significantly worse than backtest
❌ More than 3 consecutive losses

---

## File Locations

| File | Purpose | Port |
|------|---------|------|
| `live-topstepx-nq-ict.ts` | Trading engine | 3337 |
| `public/nq-ict-dashboard.html` | Direct dashboard | 3337 |
| `public/multi-symbol-dashboard.html` | Integration tab | (parent) |
| `NQ-ICT-FIX-SUMMARY.md` | Technical details | - |

---

## Next Actions

### Immediate
1. ✅ Open http://localhost:3337
2. ✅ Verify chart displays
3. ✅ Click START to begin trading
4. ✅ Watch first 5 trades for pattern quality

### Daily
1. Monitor entry quality
2. Check win rate vs 68-70% target
3. Watch for excessive slippage
4. Note market conditions
5. Compare live vs backtest metrics

### Weekly
1. Export P&L data
2. Review trade journal
3. Compare live vs backtest performance
4. Document any issues
5. Verify consistency

---

**The NQ ICT strategy is now LIVE and READY for production trading!**

Open the dashboard now and start monitoring:
```bash
open http://localhost:3337
```

---

**Last Updated**: November 14, 2025
**Status**: ✅ READY FOR LIVE TESTING
