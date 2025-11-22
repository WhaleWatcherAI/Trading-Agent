# Fabio Dashboard Integration - Quick Start Guide

This guide will get you up and running with the Fabio LLM agent + NQ ICT dashboard integration in 5 minutes.

## What You Get

✅ Real-time price chart with orderbook overlay
✅ CVD (Cumulative Volume Delta) chart
✅ Fabio's positions displayed live
✅ Trade history with win/loss tracking
✅ Activity log showing LLM decisions
✅ Account statistics (win rate, P&L, etc.)

All using the existing NQ ICT dashboard UI, but powered by Fabio's LLM trading decisions instead of ICT strategy logic.

## Prerequisites

- Node.js installed
- Python 3.9+ with pip
- TopstepX account and API credentials in .env
- All Fabio agent Python files (engine.py, topstep_client.py, etc.)

## Installation (One Time)

### 1. Install Python Dependencies

```bash
pip3 install "python-socketio[asyncio_client]" aiohttp
```

That's it! The Node.js dependencies should already be installed.

## Usage

### Step 1: Start the Dashboard Server

Open a terminal and run:

```bash
npx tsx live-topstepx-nq-ict.ts
```

You should see:
```
===================================================================================
TOPSTEPX LIVE NQ ICT/SMC STRATEGY - WEBSOCKET STREAMING
===================================================================================
Symbol: NQZ5
Dashboard: http://localhost:3337
===================================================================================
```

**Keep this terminal open.**

### Step 2: Open the Dashboard in Browser

Open your browser to:
```
http://localhost:3337
```

You should see the dashboard with a live chart, but no positions yet.

### Step 3: Test the Integration (Optional)

In a new terminal, run the test script to verify everything works:

```bash
python3 test_dashboard_bridge.py
```

This will:
- Connect to the dashboard
- Send simulated positions and trades
- Display them on the dashboard in real-time

Check your browser - you should see positions, trades, and logs appearing!

**Press Ctrl+C to stop the test.**

### Step 4: Run Fabio with Dashboard

In a new terminal, run Fabio with dashboard integration:

```bash
python3 fabio_dashboard.py --symbol NQZ5 --mode paper_trading
```

For live trading:
```bash
python3 fabio_dashboard.py --symbol NQZ5 --mode live_trading
```

You should see:
```
================================================================================
FABIO LLM TRADING AGENT - DASHBOARD INTEGRATION
================================================================================
Symbol: NQZ5
Mode: paper_trading
Dashboard: http://localhost:3337
================================================================================
[10:32:15][Fabio] Fabio agent started
[10:32:16][Fabio] Trading NQZ5 in paper_trading mode
[DashboardBridge] ✅ Connected to dashboard at http://localhost:3337
[10:32:16][Fabio] Fabio agent connected
```

**Check your browser** - you should now see:
- Fabio's activity logs appearing in real-time
- Position updates when Fabio enters trades
- Trade history when positions close
- Account statistics updating

## What You'll See in the Dashboard

### 1. Real-Time Chart
- 1-minute candlesticks with live updates
- Level 2 orderbook overlay (green bids, red asks)
- Volume profile bars on the left
- CVD chart below main chart

### 2. Current Position Panel
When Fabio opens a position, you'll see:
```
Entry: $21000.00          Pattern: Fabio LLM Decision
Stop Loss: $20996.00      TP1/TP2: $21004.00 / $21008.00
Contracts: 3              Unrealized: $50.00
```

### 3. Recent Trades Panel
List of last 10 closed trades:
```
L  +$160  TP1 @ $21004.00
S  -$80   STOP @ $20998.00
L  +$240  TP2 @ $21008.00
```

### 4. Activity Log
```
[10:32:15] Fabio agent started
[10:32:16] Trading NQZ5 in paper_trading mode
[10:32:20] Price: 21000.00
[10:32:25] Requesting LLM decision...
[10:32:28] LLM response received
[10:32:28] Position opened: LONG 3 @ 21000.00
```

### 5. Status Bar (Top)
```
Total Trades: 5    Win Rate: 80%    Total P&L: $640
Entry Price: $21000.00    Position: LONG    Unrealized P&L: $50
```

## Troubleshooting

### Problem: Dashboard shows "OFFLINE"

**Cause:** Dashboard can't connect to the Node server.

**Solution:**
1. Make sure `npx tsx live-topstepx-nq-ict.ts` is running
2. Check the terminal for errors
3. Try opening http://localhost:3337 directly

### Problem: Python script can't connect

```
[DashboardBridge] Failed to connect: Connection refused
```

**Solution:**
1. Start the Node server FIRST
2. Wait 5 seconds for it to fully initialize
3. Then start the Python script

### Problem: No positions showing

**Cause:** Fabio hasn't made any trading decisions yet.

**Solution:**
1. Wait for market data to stream in
2. Check Python console for log messages like "Requesting LLM decision..."
3. Fabio makes decisions based on market conditions, so it may take time
4. Try the test script first to verify the dashboard works

### Problem: npm/node not found

**Solution:**
```bash
# Install Node.js from https://nodejs.org/
# Or use nvm:
nvm install node
nvm use node
```

### Problem: Python module not found

```
ModuleNotFoundError: No module named 'socketio'
```

**Solution:**
```bash
pip3 install "python-socketio[asyncio_client]" aiohttp
```

## File Overview

| File | Purpose |
|------|---------|
| `dashboard_bridge.py` | Socket.IO client that connects to Node server |
| `fabio_dashboard.py` | Integration layer that runs Fabio + emits to dashboard |
| `test_dashboard_bridge.py` | Test script to verify the integration works |
| `live-topstepx-nq-ict.ts` | Node server (provides chart data + serves dashboard) |
| `public/nq-ict-dashboard.html` | Dashboard UI (browser interface) |

## Architecture

```
Browser (Dashboard)
        ↓
   Socket.IO
        ↓
Node.js Server (Chart Data + Market Depth)
        ↓
   Socket.IO
        ↓
Python Bridge (dashboard_bridge.py)
        ↓
Fabio Agent (fabio_dashboard.py)
        ↓
   LLM + TopstepX
```

## Next Steps

1. **Customize the Dashboard**
   Edit `public/nq-ict-dashboard.html` to add more metrics or charts

2. **Add Control Panel**
   Extend the bridge to listen to `start_trading` / `stop_trading` events from the dashboard

3. **Multi-Symbol Support**
   Run multiple Fabio agents, each connected to its own dashboard port

4. **Add Alerts**
   Use the `log` event to send browser notifications when Fabio opens/closes positions

5. **Historical Analysis**
   Store all trades in a database and create a separate analytics dashboard

## Configuration

### Change Dashboard Port

1. Set environment variable:
   ```bash
   export TOPSTEPX_NQ_ICT_DASHBOARD_PORT=3338
   ```

2. Update Python command:
   ```bash
   python3 fabio_dashboard.py --dashboard-url http://localhost:3338
   ```

### Change Status Update Frequency

Edit `fabio_dashboard.py`:
```python
self.status_update_interval = 2.0  # Update every 2 seconds
```

### Change Trading Symbol

```bash
python3 fabio_dashboard.py --symbol ESZ5 --mode paper_trading
```

Or set in .env:
```
TRADING_SYMBOL=ESZ5
```

## Support

For detailed documentation, see:
- **Full Guide:** [FABIO_DASHBOARD_INTEGRATION.md](FABIO_DASHBOARD_INTEGRATION.md)
- **Dashboard Events:** Check browser DevTools → Network → WS for real-time events
- **Python Logs:** Check the terminal running `fabio_dashboard.py`
- **Node Logs:** Check the terminal running `npx tsx live-topstepx-nq-ict.ts`

## Tips

1. **Always start the Node server first**, then the Python agent
2. **Check both terminals** for errors - Python terminal shows Fabio's decisions, Node terminal shows market data
3. **Use the test script** (`test_dashboard_bridge.py`) to verify integration before running Fabio
4. **Keep the browser open** to see real-time updates
5. **Check the Activity Log** in the dashboard to see what Fabio is doing

Enjoy trading with Fabio! 🚀
